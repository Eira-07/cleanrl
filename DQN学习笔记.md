# DQN 学习笔记

> 基于 CleanRL 的 `dqn.py` 逐行学习总结

---

## 一、DQN 是什么

DQN（Deep Q-Network，深度 Q 网络）是一种**价值学习**强化学习算法。它的核心思想是：**学一个 Q 函数 $Q(s,a)$，评估"在状态 s 下做动作 a 有多好"，然后每次都选 Q 值最大的动作。**

CleanRL 的 `dqn.py` 是一个 **249 行** 的完整实现。

---

## 二、DQN 的网络架构：单一 Q 网络

DQN 只有一个神经网络，和 PPO 的 Actor-Critic 双网络形成鲜明对比：

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        self.network = nn.Sequential(
            nn.Linear(4, 120),       # 输入：4 维状态（CartPole）
            nn.ReLU(),                # 激活函数用 ReLU
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),         # 输出：2 个 Q 值（左/右）
        )

    def forward(self, x):
        return self.network(x)        # 形状：[batch, 4] → [batch, 2]
```

### 和 PPO 网络对比

| | PPO | DQN |
|---|---|---|
| **网络数量** | Actor + Critic 两个 | 一个 Q 网络 |
| **输出** | Actor: 动作 logits；Critic: 标量 V(s) | 每个动作的 Q 值 |
| **激活函数** | Tanh | ReLU |
| **隐藏层** | 64 → 64 | 120 → 84 |

### Q 值的含义

```
CartPole 状态 [x, v, θ, ω]
         ↓ Q 网络
输出 [Q(左), Q(右)] = [3.2, 1.5]
                        ↑
                  Q(左) > Q(右) → 选左边
```

Q(s,a) 的意思是：**在状态 s 下做动作 a，未来累计奖励的期望值。**

---

## 三、DQN 的五大核心机制

### 机制 1：ε-greedy 探索

#### ε 调度函数（第 106-108 行）

```python
def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
```

ε 从 `start_e=1.0` 线性下降到 `end_e=0.05`，在 `exploration_fraction=0.5`（一半的训练步数）内完成。

```
ε
1.0 ┤████████████████ 全随机探索
    │    ██████████
0.5 ┤        ██████    半随机半利用
    │            ███
0.05┤              ██  基本利用，偶尔探索
    ┼──────────────────→ 训练步数
    0          250k    500k
```

#### 动作选择（第 165-170 行）

```python
epsilon = linear_schedule(1.0, 0.05, 250000, global_step)
if random.random() < epsilon:
    # 探索：随机动作
    actions = envs.single_action_space.sample()
else:
    # 利用：选 Q 值最大的
    q_values = q_network(torch.Tensor(obs))
    actions = torch.argmax(q_values, dim=1).cpu().numpy()
```

**argmax** 是关键——Q 值最大的动作被确定性地选中（不像 PPO 按概率采样）。

与 PPO 的对比：

| | PPO | DQN |
|---|---|---|
| **探索方式** | Entropy 奖励（软性） | ε-greedy（硬性） |
| **机制** | 损失函数加 `-ent_coef × entropy` | 以概率 ε 取随机动作 |
| **参数** | `ent_coef=0.01` | `ε: 1.0 → 0.05` |

---

### 机制 2：Replay Buffer（经验回放）

#### 初始化（第 152-158 行）

```python
rb = ReplayBuffer(
    args.buffer_size,      # 最多存 10000 条
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=False,
)
```

#### 存数据（第 188 行）

```python
rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
# 存储 (s, a, r, s', done) 到记忆库
```

#### 取数据（第 196 行）

```python
data = rb.sample(args.batch_size)
# 从 10000 条中随机抽取 128 条
# 包含: data.observations, data.actions, data.rewards,
#       data.next_observations, data.dones
```

**这就是 off-policy 的本质**：数据是谁产生的并不重要，只要存进了 buffer，后面的训练都能用。一条数据可以被反复抽取学习。

和 PPO 对比：

```
PPO：收集 512 条 → 学 4 遍 → 丢掉 → 重新收集（on-policy）
DQN：收集 1 条 → 存进 buffer → 从 buffer 抽 128 条来学 → 放回去（off-policy）
```

---

### 机制 3：TD Target（Bootstrap 目标）

这是 DQN 最核心的四行代码（第 197-201 行）：

```python
with torch.no_grad():
    # 1. 用 Target Network 算下一步的 Q 值，取最大值
    target_max, _ = target_network(data.next_observations).max(dim=1)

    # 2. 计算 TD Target = 当前奖励 + γ × 未来最佳 Q 值
    td_target = data.rewards.flatten()
               + args.gamma * target_max * (1 - data.dones.flatten())

# 3. 拿出 Q 网络对"实际执行的动作"的 Q 值
old_val = q_network(data.observations).gather(1, data.actions).squeeze()

# 4. 计算损失：让 Q(s,a) 逼近 TD Target
loss = F.mse_loss(td_target, old_val)
```

#### Bootstrapping（自举）

```
Q(s,a) ───── 当前估计 ────→ TD Target
  │                            ↑
  │                     r + γ·max Q'(s',a')
  │                         ↑
  └── 用"自己"的估计来更新自己 ──┘
       这就是 Bootstrap！
```

用 CartPole 来理解：

```
状态 s，动作"左移"
  → reward = 1
  → 到下一状态 s'

Q_target = 1 + 0.99 × max(Q(s', 左), Q(s', 右))

如果 s' 的 Q 值高 → "左移"是个好决策 → Q(s, 左) 上调
如果 s' 的 Q 值低 → "左移"不好 → Q(s, 左) 下调
```

#### `data.dones` 的作用

```python
(1 - data.dones.flatten())  # 如果 done=1（游戏结束），这项=0
```

游戏结束时没有"下一步"，所以 `Q_target = reward`（没有未来奖励）。

---

### 机制 4：Target Network

#### 为什么需要 Target Network？（第 149-150 行）

```python
q_network = QNetwork(envs)        # 主 Q 网络：每步都更新
target_network = QNetwork(envs)   # 目标网络：冻结更新
target_network.load_state_dict(q_network.state_dict())  # 初始相同
```

如果没有 Target Network：

```
Q(s,a) 更新 → Q(s',a') 也跟着变 → TD Target 变了
  → 目标在移动！就像"追自己的尾巴"
  → 训练不稳定 ❌
```

#### 更新方式（第 215-219 行）

```python
if global_step % args.target_network_frequency == 0:  # 每 500 步
    for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(
            args.tau * q_param.data + (1.0 - args.tau) * target_param.data
        )
```

- `tau=1.0`：完全复制 Q 网络参数到 Target 网络（硬更新）
- `target_network_frequency=500`：每 500 步更新一次

| | PPO | DQN |
|---|---|---|
| **稳定方法** | Clip 裁剪 | Target Network |
| **原理** | 限制 ratio 在 [0.8, 1.2] | 用延迟更新的网络提供稳定目标 |
| **更新频率** | 每步都生效 | 每 500 步同步一次 |

---

### 机制 5：延迟学习

```python
learning_starts: int = 10000   # 前 10000 步不训练
train_frequency: int = 10      # 每 10 步训练一次
```

**为什么需要 `learning_starts`？**

```python
if global_step > args.learning_starts:   # 前 10000 步：只填充 buffer
    if global_step % args.train_frequency == 0:  # 之后每 10 步训练一次
        data = rb.sample(args.batch_size)
        # ... 训练 ...
```

Replay Buffer 刚启动时是空的。如果立刻训练，抽到的都是重复的少量数据，学不到东西。

```
前 10000 步：往空的 buffer 里塞数据
10000 步后：buffer 里有 10000 条经验了，可以开始学
```

---

## 四、完整训练循环

```
for global_step in range(total_timesteps):
    │
    ├─ 第 1 步：ε-greedy 选择动作
    │   ├─ random() < ε → 随机动作（探索）
    │   └─ 否则 → argmax Q(s)（利用）
    │
    ├─ 第 2 步：env.step(action)，得到 (s', r, done)
    │
    ├─ 第 3 步：存到 Replay Buffer (s, a, r, s', done)
    │
    ├─ 第 4 步：如果 global_step > learning_starts（10000步后）：
    │   └─ 每 train_frequency（10）步训练一次：
    │       ├─ 从 buffer 随机抽 128 条
    │       ├─ 用 Target 网络算 TD Target
    │       ├─ 算 MSE 损失
    │       └─ 反向传播更新 Q 网络
    │
    └─ 第 5 步：每 target_network_frequency（500）步：
        └─ 复制 Q 网络参数到 Target 网络
```

---

## 五、全部参数详解

### 第一组：基础设置

| 参数 | 默认值 | 作用 |
|---|---|---|
| `env_id` | CartPole-v1 | 训练的游戏环境 |
| `total_timesteps` | 500000 | 总训练步数（预算） |
| `learning_rate` | 2.5e-4 | 学习率 |
| `num_envs` | 1 | 强制单环境（不支持并行） |
| `gamma` | 0.99 | 折扣因子 |

### 第二组：Replay Buffer

| 参数 | 默认值 | 作用 |
|---|---|---|
| `buffer_size` | 10000 | 记忆库容量 |
| `batch_size` | 128 | 每次训练抽取的经验条数 |

### 第三组：Target Network

| 参数 | 默认值 | 作用 |
|---|---|---|
| `tau` | 1.0 | 更新系数（1.0 = 硬更新，< 1.0 = 软更新） |
| `target_network_frequency` | 500 | 更新间隔（步数） |

### 第四组：ε-greedy 探索

| 参数 | 默认值 | 作用 |
|---|---|---|
| `start_e` | 1.0 | ε 初始值（100% 探索） |
| `end_e` | 0.05 | ε 最终值（5% 探索） |
| `exploration_fraction` | 0.5 | ε 衰减所需步数占比 |

### 第五组：训练控制

| 参数 | 默认值 | 作用 |
|---|---|---|
| `learning_starts` | 10000 | 开始学习的步数（之前只收集数据） |
| `train_frequency` | 10 | 每隔多少步训练一次 |

---

## 六、TensorBoard 曲线解读

| 曲线 | 正常形态 | 说明 |
|---|---|---|
| `charts/episodic_return` | 从低到高上升 | 学习效果 |
| `losses/td_loss` | 从高到低下降 | Q 值在逼近 TD Target |
| `losses/q_values` | 逐渐变大 | Q 网络学会了 Q(s,a) 越来越大 |
| `charts/SPS` | 前 1 万步高，之后下降 | learning_starts 生效 |

### DQN vs PPO 的 TensorBoard 差异

| 曲线 | PPO | DQN | 说明 |
|---|---|---|---|
| `entropy` | ✅ | ❌ | DQN 用 ε-greedy，没有熵 |
| `approx_kl` | ✅ | ❌ | DQN 没有 clip |
| `clipfrac` | ✅ | ❌ | DQN 没有 clip |
| `explained_variance` | ✅ | ❌ | DQN 没有 Critic |
| `td_loss` | ❌ | ✅ | DQN 特有 |
| `q_values` | ❌ | ✅ | DQN 特有 |

---

## 七、DQN 的公式总结

### Q 值更新（核心公式）

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

### TD Target

$$y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

### 损失函数

$$\mathcal{L} = \mathbb{E}_{(s,a,r,s') \sim \text{Buffer}} \left[ \left( y - Q(s,a) \right)^2 \right]$$

### ε-greedy

$$\pi(a|s) = \begin{cases} \text{random}, & \text{with probability } \varepsilon \\ \arg\max_a Q(s,a), & \text{with probability } 1-\varepsilon \end{cases}$$
