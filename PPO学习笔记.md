# PPO 学习笔记

> 基于 CleanRL 的 `ppo.py` 逐行学习总结

---

## 一、PPO 是什么

PPO（Proximal Policy Optimization，近端策略优化）是一种 **策略梯度** 强化学习算法。它的核心思想是：**每次更新策略时不要改太多**，通过 `clip` 机制限制更新幅度，保证训练稳定。

CleanRL 的 `ppo.py` 是一个 **313 行** 的完整实现，可以直接运行训练。

---

## 二、PPO 的网络架构：Actor-Critic

PPO 使用两个神经网络：

### Actor（策略网络）

```python
self.actor = nn.Sequential(
    Linear(4 → 64), Tanh(),      # 输入：状态（CartPole 是 4 维）
    Linear(64 → 64), Tanh(),      # 隐藏层
    Linear(64 → 2),               # 输出：每个动作的 logits
)
```

- 输入：状态 $s$
- 输出：动作概率分布 $\pi(a|s)$
- 初始化 `std=0.01`（策略需要从零开始探索）

### Critic（价值网络）

```python
self.critic = nn.Sequential(
    Linear(4 → 64), Tanh(),       # 输入：状态
    Linear(64 → 64), Tanh(),       # 隐藏层
    Linear(64 → 1),                # 输出：标量价值 V(s)
)
```

- 输入：状态 $s$
- 输出：状态价值 $V(s)$，评估"这个状态有多好"
- 初始化 `std=1.0`（价值范围较大）

### 核心方法

```python
def get_action_and_value(self, x, action=None):
    logits = self.actor(x)
    probs = Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    #       动作    该动作的log概率     熵（探索程度）   状态价值
```

---

## 三、PPO 训练流程

整个训练分为 3 个阶段，循环执行：

```
第 1 阶段：Rollout（收集数据）
第 2 阶段：GAE 优势计算
第 3 阶段：策略优化（多轮 minibatch 更新）
        ↓
    ←── 回到第 1 阶段 ──→
```

### 第 1 阶段：Rollout（第 192-215 行）

```python
for step in range(args.num_steps):  # 默认 128 步
    # 智能体决策
    action, logprob, _, value = agent.get_action_and_value(next_obs)
    # 环境执行
    next_obs, reward, done, ... = envs.step(action)
    # 存储 (obs, action, reward, done, logprob, value)
```

- `num_envs=4`：4 个并行环境
- `num_steps=128`：每个环境走 128 步
- `batch_size = 4 × 128 = 512`：每次收集 512 条经验

### 第 2 阶段：GAE 优势计算（第 218-231 行）

```python
advantages = torch.zeros_like(rewards)
lastgaelam = 0
for t in reversed(range(args.num_steps)):  # 从后往前算
    delta = rewards[t] + gamma * V(s_{t+1}) * nextnonterminal - V(s_t)
    advantages[t] = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
returns = advantages + values
```

- **GAE（Generalized Advantage Estimation）**：在偏差和方差之间取折中
- `gamma=0.99`：折扣因子，控制"远见"程度
- `gae_lambda=0.95`：GAE 参数
- 从后往前递推计算，确保每步的优势值包含未来信息

### 第 3 阶段：策略优化（第 244-293 行）

#### PPO 核心公式：Clipped Surrogate Objective

```python
ratio = newlogprob.exp() / b_logprobs.exp()   # 新旧策略比值

pg_loss1 = -advantage * ratio                    # 原始策略梯度
pg_loss2 = -advantage * clamp(ratio, 0.8, 1.2)   # 裁剪版本
pg_loss = max(pg_loss1, pg_loss2)                # 取更差的那个
```

**为什么需要 clip？**

传统策略梯度的问题：一步更新太大 → 策略直接崩了 → 再也回不来。

PPO 的 clip 限制了"一步能改多少"：
- `clip_coef=0.2` → ratio 限制在 [0.8, 1.2]
- 一步最多改 20%

#### 完整损失函数

```python
loss = pg_loss                # 策略损失：让好的动作更可能被选
     - ent_coef * entropy     # 熵奖励：鼓励探索（减去熵 = 最大化熵）
     + vf_coef * v_loss       # 价值损失：让 V(s) 更准确
```

其中价值损失也有裁剪版本（`clip_vloss=True`）：

```python
v_loss_unclipped = (V_new - returns)²
v_clipped = V_old + clamp(V_new - V_old, -0.2, 0.2)
v_loss_clipped = (v_clipped - returns)²
v_loss = max(v_loss_unclipped, v_loss_clipped)  # 取更大的那个
```

---

## 四、全部参数详解

### 第一组：环境与训练规模

| 参数 | 默认值 | 作用 |
|---|---|---|
| `env_id` | `CartPole-v1` | 训练的游戏环境 |
| `total_timesteps` | 500000 | 总训练步数（预算） |
| `num_envs` | 4 | 并行环境数（越大 SPS 越高） |
| `num_steps` | 128 | 每次 rollout 走多少步（决定 batch_size） |

关键公式：

```
batch_size = num_envs × num_steps
num_iterations = total_timesteps / batch_size
minibatch_size = batch_size / num_minibatches
```

### 第二组：学习动态

| 参数 | 默认值 | 作用 |
|---|---|---|
| `learning_rate` | 2.5e-4 | 更新步长，类似 PID 比例系数 |
| `anneal_lr` | True | 是否让学习率从高到低衰减 |
| `gamma` | 0.99 | 折扣因子：今天 1 元 ≈ 明天 0.99 元 |
| `gae_lambda` | 0.95 | GAE 参数：偏差-方差折中 |

**学习率衰减**（第 187-190 行）：

```python
frac = 1.0 - (iteration - 1.0) / args.num_iterations  # 从 1 降到 0
lrnow = frac * args.learning_rate
```

训练初期大步探索，训练后期小步微调——就像 PID 控制中接近目标时减小步长防止超调。

### 第三组：PPO 核心

| 参数 | 默认值 | 作用 |
|---|---|---|
| `clip_coef` | 0.2 | **PPO 的灵魂**，限制一步改多少 |
| `update_epochs` | 4 | 同一批数据学几轮 |
| `num_minibatches` | 4 | 分多少个小批次更新 |
| `norm_adv` | True | 是否标准化优势值（Z-score） |

**clip_coef 的效果对比**：

- `0.2` → 标准值，训练平稳
- `0.05` → 限制太严，学得慢
- `0.8` → 限制太松，训练震荡

**update_epochs 的过拟合问题**：
- 同一批数据学太多 → 策略记住这 512 条数据 → 遇到新状态表现差
- `4` 是平衡值，`16` 就会过拟合

### 第四组：损失权重与稳定

| 参数 | 默认值 | 作用 |
|---|---|---|
| `ent_coef` | 0.01 | 熵奖励权重（探索 vs 利用） |
| `vf_coef` | 0.5 | 价值损失在总损失中的占比 |
| `max_grad_norm` | 0.5 | 梯度限速，防止梯度爆炸 |
| `target_kl` | None | KL 散度阈值，超过则提前停止本轮更新 |
| `clip_vloss` | True | 价值损失也做裁剪 |

**熵（Entropy）的变化规律**：

```
训练初期 → 熵高（探索）：π(左)=0.5, π(右)=0.5
训练后期 → 熵低（利用）：π(左)=0.99, π(右)=0.01
```

- `ent_coef=0` → 策略坍缩，过早锁定一个方案，可能陷入局部最优
- `ent_coef=0.01` → 标准值
- `ent_coef=0.1` → 太鼓励探索，可能一直不收敛

---

## 五、TensorBoard 诊断速查表

| 症状 | 看哪个曲线 | 正常值 | 异常值 | 调什么 |
|---|---|---|---|---|
| 回报上不去 | `episodic_return` | 稳步上升 | 卡在低分不动 | 加 `total_timesteps`，检查 `lr` |
| 探索死了 | `losses/entropy` | 缓慢下降 | 突然垂直下降到 0 | 增大 `ent_coef` |
| 更新太猛 | `losses/approx_kl` | < 0.05 | > 0.1 | 减小 `clip_coef` 或 `lr` |
| 被裁太多 | `losses/clipfrac` | 0.3~0.5 | > 0.8 | 增大 `clip_coef` |
| 价值不准 | `losses/explained_variance` | > 0.5 | < 0 或接近 0 | 增大 `vf_coef` 或 `update_epochs` |
| 震荡严重 | `losses/policy_loss` | 平滑下降 | 锯齿状震荡 | 减小 `lr` 或 `clip_coef` |

### 诊断流程图

```
回报上不去？
  │
  ├─ entropy 突然降到 0？     → 探索死了（ent_coef 太低）
  │
  ├─ approx_kl > 0.1？       → 更新太猛（clip_coef 太小 或 lr 太高）
  │
  ├─ clipfrac ≈ 0？          → 绑太死了（clip_coef 太小）
  │
  ├─ 所有曲线都像慢动作？      → lr 太低
  │
  ├─ 回报涨到一半就停了？      → gamma 太低（目光短浅）
  │
  └─ 回报剧烈震荡？          → 更新步长太大（lr 或 clip_coef）
```

---

## 六、CartPole 环境

CartPole-v1 的状态是 4 个数字：

| 索引 | 含义 | 范围 |
|---|---|---|
| [0] | 小车位置 | -4.8 ~ +4.8 |
| [1] | 小车速度 | 无限制 |
| [2] | 杆子角度 (rad) | -0.42 ~ +0.42 |
| [3] | 杆子角速度 | 无限制 |

- 动作：0（左移）或 1（右移）
- 奖励：每坚持 1 步得 +1
- 满分：500（达到即通关）
- 训练到 500 分约需 10 万步

---

## 七、常见陷阱

### 1. CartPole 太简单了

即使参数设得很差，CartPole 通常也能收敛到 500 分。不要被它迷惑——真正的诊断要看 `entropy`、`approx_kl`、`clipfrac` 等辅助曲线。

### 2. 稀疏奖励环境（如 Acrobot）

Acrobot 只有成功/失败两种信号，PPO 默认参数容易卡在局部最优（如 -70 分上不去）。

解决方案：
- 加长 `num_steps` 到 512（确保 rollout 覆盖完整一局）
- Reward Shaping（奖励塑形）：给中间过程的奖励
- 换更合适的算法

### 3. 网络容量不够

复杂环境（更高维的状态/动作空间）需要更大的网络。默认 64 神经元可能不够，可以试试 128 或 256。
