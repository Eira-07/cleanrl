# DQN vs PPO：对比学习笔记

> 基于 CleanRL 的 `dqn.py` 和 `ppo.py`，对比两种算法的核心思想与实现差异

---

## 一、一句话总结

| | PPO | DQN |
|---|---|---|
| **类型** | 策略梯度方法（Policy-based） | 价值学习方法（Value-based） |
| **学什么** | $\pi(a\|s)$ 动作概率分布 | $Q(s,a)$ 动作价值函数 |
| **怎么决策** | 从概率分布采样（随机） | 选 Q 值最大的动作（确定） |
| **数据来源** | On-policy（用完就丢） | Off-policy（存到回放缓冲区重复用） |

---

## 二、核心思想对比

### PPO：学"怎么走"

```
输入：状态 → 输出：每个动作的概率
         π(左)=0.7, π(右)=0.3

决策：按概率采样 → 70% 左，30% 右

更新：好的动作提高概率，差的动作降低概率
      限制：一次不能改太多（clip）
```

思想：**像一个运动员，不断微调自己的策略，一步一步改进。**

### DQN：学"值多少钱"

```
输入：状态 → 输出：每个动作的 Q 值
         Q(左)=3.2, Q(右)=1.5

决策：选 Q 值最大的 → 100% 左

更新：让 Q(s,a) 逼近 r + γ·max Q(s',a')
      限制：用 Target Network 提供稳定目标
```

思想：**像一个评分系统，给每个动作打分，选最高分的做。**

---

## 三、五种核心机制对比

### 1. 网络架构

| | PPO | DQN |
|---|---|---|
| **网络数量** | 2 个（Actor + Critic） | 1 个（Q Network） |
| **输出** | Actor: 动作概率分布；Critic: 标量 V(s) | 每个动作的 Q 值 |
| **网络结构** | 64→64（两层隐藏层） | 120→84（两层隐藏层） |

PPO 代码：

```python
class Agent(nn.Module):
    self.critic = ...  # 输出 1 个值：V(s)
    self.actor = ...   # 输出 n 个 logits：π(a|s)
```

DQN 代码：

```python
class QNetwork(nn.Module):
    self.network = ...  # 输出 n 个 Q 值：Q(s,a)
```

### 2. 训练数据来源

| | PPO | DQN |
|---|---|---|
| **方式** | On-policy | Off-policy |
| **数据复用** | 学完就丢 | 存到 Replay Buffer，反复用 |
| **并行环境** | 4 个 | 1 个 |
| **batch_size** | 512 | 128 |

PPO 的数据流：

```
收集 512 条 → 学 4 遍 → 丢掉 → 重新收集 512 条 → ...
```

DQN 的数据流：

```
记忆库 [10000 条经验]
      ↓
每次随机抽 128 条来学 → 学完放回去 → 下次再抽别的
```

### 3. 稳定训练的方法

| | PPO | DQN |
|---|---|---|
| **方法** | Clip 裁剪 | Target Network |
| **原理** | 限制 `ratio` 在 [0.8, 1.2] | 用延迟更新的"慢网络"提供稳定目标 |
| **更新频率** | 每步都生效 | 每 500 步同步一次 |

PPO 的 clip（第 264-267 行）：

```python
ratio = newlogprob / old_logprob
pg_loss2 = -advantage * clamp(ratio, 1-clip_coef, 1+clip_coef)
```

DQN 的 Target Network（第 197-199 行 + 第 215-219 行）：

```python
# 使用 Target Network 计算目标值（不反向传播）
target_max, _ = target_network(next_obs).max(dim=1)
td_target = rewards + gamma * target_max * (1 - dones)

# 每 500 步更新 Target Network
if global_step % 500 == 0:
    target_network.weight = tau * q_network.weight
                         + (1-tau) * target_network.weight
```

### 4. 探索策略

| | PPO | DQN |
|---|---|---|
| **方法** | Entropy 奖励 | ε-greedy |
| **机制** | 损失函数加 `-ent_coef × entropy` | 以概率 ε 取随机动作 |
| **参数** | `ent_coef=0.01` | `ε: 1.0 → 0.05` |

PPO 的探索（第 285 行）：

```python
loss = pg_loss - ent_coef * entropy + v_loss * vf_coef
#        ↑ 熵越大，损失越小 = 鼓励探索
```

DQN 的探索（第 165-170 行）：

```python
epsilon = linear_schedule(1.0, 0.05, ...)  # ε 从 1 逐渐降到 0.05
if random.random() < epsilon:
    actions = 随机动作   # 探索
else:
    actions = argmax(Q(s))  # 利用
```

### 5. 学习目标

| | PPO | DQN |
|---|---|---|
| **优化目标** | Clipped Surrogate Objective | TD Target |
| **核心公式** | $\min(r(\theta)\hat{A}, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A})$ | $r + \gamma \max Q(s', a') - Q(s, a)$ |
| **损失函数** | MSE | MSE |

DQN 的 TD Target（第 197-201 行）：

```python
td_target = rewards + gamma * target_max * (1 - dones)  # Bootstrap 目标
loss = F.mse_loss(q_network(obs).gather(1, actions), td_target)  # 让 Q 逼近目标
```

---

## 四、完整参数对比

| 参数 | PPO 默认值 | DQN 默认值 | 说明 |
|---|---|---|---|
| `total_timesteps` | 500000 | 500000 | 训练预算 |
| `learning_rate` | 2.5e-4 | 2.5e-4 | 学习率 |
| `num_envs` | 4 | 1 | 并行环境数 |
| `gamma` | 0.99 | 0.99 | 折扣因子 |
| `batch_size` | 512（自动计算） | 128 | 每批训练数据量 |

### DQN 特有的参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `buffer_size` | 10000 | Replay Buffer 最大容量 |
| `tau` | 1.0 | Target Network 更新系数 |
| `target_network_frequency` | 500 | Target Network 更新间隔（步数） |
| `start_e` | 1.0 | ε-greedy 起始 ε 值 |
| `end_e` | 0.05 | ε-greedy 最终 ε 值 |
| `exploration_fraction` | 0.5 | ε 从 start_e 降到 end_e 占总步数的比例 |
| `learning_starts` | 10000 | 开始学习的步数（之前只收集数据） |
| `train_frequency` | 10 | 每隔多少步训练一次 |

---

## 五、训练过程对比

### 输出频率

```
PPO：每 512 步（一个 iteration）打印一次
     global_step=512, episodic_return=[32.]

DQN：每局结束打印一次（步数不固定）
     global_step=134, episodic_return=[11.]
     global_step=256, episodic_return=[18.]
     ...（更频繁）
```

### SPS（每秒步数）

```
PPO: ~3000-3800（CPU）     → 每 512 步才做一次大更新
DQN: 远高于 PPO            → 前 10000 步不训练，之后每 10 步做一次小更新
```

### SPS 下降原因（DQN）

```
前 10000 步：只收集数据，不训练     → SPS 最高 🚀
10000 步后：每 10 步训练 1 次       → SPS 下降 ⬇️

不是 ε 变化导致的，是 training starts！
```

### Return 曲线对比

同等步数下，PPO 通常比 DQN 学得快：

| 原因 | DQN | PPO |
|---|---|---|
| **开始学习时间** | 10000 步后才开始 | 从一开始就学 |
| **并行环境** | 1 个 | 4 个 |
| **数据复用** | 每条数据用 ≈1 次 | 每批学 4 遍 |

---

## 六、如何选择算法

### 选 PPO 的场景

- 连续动作空间（机器人控制：走路、飞行）
- 需要稳定的策略（安全关键应用）
- 并行环境容易获取（Simulation）
- 可以承受高训练成本

### 选 DQN 的场景

- 离散动作空间（游戏、导航）
- 样本效率重要（数据收集成本高）
- 问题有明确的 Q 值意义（评分明确）
- 需要 off-policy 的数据复用

### 选 SAC/TD3 的场景（连续动作的高级选择）

- 连续动作空间
- 需要比 PPO 更高的样本效率
- 对超参数敏感度低

---

## 七、总结

```
PPO                            DQN
──────────                     ──────────
"过程"导向                     "结果"导向
学的是动作的概率分布            学的是动作的价值评分
用 clip 保证稳定              用 Target Network 保证稳定
用 entropy 探索              用 ε-greedy 探索
on-policy 一次性数据          off-policy 重复用数据
4 个环境并行                  1 个环境单步
Actor + Critic 两个网络       一个 Q 网络
适合连续控制                  适合离散决策
```

**联系**：两者都使用 **Bootstrap**（自举），都用 **TD Target** 的思想，都需要在**探索与利用**之间权衡。
