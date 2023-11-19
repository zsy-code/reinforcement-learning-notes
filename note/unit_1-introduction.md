# 深度强化学习简介
> - 基础知识
> - 训练一个agent
> - 将agent 上传到hugging face

深度强化学习属于机器学习的一个分支，它可以使一个智能体/代理 `agent` 通过 `perform actions` 和 `observe results` 来在一个 `environment` 中作出后续行动，从而最大化环境 `reward`

## 什么是深度强化学习（Deep Reinforcement Learning）
强化学习背后的想法是，一个`agent` 通过与 `environment` 进行交互，从环境中获取 `reward` (positive or negative) 作为执行 `actions`的反馈，这种从与环境的交互中学习的思想来源于我们的日常生活经验。


举例：将一个小孩放到一个他从来没有玩过的video game前，只给他一个手柄而不教给他怎么去玩。那么接下来这个小孩就会通过按手柄的按键(action)与游戏机(environment)进行交互，从而获得分数(rewards)，比如当他躲避敌人的攻击/拾取金币时，分数+1，当他被敌人触碰/攻击到时，分数-1，这样的话他就会认识到，不断的获取更多的分数并且尽量减少损失的分数，就会最终获得游戏的胜利。最后这个小孩就会认识到，不断的拾取金币并且躲避敌人就会获得最高的分数。


由上面的例子可以看到，在没有任何监督的情况下，这个小孩就会把游戏玩的越来越好。


这也是我们人类或者其他动物学习的方式，即通过与环境的交互不断的进行学习。强化学习就是一种从动作（action）中进行学习的方法。


强化学习的官方定义：
> - Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.
> - 强化学习是一种通过构建能够与环境进行交互、通过试错并获得正负反馈奖励而对环境进行学习的智能体,来解决控制任务(也称为决策问题)的框架。


在这个框架中，智能体需要决定下一步的操作，以最大化其累积奖励。它不同于监督学习，因为训练数据中不提供正确的序列操作，也不同于非监督学习，因为存在奖励信号来指导学习。通过大量的环境交互，强化学习中的智能体能够逐步提升其策略，以取得更高的累积奖励。因此强化学习非常适合解决顺序决策、自动控制等问题。它已经在游戏、机器人控制、自动驾驶等领域得到成功应用。


## 深度强化学习框架

### 强化学习流程

RL 流程图
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process.jpg)

更容易理解的图像
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process_game.jpg)

> **解释如下**
> - agent 从environment 中接收 **state $S_0$** -- 从游戏中接收第一帧图像
> - 基于 **state $S_0$**，agent 作出 **action $A_0$** -- agent 向右移动
> - environment 变为新的状态 **state $S_1$** -- 新一帧图像
> - environment 给agent 提供了一些反馈 **rewards $R_1$** -- agent 没有死亡
> - 循环上述过程

RL 循环输出序列 state, action, reward, next_state
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/sars.jpg)

agent 的目标是最大化累计奖励，我们称之为预期回报（expected return）

### 奖励条件的设定是强化学习的核心思想
> The reward hypothesis: the central idea of Reinforcement Learning

为什么RL的目标是最大化expected return？

因为RL 是基于奖励假设的，奖励假设的所有目标都可以被描述为最大化expected return（expected cumulative return）

### 马尔可夫属性（Markov property）

在论文中，RL过程通常就被称为马尔可夫决策过程

> The Markov Property implies that our agent needs only the current state to decide what action to take and not the history of all the states and actions they took before

马尔可夫性质意味着 agent 只需要当前的 state 来决定采取什么 action，而不需要之前所有行动和状态的历史

### 观察空间/状态空间（Observations/State Space）

observation/state 是 agent 从 environment 中获取的信息。在一个视频游戏中，它可以是一帧图像或一张截图。在交易agent 中，它可以是股票的价值等。

observation 和 state 之间是有区别的：

- State s: 是对世界状态的完整描述（没有隐藏信息）。在一个完整的被观察到环境中。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/chess.jpg)
> 在一个棋类游戏中，我们可以纵观整个棋盘的信息，也就是可以从环境（棋盘）中接收状态，即完整的观察到环境

- Observation o: 是对状态的部分描述，在部分被观察的环境中
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/mario.jpg)
> 在马里奥中，我们仅能看到关卡中靠近玩家的部分，从而可以收到一个observation

本节课中，使用术语 state 来表示 state 和 observation，但在实际实现中仍然会进行区分

**回顾：**
![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/obs_space_recap.jpg)


未来选择 RL 算法的时候考虑这些信息是很必要的

### 动作空间（Action Scape）

概念：动作空间是环境中所有可能动作的集合

动作可能来自于离散（discrete）或者连续（continuous）空间

- 离散空间（discrete space）：可能的动作数量是有限的（比如马里奥中左右上下四个动作）
- 连续空间（continuous space）：可能的动作数量是无限的（比如自动驾驶，转弯 $n^\circ $）








