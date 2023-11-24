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


### 动作空间（Action Scape）

概念：动作空间是环境中所有可能动作的集合

动作可能来自于离散（discrete）或者连续（continuous）空间

- 离散空间（discrete space）：可能的动作数量是有限的（比如马里奥中左右上下四个动作）
- 连续空间（continuous space）：可能的动作数量是无限的（比如自动驾驶，转弯 $n^\circ $）

**回顾：**

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/action_space.jpg)


**未来选择 RL 算法的时候考虑这些信息是很必要的**

### 奖励和折扣（rewards and the discounting）

奖励是 RL 的核心，因为它是给 agent 唯一的反馈，多亏了它，agent 才知道所采取的 action 是好的还是不好的。

每个time step t 的 cumulative rewards 可以写为：

$R(\tau ) = r_{t+1} + r_{t+2} + r_{t+3} + r_{t+4} + \cdots$

> cumulative rewards 等于序列中所有rewards的和

等价于：

$R(\tau ) = \sum_{k=0}^{\infty } r_{t+k+1} $

事实上，我们并不能简单的将它们进行加和，更早获得的奖励(在游戏开始时)更有可能发生,因为它们比长期未来的奖励更可预测

**这是因为随着时间的推移,环境变得更加复杂和不确定。在游戏开始阶段,环境还相对稳定和可控,所以短期内的奖励是可以合理估计的。但是长期后面的奖励依赖于许多不可控因素,是否能获得具有很大不确定性。**

**因此,我们不能简单地认为所有时刻的奖励同等重要,而应该对更早的奖励给予更高的优先级。一般来说,获取近期奖励的行为策略更有可能被强化,而仅基于不确定的远期大奖励采取行动的策略难以得到有效强化。**

**这就是强化学习中需要考虑的“时间差折扣”概念。我们需要对远期奖励进行折扣处理,以反映其不确定性。只有这样,智能体才能在当前行动selection时做出最佳决策。**

举例：我们的agent 是一只老鼠，它的对手是一只猫（可以移动），老鼠每次只能移动一个格子，它的目标是在被猫吃掉之前吃掉尽可能多的奶酪

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/rewards_3.jpg)

如图所示，吃掉距离老鼠最近的奶酪比吃掉猫附近的奶酪有更多的可能性，因为这样距离猫更远更安全，也就更容易获得较多的得分（离猫越近越危险），因此，即使猫附近的奶酪更多（有更高的分数），由于我们不确定是否能够吃掉它，它的分数也就可能会大打折扣。

为了对奖励进行 discount ，我们可以这样进行：

1. 我们定义一个折扣率 gamma，它处于0到1之间，大多数时刻位于0.95-0.99之间
    - 更大的 gamma 意味着更小的 discount，这意味着我们的 agent 关心**更多的长期奖励**
    - 另一方面，更小的 gamma 意味着更大的 discount，这意味着我们的 agent 关心**更多的短期奖励（最近的奶酪）**
2. 然后,每个奖励将按时间步长的指数折现 gamma。随着时间步长的增大,猫离我们越来越近,所以未来的奖励发生的可能性越来越小。

discounted 之后的 expected cumulative rewards 就可以写成：

$R(\tau ) = r_{t+1} + \gamma r_{t + 2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \cdots$

$R(\tau ) = \sum_{k=0}^{\infty } \gamma^k r_{t + k + 1} $


## 任务类型（type of tasks）

任务是强化学习问题的一个实例。我们有两种任务类型：情景型（episodic）和 连续型（continuing）

### 情景型任务（episodic task）

在这种任务中，我们有一个任务起点和终点（最终状态），这将创建一个情景：一个状态列表（a list of States）、动作（Actions）、奖励（Rewards）、新状态（new States）

例如，在马里奥游戏中，一个情景从新的关卡启动开始，到角色死亡或到达终点结束

### 连续型任务（continuing task）

这类任务是持续长久的没有结束状态的任务，在这类任务中，agent 必须学会如何选择最佳的 actions 并同时与 environment 进行交互

例如，一个进行自动股票交易的agent，对于这类任务，没有开始状态和结束状态，它会持续执行直到我们决定停掉它。

**回顾：**

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/tasks.jpg)

## 探索/利用权衡（The Exploration/Exploitation trade-off）

在研究解决RL 的问题之前，必须cover一个非常重要的主题：探索/利用权衡

- 探索（exploration）是通过尝试随机的action来探索environment，从而得到关于environment的更多信息
- 利用（exploitation）是利用已知的信息来最大化rewords

需要注意的是，RL 的目标是为了最大化累积期望奖励，然而我们经常会陷入一个常见的陷阱

举个例子：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/exp_1.jpg)

在这个游戏中，老鼠可以有无限个小奶酪（每个加1分），但是在迷宫的顶部，有一堆大奶酪（每个加1000分）

然而，如果我们仅专注于exploitation，agent 将永远不会到达大奶酪的区域，相反，它将只会利用最近来源的奖励，即使这个来源很小（exploitation）

但是如果agent 做了一点exploration，它就会发现大的奖励（包含大奶酪的格子）

这就是我们称之为探索/利用权衡的策略，我们需要权衡我们探索环境的多少以及利用已知环境信息的多少。

因此我们必须定义一些规则用来帮助我们控制这种权衡。

如果上面的概念依然很混乱，可以考虑一个真实的问题：选择去哪一家餐厅

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/exp_2.jpg)

- Exploitation: 你每天都去同一家你已经知道的很好吃的餐厅，但是冒着错过另一家更好的餐厅的风险
- Exploration: 尝试以前从未去过的餐厅，尽管冒着可能会有糟糕体验的风险，但是也有可能获得一次更好的美妙体验

**回顾：**

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/expexpltradeoff.jpg)


**心得：**

**强化学习中的智能体需要在环境中学习最优策略。但在训练初期,智能体对环境的知识有限。这时它就面临一个问题:是应该“探索”,也就是尝试新的行动,来获取更多环境信息;还是应该“利用”,也就是选择目前看来最优的行动,以获得最大化的即时奖励。**

**探索可以获取新的信息,但不一定能得到最高的奖励;利用可以最大化当前已知信息对应的奖励,但可能陷入局部最优,错过更好的选择。一个好的强化学习算法需要在探索和利用间达到最佳平衡。探索过少会让智能体缺乏新信息;而探索过多又会导致低效。控制这种平衡是一个关键问题。**

**总之,探索/利用权衡决定了智能体收集新信息和利用已有信息的比例。处理好这种权衡对实现最终最优策略至关重要。**


## 两个解决 RL 问题的主要方法

我们如何能够构建一个agent能勾选择合适的actions从而最大化累积期望奖励？

### 策略 $\pi $：agent 的大脑

策略 $\pi $是 agent 的大脑，它是一个可以告诉我们在给定的 state 下我们要采取什么样的 action 的函数。因此在一个给定的时间点它定义了 agent 的行为。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_1.jpg)

这个策略就是我们想要学习的函数，我们的目标是找到最优策略 $\pi^*$，agent 可以通过这个策略采取 action 从而最大化累积期望奖励，我们通过训练的方式找到这个函数 $\pi^*$

有两种方法可以训练 agent 找到最优的策略 $\pi^*$：

- 直接的方法是，直接教给 agent 在给定的 state 下采取什么样的 action：基于策略的方法（Policy-Based Methods）
- 间接的方法是，教给 agent 哪种 state 更有价值，以及采取哪种 action 可以导致更有价值的 state：基于价值的方法（Value-Based Methods）

### 基于策略的方法（Policy-Based Methods）

在基于策略的方法中，我们直接学习一个策略函数

该函数将定义每个状态到最佳响应动作的映射，或者它可以定义该状态下一组可能动作的概率分布

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_2.jpg)

如图所示，这个策略直接指示每个步骤所要采取的操作

这里有两种类型的策略：

- 确定的（Deterministic）：在给定一个 state 的情况下始终都会返回相同 action 的策略

$a = \pi (s)$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_2.jpg)


- 随机的（Stochastic）：输出 actions 的概率分布

$\pi (a|s) = P(A|s)$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy-based.png)

**回顾：**

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/pbm_1.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/pbm_2.jpg)


### 基于价值的方法（Value-Based Methods）

在基于价值的方法中，我们不是学习策略函数，而是学习一个价值函数用来将一个状态映射到处于该状态的期望值。

状态的价值是 agent 从这个状态开始根据我们的策略采取行动后所获得的期望折扣回报（expected discounted return）

按照我们的策略采取行动仅仅是意味着去往有更高价值的状态

$v_\pi (s) = \mathbb{E}(R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots | S_t = s) $

可以看到价值函数为每个可能的状态定义了价值

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/value_2.jpg)

依托于价值函数，在每一步该策略都会选择具有最大价值的状态来达到目标：-7 -6 -5 等等

**回顾：**

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/vbm_1.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/vbm_2.jpg)


## 强化学习中的“深度”（The “Deep” in Reinforcement Learning）

深度强化学习引入了深度神经网络来解决强化学习问题————因此而得名“深度强化学习”

比如将要学到的两个 value-based 算法：Q-Learning（经典强化学习算法） 和 Deep Q-Learning ，它们两个的不同在于，第一个方法中使用一种传统的算法来创建一个 Q 表格，这个表格可以帮助我们找到在每个 state 下应该采取哪种 action；而第二个方法则使用神经网络来近似 Q 值。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/deep.jpg)


## 总结

- 强化学习是一个从 actions 中进行学习的计算方法。我们构建了一个 agent 通过不断的实验和试错以及和 environment 进行交互，获取 rewards（positive or negative）作为 feedback。

- 一切 RL agent 的目标都是为了最大化 expected cumulative reward（也被称为 expected return）因为 RL 是基于奖励假设的，即所有目标都可以描述为最大化期望累积奖励。

- RL 的计算过程是一个输出一个（state，action，reward，new state）的序列的循环。

- 为了计算期望累积奖励（期望回报），我们对 rewards 进行 discount：在游戏开始时更早来到的 rewards 更有可能发生，因为它们相较于更长期的未来回报更可预测。

- 要解决一个 RL 问题，你就要找到一个最优策略。这个策略是 agent 的大脑，它会在给定一个 state 的情况下要采取什么样的 action。最优的策略就是可以提供能够最大化期望回报的 actions 的策略。

- 有两种方法找到最优策略：
    1. 通过直接训练策略：policy-based methods（基于策略的方法）
    2. 通过训练一个价值函数，该函数将告诉我们在每个状态下将获得的预期回报，并且用函数来定义我们的策略：value-based methods（基于价值的方法）

- 最后，我们讨论了深度强化学习，因为我们引入了深度神经网络来估计要采取的action（policy-based methods）或者估计一个状态的价值（value-based methods），因此而得名“深度强化学习”。


## 概念
这里是一个术语表

- Agent
    agent 从实验和试错中学习决策，并伴随着来自周围环境的奖励或者惩罚

- Environment
    environment 是一个模拟世界，在这里 agent 可以通过与之交互进行学习

- Markov Property
    这意味着 agent 要采取的 action 仅依赖于当前的 state 而不依赖于过去的 actions & states

- Observation/State
    1. State：世界状况的完整描述
    2. Observation：环境或者世界的部分描述

- Actions
    1. Discrete Actions：有限数量的 actions 例如上、下、左、右
    2. Continuous Actions：actions 的无限可能性 例如自动驾驶

- Rewards & Discounting
    1. Rewards：RL的基本要素，告诉 agent 采取的 action 是好是坏
    2. RL 算法专注于最大化**累积奖励**
    3. Reward Hypothesis：强化学习问题都可以被表述为（累积）回报的最大化
    4. Discounting 是因为一开始获得的 rewards 比长期的 rewards 更容易预测

- Tasks
    1. Episodic：有起始点和终止点
    2. Continuous：有起始点但是没有终止点

- Exploration v/s Exploitation Trade-Off
    1. Exploration：一切都是通过尝试随机的 actions 并且通过接收环境中的奖励或反馈来探索环境
    2. Exploitation：利用我们对环境的了解来获取最大 rewards
    3. Exploration-Exploitation Trade-Off：它平衡了我们对环境的探索程度和对环境的了解程度

- Policy
    1. Policy：agent 的大脑，它告诉我们在给定的 state 下要采取什么样的 action
    2. Optimal Policy：一个 agent 可以依据该策略采取 action 从而最大化 expected return，该策略通过训练学习

- Policy-Based Methods
    1. 解决 RL 问题的一种方法
    2. 在这种方法中，policy 可以直接被学习
    3. 它将每种状态以及该状态下最佳的响应行为进行映射，或者该状态下可能采取的行动集合的概率分布

- Value-Based Methods
    1. 另一种解决 RL 问题的方法
    2. 训练一个价值函数而不是一个策略，可以映射每种状态到处于该状态下的期望价值


## 进阶阅读

### Deep Reinforcement Learning

- [Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto Chapter 1, 2 and 3](http://incompleteideas.net/book/RLbook2020.pdf)

- [Foundations of Deep RL Series, L1 MDPs, Exact Solution Methods, Max-ent RL by Pieter Abbeel](https://youtu.be/2GwBez0D20A)

- [Spinning Up RL by OpenAI Part 1: Key concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

### Gym

- [Getting Started With OpenAI Gym: The Basic Building Blocks](https://blog.paperspace.com/getting-started-with-openai-gym/)

- [Make your own Gym custom environment](https://www.gymlibrary.dev/content/environment_creation/)
