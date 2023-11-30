# Q-learning 算法

## 简介

本单元我们将深入探究一种value-based强化学习算法：Q-learning，同时从头创建我们的第一个 RL agent，并在两个环境中训练它。

1. Frozen-Lake-v1（防滑版）：agent 将仅通过从冰冻的地板（F）上行走并且避开湖（H）来从开始的状态（S）达到结束的状态（G）
2. An autonomous taxi：agent 将学习从城市中进行导航以将乘客从 A 运送到 B

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/envs.gif)

具体来说，我们将：
- 学习 value-based methods
- 了解蒙特卡罗（Monte Carlo）和时序差分（Temporal Difference）学习之间的差异
- 学习并实现我们的第一个强化学习算法：Q-learning

## 两种value-based methods

在基于价值的方法中，我们学习一个价值函数用来将一个状态映射到处于该状态的期望价值。

$v_{\pi }(s)=\mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3} + \cdots | S_t = s]$

一个state的价值就是当agent从这个状态开始并依照我们的policy采取行动时，能够获得的期望折扣回报

> 但依照我们的policy采取行动是什么意思呢？毕竟在value-based方法中我们并没有一个policy，因为我们训练的是一个value function而不是一个policy。

要记得一个RL agent的目标是得到一个最优的policy $`\pi^*`$

为了找到最优的policy，我们学习了两种不同的方法：

- policy-based methods：直接训练policy来选择当给定一个state的时候应该采取什么action（或在当前state下actions的概率分布）。这种情况下我们没有一个value function。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/two-approaches-2.jpg)

这个policy采用一个state作为input并提供在当前state下应该采取什么action作为outputs（确定性策略[deterministic policy]：在给定state下输出一个action的策略，和输出一个actions的概率分布的随机性策略相反）

因此，我们不会手动定义policy的行为，而是由训练来定义。

- value-based methods：不直接训练policy，而是训练一个输出state 或 state-action pair的价值的函数。给定这个价值函数之后，我们的policy 就将会采取一个action

由于policy没有经过训练/学习，我们需要指定它的行为。例如，如果我们想要一个policy，给定一个value function，将会采取永远指向最大奖励的行为，我们将创建一个贪婪策略。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/two-approaches-3.jpg)

给定一个state，我们的action-value function输出当前状态下每个action的价值，然后，我们预定义的贪婪策略将会选择在给定state 或state-action pair情况下产生最高价值的action。

因此，无论你采取什么方法解决问题，你都会有一个策略。在value-base method下，你不用训练一个policy：你的policy只是一个简单的预先指定的函数（比如贪婪策略），它使用value action给出的值来选择对应的action。

所以不同之处在于：

- 在policy-based training中，最优的策略（$`\pi^*`$）通过直接训练policy得到

- 在value-based training中，找到一个最优的value function（$`Q^*`$ 或 $`V^*`$，后续讨论差异）会引导得到一个最优策略

$`\pi^*(s) = \arg \max_a Q^*(s,a)`$

事实上，大多数时候我们会采取Epsilon-Greedy Policy来进行探索/利用的权衡

基于上述内容，我们有两种形式的value-based function：

### the state-value function

我们在策略 $\pi$ 下编写state-value function如下：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/state-value-function-1.jpg)

$V_{\pi}(s) = \mathbf{E}_{\pi}[G_t|S_t=s]$

对于每个state，如果agent从当前state 开始，并遵循policy采取后续行动（在未来所有time steps下），state-value function将输出期望回报。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/state-value-function-2.jpg)

如果我们处于state value 为-7（在当前状态下，根据我们的policy即贪婪策略采取行动的预期回报）的状态，后续的行为应该是（右右右下下右右）

### the action-value function

在action-value function中，对于每个state-action pair，action-value function都将会输出在state下开始，采取对应action，并且后续永远遵循该policy所取得的期望回报。

在遵循policy $`\pi^*`$ 的基础上，在state s下采取action a所得到的value：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/action-state-value-function-1.jpg)

$Q_{\pi}(s,a) = \mathbf{E}_{\pi}[G_t|S_t=s, A_t=a]$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/action-state-value-function-2.jpg)

我们可以看到不同之处在于：

- 对于state-value function：我们计算state $S_t$ 的 value

- 对于action-value function：我们计算state-action pair 的 value，从而计算在该状态下采取对应行动的值

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/two-types.jpg)

无论我们采取哪种价值函数，返回值都是预期回报。

然而问题是，要计算state或state-action pair的每个值，我们需要将agent从这个state开始能获取到的所有rewards进行加和。

这可能是一个计算成本很高的过程，也是贝尔曼方程可以帮助我们的地方。


## 贝尔曼方程（The Bellman Equation）：简化我们的价值估计

贝尔曼方程简化了state value/state-action value 的计算

如果我们计算 $V(S_t)$ (the value of a state)，我们需要计算从当前state开始并且后续持续遵循policy所取得的回报（在后面的例子中，我们定义policy为贪婪策略，并且为了简化计算，不对reward进行discount）

因此为了计算 $V(S_t)$ ，我们需要计算期望回报的和：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/bellman2.jpg)

为了计算state 1的价值：如果agent从这个state开始，并且在每一个time step遵循贪婪策略（采取指向最大state值的行动）所得的reward的和

然后，为了计算 $V(S_{t+1})$ ，我们需要计算从state $S_{t+1}$ 开始计算回报：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/bellman3.jpg)

为了计算state 2的价值：如果agent从这个state开始，并且在每一个time step遵循贪婪策略（采取指向最大state值的行动）所得的reward的和

你可能会注意到，我们正在重复计算不同state的value，如果你需要为每个state/state-action value做同样的计算，那可能会十分乏味。

我们可以使用贝尔曼方程，而不是计算每个state/state-action pair的期望回报（提示：和动态规划策略很相似）

贝尔曼方程是一个递归方程，我们可以像下面这样考虑任意state的值，而不是从头开始计算每个状态的回报：

t+1 状态的即时奖励（ $R_{t+1}$ ） + 后续state的折扣奖励（ $`\gamma * V(S_{t+1})`$ ），即：

$`V_{\pi}(s) = \mathbf{E}_{\pi}[R_{t+1} + \gamma * V_{\pi }(S_{t+1})|S_t=s]`$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/bellman4.jpg)

回到上面的例子，对于state 1的value，计算过程就可以化简为：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/bellman6.jpg)

## 蒙特卡罗和时序差分学习的比较（Monte Carlo vs Temporal Difference Learning）

在深入研究 Q-learning 之前我们需要讨论的最后一件事是两种学习策略。

需要记得 RL agent 是通过和环境交互进行学习，它的思想是，给定经验并且接收奖励，agent就可以更新它的value function 或 policy。

蒙特卡罗和时序差分学习是训练value function/policy function的两种不同策略，它们都是用经验来解决 RL 问题。

蒙特卡罗在学习之前使用全部的经验集，而时序差分仅使用一个step（ $S_t, A_t, S_{t+1}, A_{t+1}$ ）进行学习, 下面通过一个value-based method 例子来进行说明。

### 蒙特卡罗：在回合结束时进行学习（Monte Carlo: learning at the end of the episode）

蒙特卡罗会等到回合结束时计算 $G_t$ (return) 并将它用作更新 $V(S_t)$ 的target

所以在更新value function之前需要一个完整的交互过程。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/monte-carlo-approach.jpg)

举例：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/MC-2.jpg)

- 总是在同样一个起点开始一个回合
- agent 通过policy 采取行动，比如，使用一个Epsilon Greedy策略（一种在探索和利用之间交替的策略）
- 获取到reward 和下一个状态
- 如果老鼠移动十步以上或者猫吃掉老鼠则终止回合
- 在回合结束时，会得到一个 State, Action, Reward, Next State的元组列表： [[State tile 3 bottom, Go Left, +1, State tile 2 bottom], [State tile 2 bottom, Go Left, +0, State tile 1 bottom]…]
- agent 将所有reward进行加和得到 $G_t$
- 通过公式更新 $V(S_t)$
- 用这些知识开始新的game

随着执行越来越多的回合，agent将学会表现的越来越好。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/MC-3p.jpg)

例如，如果我们通过蒙特卡罗训练一个state-value function：

- 初始化value-function，使每个state返回value 0
- 设置learning_rate = 0.1 & discount_rate = 1(no discount)
- 老鼠开始探索环境并采取随机action
- 老鼠行动10 steps 以上，结束回合

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/MC-4p.jpg)

- 得到state, action, rewards, next_state的一个元组列表，需要计算回报 $G_{t=0}$ ，$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots $ （为了简化，不计算折扣奖励）， $G_0 = R_1 + R_2 + R_3 + \cdots $ ， $G_0 = 1+0+0+0+0+0+1+1+0+0 = 3$
- 现在可以计算新的 $V(S_0)$ ：
    $V(S_0) = V(S_0) - lr * [G_0-V(S_0)]$
    $V(S_0) = 0 + 0.1 * [3-0]$
    $V(S_0) = 0.3$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/MC-5p.jpg)

### 时序差分学习：在每个step学习（Temporal Difference Learning: learning at each step）

时序差分学习仅等待一次交互（one step） $S_{t+1}$ 来形成TD目标，并且使用 $R_{t+1}$ 和 $`\gamma * V(S_{t+1})`$ 来更新 $V(S_{t})$ 。 这个包含TD的想法是在每个时间步更新 $V(S_{t})$ 。

但因为我们没有经历整个回合，我们就没有 $G_t$ (expected return)。相反，我们通过加和 $R_{t+1}$ 和下一个state的 discounted value来估计 $G_t$ 值。

这个过程被称为引导（bootstrapping），因为时序差分(TD)的更新部分基于现有的估计 $V(S_{t+1})$ 而不是一个完整的样本 $G_t$ 。

$V(S_t) = V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-1.jpg)

此方法称为 TD(0) 或单步TD（在每个单独step 后更新value function）

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-1p.jpg)

举例：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-2.jpg)

- 初始化value function对于每个state 返回value 0
- learning_rate = 0.1, discount_rate = 1
- 老鼠开始探索环境并采取一个随机行动：向左
- 得到一个reward $R_{t+1}=1$ 因为它吃到了一块奶酪

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-2p.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-3.jpg)

现在可以更新 $V(S_0)$ ：

New $`V(S_0) = V(S_0) + lr * [R_1 + \gamma * V(S_1) - V(S_0)]`$

New $`V(S_0) = 0 + 0.1 * [1 + 1 * 0 - 0]`$

New $`V(S_0) = 0.1`$

现在就可以用更新后的value function 来和环境进行交互了

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-3p.jpg)

### 总结：

- 使用蒙特卡罗，我们从一个完整的回合中更新value function，因此我们使用该回合的实际准确的折扣回报。
- 使用时序差分，每一步我们更新一次value function，我们将 $G_t$ 替换为叫作 TD target的估计回报。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Summary.jpg)


## 中途回顾

在深入了解 Q-learning 之前，我们总结一下之前学过的内容。

我们有两种类型的value-based function：

- state-value function：如果agent 从给定的state开始并且后续永远遵循policy 采取行动，输出所获得的期望回报。

- action-value function：如果agent 从一个给定的state 和一个给定的action 开始，并且后续永远遵循policy 采取行动，输出所获得的期望回报。

- 在value-based function中，而不是学习policy，我们手动定义一个policy并且学习value function。如果我们有了一个最优的value function，就等于有了一个最优的policy。

这里有两种方法来为一个value function学习policy：

- 蒙特卡罗方法，我们从一个完整的回合中更新value function，因此我们使用该回合的实际准确的折扣回报。
- 时序差分方法，每一步我们更新一次value function，我们将 $G_t$ 替换为叫作 TD target的估计回报。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/summary-learning-mtds.jpg)

## 介绍 Q-learning

### 什么是 Q-learning

Q学习(Q-Learning)是一个基于时序差分(TD)方法训练其动作价值函数的离策略(off-policy)基于价值的(value-based)方法

> Q-Learning is an off-policy value-based method that uses a TD approach to train its action-value function
> - "off-policy" 离策略 - 表示Q学习可以在行动策略和行动价值函数更新策略无关的情况下工作。也就是说,它可以对一个不同的策略学习 optimal Q值。
> - "value-based" 价值基础 - 表示Q学习是一个基于状态-动作价值函数(即Q函数)的方法。它试图学习一个表示每个状态下每个动作的长期价值的Q函数。
> - "uses a TD approach" 使用时序差分方法 - 这意味着Q学习使用TD学习的思想,通过bootstrapping从过去的经验中迭代学习,而不是直接从返回中学习。
> - "to train its action-value function" 训练其动作价值函数 - 指的是Q学习通过交互环境,采用TD误差来更新和改进其Q函数的估计。Q函数表示最终的训练目标。

Q-learning 是我们用来训练 Q 函数的一个算法。Q函数是一个action-value function，它用于确定处于特定状态并在该状态下采取特定行为的价值。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-function.jpg)

给定一个state 和 action 输入，Q 函数输出一个state-action value（也叫做Q value）

> The Q comes from “the Quality” (the value) of that action at that state.

回顾一下value 和reward 的不同之处：

- state/state-action 的 value 是agent 从当前state（state-action）开始，且后续遵循policy采取行动所获得的期望累积奖励。
- reward 是在一个state下执行某个action后从environment中获得的feedback。

在内部，我们的Q函数由Q-table进行编码，该表中的每个单元格对应一个state-action pair value 值，可以认为该表是Q函数的内存/备忘录。

举例：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-1.jpg)

Q-table 的每个单元格被初始化为0，该表包含每个state-action对应的value。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-2.jpg)

这里可以看到起始的state value 和 向上的 value都为0

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-3.jpg)

因此：在给定一个state-action时，Q函数将在它的Q-table中查找输出对应值。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-function-2.jpg)

回顾一下，Q-learning是这样的RL算法：

- 训练一个Q函数（一个action-value function），它的内部有一个Q-table包含所有的state-action pair values
- 给定一个state 和action，Q函数将在Q-table中查找相应的value
- 当训练结束的时候，我们会得到一个最优的Q function，也就意味着我们得到了一个最优的Q-table
- 如果我们有了一个最优的Q-table，我们就有了一个最优的policy因为我们可以知道在每个state下的最优的action

$`\pi^*(s) = \arg \max_{a} Q^*(s,a)`$
