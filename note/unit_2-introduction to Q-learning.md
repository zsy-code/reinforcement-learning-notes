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

因此，无论你采取什么方法解决问题，你都会有一个策略。在value-base method下，你不用训练一个policy：你的policy只是一个简单的预先指定的函数（比如贪婪策略），它使用value function给出的值来选择对应的action。

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

在最开始的时候，Q-table是没有用的，因为它将为我们提供任意值（一般我们会把Q-table初始化为0），但随着agent对环境不断的探索以及我们对Q-table的更新，它将会提供越来越好的最优策略近似值。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-1.jpg)

### Q-learning 算法

下面是Q-learning的伪代码：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-2.jpg)

Step 1: 初始化Q-table

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-3.jpg)

我们需要给每个state-action pair初始化 Q-table，大多数时候，我们将初始值设为0.

Step 2: 使用epsilon-greedy 策略选择一个action

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-4.jpg)

epsilon-greedy 是一个可以平衡探索/利用的策略，它的思想是，在初始值 $\varepsilon = 1.0$：

- 对于probability $1-\varepsilon$：我们进行利用（也就是agent 选择state-action值最高的动作）

- 对于probability $\varepsilon$：我们进行探索（尝试随机动作）

在训练开始时，探索的可能性将会非常大因为 $ \varepsilon $ 很高，因此大多数时刻我们进行探索。但随着训练的进行，我们的Q-table的估计变得越来越好，就要逐渐的减小epsilon的值，因为我们将需要更少的探索和更多的利用。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-5.jpg)

Step 3: 执行 action $` A_{t} `$ ，获得reward $` R_{t+1} `$ 和 next state $` S_{t+1} `$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-6.jpg)

Step 4: 更新 $` Q(S_t, A_t) `$

在时间差分学习中，我们在每一次交互之后更新policy或者value function（基于我们选择的RL method）

为了达到我们的TD 目标，我们将及时奖励reward $` R_{t+1} `$ 和下一状态的折扣价值（通过找到下一状态能够最大化Q-function的动作来计算）进行加和，我们称之为bootstrap。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-7.jpg)

因此，$` Q(S_{t}, A_{t}) `$ 的更新公式就变成了：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-8.jpg)

这就意味着为了更新 $` Q(S_{t}, A_{t}) `$：

- 我们需要 $` S_{t}, A_{t}, S_{t+1}, R_{t+1} `$

- 为了更新一个给定的state-action pair的Q-value，我们使用 TD target

我们如何制定 TD target？

- 在执行action $` A_t `$ 后我们获取到reward $` R_{t+1} `$

- 为了得到下一时刻最好的state-action pair的value，我们使用一个贪婪策略来选择下一个最好的action，请注意，这不是epsilon-greedy 策略，它将总是选择最高的state-action value 对应的action

然后当Q-value 更新结束，我们从一个新的状态开始并再次使用epsilon-greedy 策略选择一个action

这就是为什么我们称 Q-learning 是一个离策略（off-policy）的算法

### 离策略（off-policy）vs 策略内（on-policy）

它们的差别很微妙：

- off-policy：使用不同的策略来执行（推理）和更新（训练）

比如，对于 Q-learning，epsilon-greedy 策略（acting policy），它和用来选择最好的下一状态的state-action value 来更新Q-value的greedy策略是不同的，即：

1. action policy：使用源自Q的policy（epsilon-greedy）选择action $` A_t `$ 

2. updating policy：$` \gamma \mathrm{max}_{a} Q(S_{t+1}, a) `$

- on-policy：acting 和 updating 使用相同的策略

比如，对于另一个value-based 算法 Sarsa，使用epsilon-greedy选择下一个action 而不是greedy policy。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/off-on-3.jpg)

汇总：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/off-on-4.jpg)


## 一个Q-learning示例

为了更好的理解Q-learning算法，让我们来举一个简单的例子：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Maze-Example-2.jpg)

- 在这个小迷宫中你是一只老鼠，你总是从相同的起点开始行动

- 目标是吃掉右下角的一堆奶酪并避免中毒

- 如果吃到毒药或吃到最大块的奶酪，或者行动超过5个step，则回合结束

- 学习率设为0.1，折扣率设为0.99

奖励函数如下：

- +0：来到一个没有奶酪的地方

- +1：来到一个有小奶酪的地方

- +10：来到有一大堆奶酪的地方

- -10：来到一个有毒药的地方然后死亡

- +0：如果行动超过5步

为了训练agent得到一个最优的policy（右右下），我们使用Q-learning算法

**Step 1: 初始化Q-table**

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Example-1.jpg)

现在，Q-table是无用的，我们需要使用Q-learning算法来训练Q-function

让我们执行两个训练step：

训练步骤1：

**Step 2: 使用epsilon-greedy 策略选择一个action**

因为epsilon 很大（=1.0），因此采取一个随机行动，在本例中，选择向右。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-3.jpg)

**Step 3: 执行Action $` A_t `$，得到 $` R_{t+1} `$ 和 $` S_{t+1} `$**

通过向右走，我们得到了一小块奶酪，因此 $` R_{t+1}=1 `$，并且处于一个新的状态。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-4.jpg)

**Step 4: 更新 $` Q(S_{t}, A_{t}) `$**

使用下面公式更新 $` Q(S_{t}, A_{t}) `$

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-5.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Example-4.jpg)

训练步骤2：

**Step 2: 使用epsilon-greedy 策略选择一个action**

我们再次选择一个随机的action，因为epsilon=0.99依然很大（我们需要逐渐一点点的减小epsilon，因为我们希望随着训练执行，进行的探索越来越少）

我们选择向下的action，这并不是一个好的action因为它导致的了死亡

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-6.jpg)

**Step 3: 执行Action $` A_t `$，得到 $` R_{t+1} `$ 和 $` S_{t+1} `$**

由于吃到了毒药，因此 $`R_{t+1}=-10`$ ，老鼠死亡

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-7.jpg)

**Step 4: 更新 $` Q(S_{t}, A_{t}) `$**

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-8.jpg)

因为角色死亡，所以要开始一个新的回合，通过两次训练step，agent变得更加智能了。

随着我们继续探索和利用环境，并使用TD target更新Q值，Q-table将会给我们一个越来越好的近似值，在训练结束时，我们将会得到一个最优的Q-function估计


## Q-learning回顾

Q-learning是这样的强化学习算法：

- 通过一个包含所有state-action value的Q-table训练Q function（一个在内部存储器中编码的action-value function）

- 给定一个state和action，Q-function将从Q-table中查找对应的值

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-function-2.jpg)

- 训练完成后，我们就有了一个最优的Q-function，或者等价的，得到了一个最优的Q-table

- 一旦我们有了一个最优的Q-function，我们就有了一个最优的policy，因为我们知道，对于每个state，都会采取最优的action

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/link-value-policy.jpg)

但是在训练开始时，Q-table是无用的，因为我们给Q-table提供了任意值进行初始化（大多数时候初始化为0），但随着对环境的探索以及对Q-table的不断更新，它将给我们提供越来越好的近似值。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/unit2/q-learning.jpeg)

Q-learning的伪代码如下：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-2.jpg)

## 术语

寻找最优策略的方法：

- policy-based 方法：该策略通常使用神经网络进行训练，以选择在给定状态下采取什么操作。在这种情况下，神经网络输出agent应该采取的action，而不是使用值函数。根据环境收到的经验，神经网络将重新调整并提供更好的action。

- value-based 方法：在这种情况下，训练价值函数来输出代表我们策略的state或state-action pair的值。但是，该值并没有定义agent应该采取什么操作。相反，我们需要在给定值函数的输出的情况下指定agent的行为。例如，我们可以决定采用一项策略来采取总是带来最大奖励的action（贪婪策略）。总之，该策略是贪婪策略（或用户采取的任何决策），它使用价值函数的值来决定要采取的操作。

在value-based method中，可以发现两种主要策略：

- state value function: 对于每个状态，state value函数是如果agent从该状态开始并遵循策略直到结束的预期回报。

- state-action value function: 与state value函数相反，如果agent在该状态下启动、采取该操作，然后永远遵循该策略，则操作值会为每个状态和操作对计算预期回报。


epsilon-greedy 策略：

- 强化学习中使用的常见策略涉及平衡探索和利用。

- 以 1-epsilon 的概率选择具有最高预期奖励的操作。

- 选择概率为 epsilon 的随机操作。

- Epsilon 通常会随着时间的推移而减少，以将重点转向利用。

greedy 策略：

- 涉及始终根据当前对环境的了解，选择有望带来最高回报的行动。

- 总是选择期望奖励最高的行动。

- 不包括任何探索。

- 在不确定或未知最佳行动的环境中可能是不利的。

off-policy 与 on-policy 算法：

- off-policy 算法：在训练时和推理时使用不同的策略

- on-policy 算法：在训练和推理过程中使用相同的策略

蒙特卡罗和时间差分学习策略：

- 蒙特卡罗（MC）：在回合结束时学习。使用蒙特卡罗，我们等到回合结束，然后根据完整的回合更新价值函数（或策略函数）。

- 时间差分（TD）：在每一步之后学习。通过时间差异学习，我们可以在每一步更新价值函数（或策略函数），而不需要完整的回合。

## 进阶阅读

### 蒙特卡罗和时间差分

- [Why do temporal difference (TD) methods have lower variance than Monte Carlo methods?](https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-met)
- [When are Monte Carlo methods preferred over temporal difference ones?](https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones)

### Q-learning

- [Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto Chapter 5, 6 and 7](http://incompleteideas.net/book/RLbook2020.pdf)
- [Foundations of Deep RL Series, L2 Deep Q-Learning by Pieter Abbeel](https://youtu.be/Psrhxy88zww)