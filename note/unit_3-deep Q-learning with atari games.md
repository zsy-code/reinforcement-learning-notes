# Deep Q-learning 算法

## 介绍

上一单元我们学习了第一个强化学习算法（Q-learning），从头实现并在两个环境（Frozen-Lake-v1、Taxi-v3）下训练了它。

我们使用这个简单的算法取得了出色的效果，但这些环境的状态空间是离散的而且相对简单（Frozen-Lake只有16个不同的状态，Taxi-v3有500个）。最为对比，Atari 游戏的状态空间有 $`10^9 ~ 10^11`$ 个state。

但正如我们所见，生成和更新Q-table在大型state space的environment上很可能变得无效。

因此在这个单元，我们将学习第一个深度强化学习agent：Deep Q-learning。它使用神经网络而不是Q table来对于给定的state估计每个action的Qvalue。

并且我们将会训练它玩space invaders以及其他的一些Atari环境的游戏，这个过程中我们使用RL-Zoo，一个使用Stable-baselines的RL 训练框架，它提供了训练、测试agent、超参数优化、可视化结果以及记录视频的脚本。

## 从Q-learning到Deep Q-learning

Q-learning是一个我们用来训练Q-function（一个动作价值函数，用于确定在给定状态以及当前状态下采用特定动作的价值）的算法。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-function.jpg)

"Q" 的说法来自于处于特定状态下特定动作的 "the Quality"。

在内部，我们的Q-function由Q-table编码，它的每个单元格都对应着一个state-action value，将Q-table视为Q-function内存或备忘单。

问题是Q-learning是一个表格方法，如果状态或动作空间不够小到有效的表示成数组或表格，它就会出现问题。换句话说，它不能够扩展。Q-learning 在一些很小的状态空间环境下表现良好，比如：

- Frozen-Lake，16个state
- Taxi-v3，500个state

但是想想我们今天要做的事情，我们将训练一个可以玩Space Invaders的agent，它是一个更复杂的游戏并且使用帧作为input。

正如[Nikita Melkozerov mentioned](https://twitter.com/meln1k)所说，Atari环境有一个形状为（210，160，3）的状态空间，包含（0～255）的值，因此我们可能有 $`256^{210 * 160 * 3}`$ 个状态（作为比较，可观测宇宙中大约有 $`10^80`$ 个原子）

- Atari的一个单独帧由一张210 * 160的图片构成，由于给定的图片是彩色的（RGB），所以有三个通道。这就是为什么形状是（210，160，3）。对于每个像素，取值为0～255。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/atari.jpg)

可以看到，状态空间是十分巨大的，因此在该环境下创建和更新Q-table可能是无效的，在这种情况下，更好的方法是使用参数化的Q函数 $`Q_{\theta }(s,a)`$ 。

在给定状态下，神经网络将会估计该状态下每个可能动作的不同 Q value，这就是Deep Q-learning的作用。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/deep.jpg)

## The Deep Q-Network(DQN)

下面是Deep Q-Learning network的架构：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/deep-q-network.jpg)

我们将选择4帧作为state输入给network，并输出一个该state下每个可能的action的Q-values向量。然后，像Q-learning一样，我们只需要使用epsilon-greedy policy来选择需要执行的action即可。

当神经网络初始化时，对于Q-value的估计将会很差，但随着训练的不断进行，Deep Q-Network agent会将具体情况和适当的动作联系起来并学会玩好游戏。

### 预处理输入和时间限制

我们需要对输入进行预处理，这是很重要的步骤，因为我们希望降低状态空间的复杂度，以减少训练的计算时长。

为了实现它，我们需要将状态空间减少到84*84，并且对它进行灰度化。因为Atari环境的颜色并没有提供重要的信息，这是一个很大的提升因为我们把三个通道减少到一个。

在一些游戏里我们也可以裁剪屏幕的一部分如果它不包含一些重要信息的话，然后再将4帧堆叠在一起。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/preprocessing.jpg)

为什么我们需要将4帧堆叠在一起呢？因为它可以帮助我们解决时间限制的问题。以Pong游戏举例，当你看到下面这一帧的时候：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/temporal-limitation.jpg)

你能知道这个球是往哪个方向移动的吗？很显然不能，因为一帧图像是不足以感知运动的。但如果我们添加另外三帧进来呢？就很明显的可以看出来球是往右侧移动的。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/temporal-limitation-2.jpg)

这就是为什么为了捕获时间信息，我们需要将4帧堆叠到一起。

然后堆叠起来的帧被三个卷积层处理，这些层能够让我们捕获和利用图像中的空间关系，而且，因为叠加到一起的帧，我们可以利用这些帧之间的一些时间信息。

如果你不知道什么是卷积层，可以看[Lesson 4 of this free Deep Learning Course by Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)

最后我们有一些全连接层，用来给特定状态下的每个可能的动作输出Q值。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/deep-q-network.jpg)

## Deep Q-Learning Algorithm

不同之处在于，在训练阶段，不像我们在Q-learning中那样直接更新state-action pair的Q值：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/q-ex-5.jpg)

在Deep Q-learning中，我们创建了一个损失函数来比较Q-value的预测值和Q-target，并使用梯度下降来更新Deep Q-Network的权重，来让估计Q-value变得更好。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/Q-target.jpg)

Deep Q-Learning Algorithm有两个阶段：

- 采样（Sampling）：我们采取actions并且将观察到的experience元组存储在重放内存中。
- 训练（Training）：随机选择一小个batch的tuples，并使用梯度下降的方法更新step从该batch中进行学习

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/sampling-training.jpg)

与Q-learning相比这并不是唯一的区别。Deep Q-learning的训练可能不稳定，主要是因为它由一个非线性的Q value函数（神经网络）和引导（我们用已经存在的估计而不是真实完整的回报来更新目标）组成。

为了帮助我们稳定训练，我们实施了三种不同的解决方案：

1. Experience Replay以更好的应用experience
2. Fixed Q-target以稳定训练
3. Double Deep Q-Learning，解决Q-value的高估问题

### Experience Replay以更有效的应用experiences

为什么我们要创建一个replay memory？

在Deep Q-learning中进行Experience Replay有两种方法：

1. 更有效的利用训练期间的experience。通常，在在线的强化学习中，agent与environment进行交互，并获取experience（state，action，reward，next state），并从它们中进行学习（更新神经网络），然后丢弃它们，这么做效率不高。

Experience Replay通过更有效的利用训练时的经验来帮助我们。我们使用replay buffer来保存在训练期间我们可以重用的经验样本。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/experience-replay.jpg)

这将允许agent多次从相同的经验中进行学习。

2. 避免忘记之前的经验（又叫灾难性干扰或灾难性遗忘）并且减少经验之间的相关性

- [灾难性遗忘（catastrophic forgetting）](https://en.wikipedia.org/wiki/Catastrophic_interference)：如果我们像神经网络提供连续的经验样本，我们会遇到的问题是，当它获得新的经验时，往往会忘记以前的经验，例如，如果agent处于first level，后来处于second level，它可能会忘记如何在first level中进行play。

解决方案是创建一个replay buffer，用来存储和环境交互产生的经验元组，然后从一小批元组中进行采样。这可以防止网络仅学习自己刚刚做的事情。

Experience Replay也有另一个好处，通过对经验的随机采样，我们移除了观察序列中的相关性，并且避免了action values发生灾难性的震荡或发散。

在Deep Q-Learning伪代码中，我们初始化了一个容量为N的replay memory buffer（N是一个你可以自己定义的超参数），然后我们在内存中存储experiences并在训练阶段采样一小批experiences喂给Deep Q-Network。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/experience-replay-pseudocode.jpg)

### Fixed Q-target以使训练稳定

当我们想要计算TD error（也称为loss）时，我们计算TD target（Q-target）和当前Q-value（Q的估计值）之间的差异。

但我们对真正的TD target一无所知，我们需要估计它。使用布尔曼方程，我们可以发现，TD target只是在该状态下采取动作的奖励以及下一状态下的最高的折扣Q值的加和。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/Q-target.jpg)

然而，问题是我们使用相同的参数（weights）来估计TD target和Q value。因此，TD target和我们正在修改的参数之间有显著的相关性。

因此，在训练的每一步，Q-values 和target values都会发生变化。我们会逐渐的离target更近，但是target也在移动。这就像在追逐一个移动的目标，因此可能会导致训练时的显著波动。

这就像如果你是一个牛仔（the Q estimation）并且你想要抓住一头牛（the Q-target）。你的目标是离牛更近（reduce the error）。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/qtarget-1.jpg)

在每一个时间步，你要尝试接近牛（牛在每一个步骤也会进行移动，因为你使用的是一样的参数）。

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/qtarget-2.jpg)

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/qtarget-3.jpg)

这导致了一条奇怪的追逐路径（训练时的显著震荡）

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/qtarget-4.jpg)

相反，我们在伪代码中看到的是：

- 使用具有固定参数的单独网络来估计 TD target
- 没 C 个steps 从Deep Q-Network 中复制参数来更新target network

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/fixed-q-target-pseudocode.jpg)

### Double DQN

Double DQN(Double Deep Q-Learning neural networks) 由[Hado van Hasselt](https://papers.nips.cc/paper/3964-double-q-learning)提出，这个方法旨在解决Q值高估的问题。

为了理解这个问题，需要回顾一下我们是如何计算TD target的：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/TD-1.jpg)

通过计算TD 目标，我们面临着一个很简单的问题，我们如何确定下一状态的最优动作是有最高Q-value的动作？

我们知道Q-value的准确性取决于我们尝试的动作以及我们探索的邻近状态。

因此，在训练开始时我们没有足够的信息来采取最优的动作。因此，将最大Q值（有噪声）作为最优动作又可能导致误报。如果定期给非最优动作高于最优动作的Q-value，学习可能会变得很复杂。

解决方法是：当我们计算Q-target时，我们用两个网络来将动作选择和target Q-value生成解耦：

- 使用DQN network来选择下一状态的最优动作（有最高Q value的动作）
- 使用Target network来计算下一状态下采取相应动作的target Q-value

因此，Double DQN帮助我们减少了Q-values的高估，并且帮助我们伴随着更稳定的学习进行更快的训练。

除了上面的三项改进之外，还衍生了许多其他的优化方法：比如Prioritized Experience Replay 和Dueling Deep Q-Learning等。

## 术语

- Tabular Method(表格方法)：state space 和action space 足够小到将value function近似表示为数组或表格的问题类型。Q-learning就是表格问题的一个例子因为它可以用表格来表示不同state-action pair的值。
- Deep Q-Learning：在给定状态下，训练神经网络来近似估计该状态下每个可能action的不同Q-value。它用来解决观察空间很大而不能用表格方法比如Q-learning方法来解决的问题。
- Temporal Limitation：当环境状态是用帧表示时，时间限制是一个难题。一帧图片本身不提供时间信息，为了捕获时间信息，我们需要将一组帧堆叠起来。
- Phases of Deep Q-Learning：
    - Sampling：采取动作后，观察到的经验元组被存储在一个replay memory中。
    - Training：若干批次的元组被随机选择，神经网络使用梯度下降更新参数。
- Solutions to stabilize Deep Q-Learning：
    - Experience Replay：一个replay memory被创建用来存储训练过程中可能重用的experiences。它使得agent可以多次从相同的experience中进行学习。除此之外，它也帮助agent在得到新的经验时忘掉之前的经验。
    - Random Sampling：从replay memory buffer中随机采样可以移除observation sequences中的相关性，并且避免action values的灾难性震荡或发散。
    - Fixed Q-Target：为了计算Q-Target我们需要通过布尔曼方程估计下一状态的折扣最优Q-value。问题是计算Q-Target和Q-value时使用的是同一套参数（weights）。这意味着任何时候我们修改Q-value时，Q-target也会随之发生变化。为了避免这个情况，使用具有固定参数的单独网络来估计TD target。target network在每C步之后通过从DQN网络中复制参数进行更新。
    - Double DQN：解决Q-value高估的问题。这个方案采用两个神经网络来将动作选择和target value生成进行解耦：
        - DQN Network：选择下一状态的最优动作（有最高Q-value值的动作）
        - Target Network：计算下一状态采取对应action的 target Q-value。该方法减少了Q-value的高估，它使得训练更快学习更稳定。