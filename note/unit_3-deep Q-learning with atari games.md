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

