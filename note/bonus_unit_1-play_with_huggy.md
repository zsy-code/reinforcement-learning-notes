# 使用huggy 对深度强化学习进行进一步介绍

## huggy 的工作原理

huggy 是一个基于unity的环境，在这个环境中，我们的目标是训练huggy来抓住我们抛出去的物品，这意味着它需要朝着物品的方向正确的移动位置。

### huggy获取到的状态空间（state space）

Huggy 不能看到它所处的环境，因此我们提供给它具体的环境信息：

- target/stick的位置
- huggy 和 target 的相对位置
- huggy 的 legs 的方向

通过这些提供的信息，huggy 可以使用它的策略来决定下一步应该做什么以达到它的目的。

### 动作空间，指示huggy可以如何移动的（action space）

关节马达驱动huggy的腿，因此为了达到目标，huggy 需要学会如何旋转它每条腿的线性马达以帮助它能正确的进行移动。

### 奖励函数（the reward function）

为了使huggy达到它的目标（捡到扔出去的物品），因此才设计了奖励函数。

要记得深度强化学习的核心之一就是奖励机制：一个任务的目标可以被描述为最大化期望累积奖励。

在这里，我们的目标是huggy走向stick 而不需要它旋转太多次，因此我们的奖励函数必须对这个目标进行转化。

我们的奖励函数：

![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/unit-bonus1/reward.jpg)

- 方向奖励（Orientation Bonus）：当它更接近目标是我们对它给予的奖励

- 时间惩罚（Time penalty）：每次行动后我们都会给它一个固定的时间惩罚，从而强制它以最短的时间到达目标

- 旋转惩罚（Rotation penalty）：如果它旋转太多次我们对它给予惩罚

- 达到目标奖励（Getting to the target reward）：当huggy 达到目标时给予的奖励

### 训练huggy

huggy的训练目标是尽可能快的并且正确的跑向target，在每个step以及给定的环境observation的条件下，要实现这样做，它需要决定如何旋转它每条腿上的线性马达来实现正确移动（而不是多次旋转）并且朝向target

