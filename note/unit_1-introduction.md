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


在这个框架中,智能体需要决定下一步的操作,以最大化其累积奖励。它不同于监督学习,因为训练数据中不提供正确的序列操作,也不同于非监督学习,因为存在奖励信号来指导学习。通过大量的环境交互,强化学习中的智能体能够逐步提升其策略,以取得更高的累积奖励。因此强化学习非常适合解决顺序决策、自动控制等问题。它已经在游戏、机器人控制、自动驾驶等领域得到成功应用。