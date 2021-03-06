# Key Concepts and Terminology

![../_images/rl_diagram_transparent_bg.png](https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png)

RL(强化学习)的主要特征是智能体和环境，环境是智能体交互的世界，在每一步的交互中，智能体看到（可能是部分的看到）世界状态的观测，然后决定采取的行动。环境随着智能体的行动而改变，也可能随着自己改变。

智能体能感知到来自环境的**reward**(奖励)信号,这是一个告诉它当前世界状态好坏的数字，智能体的目标是最大化当前的累积奖励，叫做**return**，强化学习是让智能体学习行为去获得它目标的学习方法。

## 1.States and Observations

**State**(状态)s是对世界状态的完整描述，对状态来说世界没有隐藏的信息。**Observation**(观测)是对状态的部分描述，可能忽略了一些信息。

在深度强化学习中，我们几乎总是用一个实值向量，矩阵，高阶张量来表示状态和观测。举个例子，一个视觉观测可以用其像素值的RGB矩阵来表示，一个机器人的状态可以用它关节角度和它的速度来表示。

如果一个智能体能观测环境的完整状态，我们称环境为**fully observed**(完全观测的)。如果一个智能体只能看见部分观测，我们称环境为**partially observed**(部分观测的)。

## 2. Action Spaces

不同的环境允许不同种类的动作，一个给定环境中的合法动作集被称为**action space**(动作空间)，有些环境，像Atari和Go，有**discrete action spaces**(离散动作空间)，其中智能体只能执行有限数目的移动，其他环境，像智能体控制一个机器人在物理世界中，有**continuous action spaces**(连续动作空间)，在连续动作空间中，动作是实值向量。

这种区别对深度强化学习中的方法有相当深刻的影响，一些算法只能直接应用于离散空间，只能通过大幅修改应用于其他环境。

## 3.Policies

一个**policy**(策略)是智能体采取动作的规则，它可以是确定性的，这种情况下常用$\mu$表示
$$
a_t=\mu(s_t)
$$

或者它可能是离散的，常用$\pi$表示
$$
a_t=\pi(\cdot|s_t)
$$
因为策略是智能体的大脑，因此使用policy代替agent并不罕见，例如：The policy is trying to maximize reward.

在深度强化学习中，我们处理**parameterized policies**(参数化策略)：策略的输出依赖于可计算函数的一组参数(就好像神经网络中的weights和bias)，我们可以通过一些优化算法调整行为的改变。

我们通常用$\theta$和$\phi$来表示策略的参数，然后将其作为策略符号的下表来强调联系：
$$
\begin{aligned}
a_t&=\mu_\theta(s_t)\\
a_t&=\pi_\theta(\cdot|s_t)
\end{aligned}
$$

### 3.1 Deterministic Policies

Example:以下为TensorFlow中连续动作中的确定性策略

```python
obs = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
net = mlp(obs, hidden_dims=(64,64), activation=tf.tanh)
actions = tf.layers.dense(net, units=act_dim, activation=None)
```

其中mlp表示通过给定的隐藏层和激活函数连接起来的多层神经网络。

### 3.2 Stochastic Policies

深度强化学习中最常用的两种随机策略是 **categorical policies**(分类策略)和**diagonal Gaussian policies**(对角高斯策略)

分类策略被用于离散动作空间，连续策略被用于连续动作空间。

对于使用和训练随机策略，有两个计算是十分重要的：

- 从策略中采样动作

- 计算相应动作的log 似然，$log\pi_\theta(a_t|s_t)$

#### 3.2.1 **Categorical Policies**

一个分类策略像一个离散动作的分类器，为分类策略构建神经网络的方法和为分类器构建神经网络的方法一样：输入是观测，紧接着是一些层（可能是卷积层或者是全连接层，这依赖于输入的种类），然后你有一个最终的线性层给出每个动作的对数，之后跟随一个softmax将对数转移为概率。

**Sampling**(采样)：给定了每个动作的概率，像TensorFlow这样的框架有内建的工具用于采样，举个例子，像 [tf.distributions.Categorical](https://www.tensorflow.org/api_docs/python/tf/distributions/Categorical)或[tf.multinomial](https://www.tensorflow.org/api_docs/python/tf/multinomial)

**Log-Likelihood**(对数似然)：定义最后一层概率为$P_\theta(s)$，它是一个向量，有多少项就有多少个动作，所以我们可以把这些动作视为向量的索引。通过一个向量的索引来获得一个动作$a$的对数似然：
$$
log\pi_\theta(s,a)=log[P_\theta(s)]_a
$$

#### 3.2.2 Diagonal Gaussian Policies

一个多元高斯分布（或者说多元正态分布）由一个均值向量$\mu$和一个协方差矩阵$\Sigma$描述。对角高斯分布是一个特例，它的协方差矩阵只有对角项有元素，所以我们能用一个向量表示它。

一个对角高斯策略总是用一个神经网络将观测（状态）映射为动作均值$\mu_\theta(s)$，而协方差矩阵有两种方式来表示。

**第一种**：用一个单独的向量表示对数标准差$log\sigma$，它并不是状态$s$的函数，只是单独的参数（spinning up中的VPG，TRPO和PPO都使用了这种方式实现）。

**第二种**：用一个神经网络将状态映射为对数标准差$log\sigma_\theta(s)$，可以和平均值网络共享一些层。

注意在这两种方式下我们都输出对数标准差而不是标准差，因为对数标准差可以是$(-\infty,+\infty)$中的任何值，而标准差必须是非负的，如果不需要这些约束训练参数会更容易，标准差可以从对数标准差里求指数而来，所以我们这样做不会损失任何表示。

**Sampling**(采样)：给定均值动作$\mu_\theta(s)$和标准差$\sigma_\theta(s)$，一个来自球面高斯函数的噪声向量$z$（$z\sim N(0, I)$），一个动作采样可以如下计算：
$$
a=\mu_\theta(s)+\sigma_\theta(s)\odot z
$$
$\odot$定义为两个向量的点积，标准框架中内置了方法去计算噪声向量，像[tf.random_normal](<https://www.tensorflow.org/api_docs/python/tf/random_normal>)，或者你可以提供均值和标准差直接使用 [tf.distributions.Normal](https://www.tensorflow.org/api_docs/python/tf/distributions/Normal) 来采样

**Log-Likelihood**(对数似然)：一个$k$维的动作，一个均值$u=\mu_\theta(s)$和标准差$\sigma=\sigma_\theta(s)$的对角高斯策略形成的对数似然为：
$$
\log \pi_{\theta}(a | s)=-\frac{1}{2}\left(\sum_{i=1}^{k}\left(\frac{\left(a_{i}-\mu_{i}\right)^{2}}{\sigma_{i}^{2}}+2 \log \sigma_{i}\right)+k \log 2 \pi\right)
$$

> 推导：对某一个状态$s$，采样某一维动作$a_i$的概率为：
> $$
> \pi_\theta(s,a_i)=\frac{1}{\sqrt{2\pi}\sigma_i}e^{-\frac{(\mu_\theta(s)-a_i)^2}{2\sigma^2}}
> $$
> 其中$\mu_\theta(s)$为神经网络映射的某一维动作均值$\mu_i$
>
> $k$维动作的联合概率（对角协方差表示了独立）为：
> $$
> \pi_\theta(s,a)=\prod_{i=1}^{k}\pi_\theta(s,a_i)=\prod_{i=1}^{k}\frac{1}{\sqrt{2\pi}\sigma_i}e^{-\frac{(\mu_i-a_i)^2}{2\sigma^2}}
> $$
> 对其求对数可得

## 4.Trajectories

一个轨迹$\tau$是一条状态动作序列，
$$
\tau = (s_0,a_0,s_1,a_1,\cdots).
$$
初始状态$s_0$是由开始状态分布采样而来，有时定义为$\rho_0$:
$$
s_0\sim\rho_0(\cdot)
$$
状态转移（时刻$t$的状态$s_t$和时刻$t+1$的状态$s_{t+1}$之间），由环境的规则而定，依赖于最近的动作$a_t$。它可能是确定性的：
$$
s_{t+1}=f(s_t,a_t)
$$
或者是离散的：
$$
s_{t+1}\sim P(\cdot|s_t,a_t)
$$
动作来自于智能体的策略

> 轨迹也被称作episodes或rollouts.

## 5. Reward and Return

(reward)奖励函数$R$在强化学习中是重要的，它依赖于目前的状态，采取的动作，下一个状态：
$$
r_t=R(s_t,a_t,s_{t+1})
$$
常常被简化为只依赖于当前状态，$r_t=R(s_t)$，或状态动作对$r_t=R(s_t,a_t)$

智能体的目标是最大化一个轨迹的累积奖励，我们将其定义为$R(\tau)$

一种return(回报)是**finite-horizon undiscounted return**(有限长度的未折扣回报)，它是固定窗口步数的奖励的累加：
$$
R(\tau)=\sum_{t=0}^Tr_t
$$
另一种return是 **infinite-horizon discounted return**(无限长度的折扣回报)，它是智能体获得奖励的折扣累加和，折扣根据未来的距离，$\gamma\epsilon(0, 1)$:
$$
R(\tau)=\sum_{t=0}^\infty \gamma^tr_t
$$
折扣因子在直观上和数学上是便利的，直觉上来说：现在的现金比以后的现金好，数学上：一个无穷长度的奖励累加和可能不会收敛到一个有限值，且在方程中很难处理。但是有折扣因子且处于合理条件下，无穷和会收敛。

> 在RL的形式主义中，这两种return公式的界限是相当明显的，而深度RL尝试去模糊这个界限——举个例子，我们常常建立算法去优化未折扣return，但是使用折扣因子去评估**value functions**(值函数)

## 6.The RL Problem

不管选择什么样的return度量（不管是无限长度带有折扣因子，或是有限长度不带折扣因子），不管选择什么样的策略，RL的目标是选择一个策略，使智能体根据这个策略最大化**期望return**。

为了谈论期望return，我们首先不得不讨论轨迹的概率分布。

假定环境的转移和策略都是随机的，在这种情况下，T-step 轨迹的概率是：

$$
P(\tau|\pi)=\rho_0(s_0)\prod_{t=0}^{T-1}P(s_{t+1}|s_t,a_t)\pi(a_t|s_t)
$$
期望return，定义为$J(\pi)$:
$$
J(\pi)=\int_\tau P(\tau|\pi)R(\tau)=E_{\tau\sim\pi}[R(\tau)]
$$
关键的优化问题可以表示为：
$$
\pi^\ast=argmaxJ_\pi(\pi)
$$
$\pi^\ast$表示最优化策略

## 7. Value Functions

知道状态/状态动作对的价值常常是有用的，所谓价值，我们指的是你在某个状态或某个状态动作对开始的期望return，然后根据特定的策略做出动作，在几乎所有的RL算法中，都以这样或那样的方式使用(**Value functions**)值函数。

这里有四种主要的函数

1. **On-Policy Value Function**(在线策略值函数)，$V^\pi(s)$，当你在状态$s$开始而且总是以策略$\pi$做出动作而获得的期望return:
   $$
   V^\pi(s)=E_{\tau\sim\pi}[R(\tau)|s_0=s]
   $$
   
2. **On-Policy Action-Value Function**(在线策略动作值函数)，$Q^\pi(s,a)$，当你在状态$s$开始，做出任意的动作$a$（可能并不来自于策略$\pi$），之后根据策略$\pi$做出动作而获得的期望return:
   $$
   Q^\pi(s,a)=E_{\tau\sim\pi}[R(\tau)|s_0=s,a_0=a]
   $$
   
3. **Optimal Value Function**(最优值函数)，$V^\ast(s)$，当你在状态s开始之后根据optimal policy(最优策略)所获得的期望return:
   $$
   V^\ast(s)=max_\pi(E_{\tau\sim\pi}[R(\tau)|s_0=s])
   $$

4. **Optimal Action-Value Function**(最优动作值函数)，$Q^\ast(s,a)$，当你在状态$s$开始，做出任意的动作$a$，之后根据最优策略做出动作而获得的期望return：
   $$
   Q^\ast(s,a)=max_\pi(E_{\tau\sim\pi}[R(\tau)|s_0=s,a_0=a])
   $$
   
> 当我们讨论值函数时，如果我们不考虑时间依赖，我们意思是无限长度的折扣return。值函数作为有限长度的非折扣return需要以时间作为参数。

> 在值函数和动作值函数之间有两个关键的联系：
> $$
> V^\pi(s)=E_{a\sim\pi}[Q^\pi(s,a)]
> $$
>
> $$
> V^\ast(s)=max_aQ^\ast(s,a)
> $$

## 8. The Optimal Q-Function and the Optimal Action

在最优动作值函数$Q^\ast(s,a)$和最优策略下的动作选择之间有着重要的联系，通过定义，$Q^\ast(s,a)$意思为开始在状态$s$，采取任意的动作$a$，之后永远按照最优策略所获得的期望return。

在$s$下的最优策略将选择一个能最大化以状态s开始的期望return的动作，如果我们有$Q^\ast$，我们可以直接获得最优动作$a^\ast(s)$通过：
$$
a^\ast(s)=argmax_aQ^\ast(s,a)
$$
注意：可能有多个动作能最大化$Q^\ast(s,a)$，在这种情况下，他们都是最优的，最优策略可以选择它们中的任何一个，但总是存在一个最优策略确定性的选择一个动作。

## 9. Bellman Equations

这四种值函数都遵守一种特殊的自治方程叫做**Bellman equations**(贝尔曼方程)，贝尔曼方程的基本思想如下：

> 起始点的价值是你从那儿获得的期望奖励，加上你到达的下一个点的价值。

在线策略值函数的贝尔曼方程是：
$$
\begin{aligned} 
V^{\pi}(s) &=\underset{a\sim\pi, s^{\prime} \sim P}{\mathrm{E}}\left[r(s, a, s^{\prime})+\gamma V^{\pi}\left(s^{\prime}\right)\right] \\
Q^{\pi}(s, a) &=\underset{s^{\prime} \sim P}{\mathrm{E}}\left[r(s, a, s^{\prime})+\gamma \underset{a^{\prime} \sim \pi}{\mathrm{E}}\left[Q^{\pi}\left(s^{\prime}, a^{\prime}\right)\right]\right] 
\end{aligned}
$$
$s^{\prime}\sim P$是$s^{\prime}\sim P(\cdot|s,a)$的缩写，表示下一个状态$s^{\prime}$是由环境的转移规则采样而来，$a\sim\pi$是$a\sim \pi(\cdot|s)$的缩写，$a^{\prime}\sim \pi(\cdot|s^{\prime})$的缩写

> 推导：将$V^\pi(s)$展开
> $$
> \begin{aligned} 
> V^\pi(s)&=E_{\tau\sim\pi}[R(\tau)|s_0=s]\\
> &=E_\pi[\sum_{t=0}^{\infty}\gamma^tr_t|s_0=s]\\
> &=E_\pi[r_0+\sum_{t=1}^{\infty}\gamma^tr_t|s_0=s]\\
> &=E_\pi[r_0|s_0=s]+E_\pi[\sum_{t=1}^{\infty}\gamma^tr_t|s_0=s]\\
> &=\sum_{a}\pi(a|s)\sum_{s^{\prime}}p(s^{\prime}|s,a)r(s,a,s^{\prime})+\sum_{a}\pi(a|s)\sum_{s^{\prime}}p(s^{\prime}|s,a)\gamma E_{\pi}[\sum_{t=0}^{\infty}\gamma^tr_{t+1}|s_1=s^{\prime}]\\
> &=\underset{a\sim\pi, s^{\prime} \sim P}{E}\left[r(s, a, s^{\prime})+\gamma V^{\pi}\left(s^{\prime}\right)\right] 
> \end{aligned}
> $$
> 同理：将$Q^\pi(s,a)$展开
> $$
> \begin{aligned}
> Q^\pi(s,a)&=E_\pi[\sum_{t=0}^{\infty}\gamma^tr_t|s_0=s,a_0=a]\\
> &=E_\pi[r_0+\sum_{t=1}^{\infty}\gamma^tr_t|s_0=s,a_0=a]\\
> &=E_\pi[r_0|s_0=s,a_0=a]+E_\pi[\sum_{t=1}^{\infty}\gamma^tr_t|s_0=s,a_0=a]\\
> &=\sum_{s^{\prime}}p(s^{\prime}|s,a)r(s,a,s^{\prime})+\sum_{s^{\prime}}p(s^{\prime}|s,a)\sum_{a^\prime}\gamma E_\pi[\sum_{t=0}^{\infty}\gamma^tr_{t+1}|s_1=s^\prime,a_1=a^\prime]\\
> &=E_{s^{\prime}\sim p}[r(s,a,s^{\prime})+E_{a^\prime\sim\pi}\gamma Q(s^\prime,a^\prime)]
> \end{aligned}
> $$
> 

最优值函数的贝尔曼方程如下：
$$
\begin{aligned} 
V^{\ast}(s)&=\underset{a}{max}\underset{s^{\prime}\sim P}{\mathrm{E}}[r(s,a,s^{\prime})+\gamma V^{\ast}(s^{\prime})]\\
Q^{\ast}(s,a)&=\underset{s^{\prime} \sim P}{\mathrm{E}}\left[r(s,a,s^{\prime})+\underset{a^{\prime}}{max}[\gamma Q^{\ast}(s^{\prime},a^{\prime})]\right]
\end{aligned}
$$
在线策略值函数的贝尔曼方程和最优值函数的贝尔曼方程的关键不同在于是否有$max$操作符，这反映了智能体在选择动作时，为了最优化动作，它不得不挑选能让值最大化的动作。

> 推导：
> $$
> \begin{aligned} 
> V^{\ast}(s)&=max_a[Q^{\ast}(s,a)]\\
> &=max_a\left[E_{\pi^{\ast}}[r_0+\sum_{t=1}^{\infty}\gamma^tr_t|s_0=s,a_0=a]\right]\\
> &=max_a\left[E_{s^\prime\sim p}\left[r(s,a,s^{\prime})+\gamma E_{\pi^{\ast}}\left[\sum_{t=0}^{\infty}\gamma^tr_{t+1}|s_1=s^{\prime}\right]\right]\right]\\
> &=max_a\left[E_{s^\prime\sim p}[r(s,a,s^{\prime})+\gamma V^{\ast}(s^\prime)t]\right]\\
> \end{aligned}
> $$



> $$
> \begin{aligned} 
> Q^{\ast}(s,a)&=E_{s^\prime\sim p}\left[r(s,a,s^{\prime})+\gamma V^{\ast}(s^\prime)\right]\\
> &=E_{s^\prime\sim p}\left[r(s,a,s^{\prime})+\gamma max_{a^{\prime}}[Q^{\ast}(s^{\prime},a^{\prime})]\right]
> \end{aligned}
> $$


## 10. Advantage Functions

有时候在RL中我们不需要描述一个动作在绝对意义上有多好，只需要描述它比平均行为好多少。也就是说，我们想知道这种行为的相对优势。我们用 **advantage function**(优势函数)这个概念来描述它。

与策略$\pi$相关的优势函数$A^\pi(s,a)$描述了在状态$s$下采取动作$a$有多好，相对于在策略$\pi(\cdot|s)$下随机的采取一个动作。数学上，优势函数定义为：
$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

> 优势函数在策略梯度法中很重要。
