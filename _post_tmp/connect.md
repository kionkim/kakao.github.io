Supervised deeplearning

An algorithm is designed to solve the described problem
Machine learning: data-driven algorithm designed
  - A set of examples provided
  - A machine learning model is trained to solve the problem


Supervised learning

provided
  1. a set of N samples
  2. A per-example loss function
  3. Evaluation assets

What we must decide:
  1. Hypothesis sets
  - each set consists of all compatible models
  2. Optimization algorithm

---

1. For each hypothesis set H_m, find the best model

2. Model selection. Among the trained models, select the best one using validation sets

 - Hyperparameter optimization

3. Reporting. Report how whell the best model would work

실전에서 validation, train의 구분과 그런 것들이 중요하지는 않다. 하지만 test set은 꼭 빼놓아야 한다. 이건 cheating

---

Three points to consider
1. How do we deide/design a hypothesis set?
2. How do we decide a loss function?
3. How do we optimize the loss function?

----

Hypothsis sets-Neural Networks (parametric model)

1. The architecture of a network defines a set $\mathcal H$
2. Each model in the set $M\in \mathcal H$ is characterized by its parameters $\theta$

We use optimization to find 'a' good model from the hypothesis set.

---
Network Architectures

What is a neural network? An (arbitrary) directed acyclic graph(DAG)

1. Solid Circles: parameters (변수)
2. Dashed Cricles: vector inputs/outputs (주어진 값)
3. Squares: Compute nodes

방향이 정해져 있는 임의의 graph
어떤 모형이든 DAG로 변환할 수 있음

---

Inference - Forward Computation

Forward computation: how you 'use' a trained neural network.
  - Topological sweep(breadth-first)

---

DAG  <-> Hypothesis set

* Implication in practice
  - Naturally supports high-level abstraction
    - Object-oriented paradigm fits well
      - Base classes: variable node, operation nodes
      - Define the iternal various types of variables and operations by inherence
    - Maximal code reusability

---

A Neural network computes a conditional distribution

Supervised learning: what is y given x?

$$ f_\theta(x) = ?$$

How probable is a certain value $y$ of $y$ given $x$?

---

Bernoulli -> 잘은 모르겠지만, 어쨌든 DAG를 통과하면 확률같은 숫자를 내 보내 준다.
그 뒤에 있는 이론은 그닥 필요 없다.

---

Categorical

$$softmax(a) =\frac {\exp(a)}{\sum_{v=1}^C \exp(a_v)}$$

1992년 NIPS 논문에서 처음 소개된 방법.

---

Loss Function - negative log-probability

Training이란 데이터가 가장 likely한 것을 찾는 것임. 모델을 찾았는데 training set을 regenerate하면 되면 최고다.

negative log probability.. => Likelihood

* An OP node: negative log-probability
  - Inputs: the conditional distributio nand the correct output
  - Output: the negative log-probability

모든 문제를 확률을 출력하는 문제로 만들 수 있다.

----

## Local, iterative Optimization

* Ana rbitrary function
* Given the current value $\theta_0$
*
Random guided search
  - Stochastically perturb
  - Test each perturbed points
  - Find the best perturbed point
  - Repeat this until no improvement could be made

Applicable to any arbitrary loss function nad neural network
Inefficient in the high-dimensional parameter space: large $d$

전통적인 문제의 경우에는 parameter space가 compact하므로, guided random search가 잘됨. 하지만 NN의 경우에는 잘 되지 않음. 하지만, 장비빨이 수퍼갑

---

## Gradient descent

A continuous, differentiable function $L: \mathbb R^d \rightarrow \mathbb R$

Given the current value $\theta_0$, how should I move to minimize $L$?

아주 작은 neighborhood에서만 성립하기는 하지만, 그 점에서 loss를 줄여주는 방향은 확실하게 찾아준다.

---

Backward computation

* Manual derivation
* Automatic differentiation (autograd)
  - Use the chain rule of derivatives
  - The DAG is nothing but a composition of (mostly) differentiable functions
  - Automatically apply the chain rule of derivatives.

각 node마다 미분이 가능한 기능을 만들어 놓으면 backprop이 가능


---

Automatic differentiation

1. Implement the Jacobian-bector product of each OP node:
  - Can be implemented efficiently without explicitly computing the Jacobian
  - The same implementation can be reused every time of the OP node is called

---

## Backward Computation

사람들이 detail에 메이지 않고 창의적으로 architecture를 만들수 있고 엄청난 속도로 발전할 수 있게 됨

* Practical Implications - Automatic differentiation
  - Unless a complete new OP is introduced, no need to manually derive the Gradient
  - Nice de coupling of specification (front-end) and implementation (back-end)
    - [Front-end] Design a neural network by creating a DAG
    - [Back-end] The DAG is compiled into an efficient code for a traget compute divice


---

Gradient-based Optimization

이미 만들어진 최적화 도구들은 모수가 1만개 수준까지만 support

Backpropagation gives us the gradient of the loss function w.r.t. $\theta$
Readily used by off-the-shelf gradient-based optimizers

상용 도구들은 모든 graient의 합을 계산에 이용. 모두 계산을 하는 게 너무 어려운거니깐 stochastic gradient descent를 사용 (어차피 우리는 기대값을 찾아나가는 거고, 표본 평균이 이를 찾아낼 수 있다고 생각함)

de factor

---

Stochastic Gradient Descent

* Stochastic gradient descent in practice


학습하는 과정이 매 epoch마다 혹은 매 step마다 Hypothesis set이 계속 생긴다고 볼 수 있음. 그렇게 만들어지는 모형 중에서 이미 충분히 좋은 모형을 봤을 때 멈추는것. early stopping.. Generalization의 일부임

---

Stochastic Gradient Descent - Adaptive Learning Rate

어떻게 찾아가야할지 모른다. learning rate를 잘 찾고 싶다.
최근 논문은 Adam의 증명이 틀렸다... 라는게 논문

결국 현업에서는 detail을 알 필요 없다.

Abstraction이 너무 잘되어 있기 때문에 서로의 일에 대해서 별로 알 필요 없다. 그래서 발전 속도가 빠를 수 있다.

Abstraction..

---

Bias를 알면 importance sampling을 해서 stochastic gradient를 구할 수 있다.
