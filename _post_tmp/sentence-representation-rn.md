---
layout: post
title: 'Sentence representation - Relational Network'
author: kion.kim
date: 2018-06-20 17:00
tags: [deeplearning, CNN, nlp, sentence representation, relation network, skip-gram]
---


## 시작하며

문장을 컴퓨터가 이해할 수 있는 언어로 표현하는 방법에 대해서 계속 이야기 하고 있습니다. 문장을 단어든 문자든 token이라는 최소 단위로 나누고 이들을 어떤 식으로 요약을 해서 하나의 문장으로 요약을 해보자는 것입니다. 그래야만 컴퓨터가 학습을 수행할 수 있을테니까요.

가장 처음에 알아봤던 것은 BoW에 의한 방법입니다. Token을 one-hot 벡터로 나타내고 이를 단순히 더하거나 평균을 낸 후 이 결과를 machine learning모형에 feeding을 하기만 하면 어느정도 quality의 분석 결과를 보여줍니다. 아주 naive하지만 강력한 방법으로 알려져 있습니다. 적어도 text classification 문제에서는 말입니다.

그뒤에는 Convolution netwowrk를 적용하고 있는 것들이 많아서 잘 추천을 할 수는 없을 것 같습니다.

아주 강력한 문젠는 아니지만, 자주 소주 2~3잔을 마셔보이고 합니다.
그다음 생각할 수 있는 것이 문장에 존재하는 모든 단어들에 대해 그 단어와의 관계를 하나의 벡터로 담는 것입니다. 이것을 문장 표현(sentence representation)을 위한 Relation Netowrk이라고 합니다.  다음과 같이  두 단어간의 관계를 표현 합니다.


$$f(x_i, x_j) = W\phi(U_{left}e_i +  U_{right}e_j )$$

여기서 $U$와 $W$는 학습해야 할 대상입니다. $\phi$는 non-linear transformation입니다. 모수들이 확정이 되고 나면, sentence representation은 다음과 같이 할 수 있습니다.

$$RN(S) = \frac 1 {2 N(N-1)}\sum_{i=1}^{T-1}\sum_{j = i +1}^T f(x_i, x_j)$$

그 다음으로 사용할 수 있는 방법은 CNN을 들 수 있습니다. $t$번 째 단어를 $k$-gram을 나타낼 수 있는 커널 $W_tau$를 적용시켜서 다음과 같이 $h_t$를 정의할 수 있습니다.

$$h_t = \phi\left(\sum_{i = - k/2}^{k/2} W_\tau e_{t+\tau}\right)$$

이렇게 정의된 $\h_t$를 모아서 $h = (h_1, \ldots, h_T)$로 표현하고 이 표현을 sentence representation으로 이용할 수 있습니다. 만약 여러개의 1D CNN을 사용한다면, token을 종합해서 구절을 표현하고, 구절을 압축해서 문장을 표현하는 식의 일반적인 문장에 대한 직관을 표현할 수 있다는 장점이 있습니다. 이미 구현되어 있는 Convolution layer들을 사용함으로써 손쉽게 구현할 수 있습니다. 이런 CNN을 sentence representation에 활용하는 기법들은 계속 발전하고 있습니다.

* Multi width convolutional layers [Kim, 2014; Lee et al., 2017]
* Dilated Convolutioal layers [
$$ h_t = f(x_t, x_1) + \cdots + f(x_t, x_{t-1})  + f(x_t, x_{t+1}) + \cdots + f(x_t, x_T)$$

위에서 $f$는 단어 간의 관계를 추출하는 함수라고 볼 수 있습니다. 위의 식을 설명하면, $t$ 번째 르어의 예측값은 해당 단어와 그 이외의 단어들의 관계를 모두 고려한 값이라고 할 수 있겠습니다. 반면에 CNN의 경우는 해당 단어의 주변 단어 정보만을 활용합니다.

$$ h_t = f(x_t, x_{t-k}) + \cdots + f(x_t, x_{t-1})  + f(x_t, x_{t+1}) + \cdots + f(x_t, x_{t+ k})$$

이 두가지 개념의 중간이 바로 attention mechanism이라고 할 수 있겠습니다. Recurrent network류의 모형들이 긴 문장을 번역하지 못하는 이유 중의 하나로 많이 드는 것이 모든 단어의 정보를 하나의 hidden vector로 압축을 하려하기 때문이라고 합니다. 단어들 간의 복잡한 관계를 하나의 vector로 표현한다는 것은 얼핏 생각해봐도 무리라는 생각이 듭니다.

그래서 attention mechanism은 다음과 같은 형태를 띕니다.


$$h_t = \sum_{t' = 1}^T \alpha(x_t, x_{t'}) f(x_t, x_{t'})$$

위에서 $alpha(\cdot,\cdot)$은 해당 위치에 있는 단어의 관계를 얼마나 반영할지를 나타내는 정도입니다. 만약에 지금 5번째 단어를 예측하는 데 있어 2번째 단어의 중요도는 얼마나 되는지를 측정한다고 볼 수 있겠습니다. $\alpha(\cdot, \cdot)$은 0과 1사이의 값을 가지며, 모두 합하면 1이 됩니다. 일종의 가중치라고 보시면 되겠습니다.



$$P(y| y_{-N}, y_{-N + 1}, \ldots, y_{-1})$$
