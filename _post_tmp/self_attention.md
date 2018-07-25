---
layout: post
title: 'Sentence representation - Self attention'
author: kion.kim
date: 2018-0x-xx 17:00
tags: [deeplearning, self-attention, nlp, sentence representation]
---

Self Attention은 


GLUON의 LSTM 모듈에 대해서 자세하게 설명하고자 합니다.

* Unroll only works if length = time step

* LSTM output = The first output of unroll output

* last cell state, last hidden state = Second ouput of unroll output




Sentence representation을 하는 방법에는 여러가지가 있습니다. 먼저 token(word나 character)을 수치화 한 후 이를 다시 한번 요약해서 sentence representation을 진행합니다.

Token representation에는 가장 간단하게는 단어를 one-hot으로 표현하는 방법이 있고, 단어를 단순히 one-hot vector가 아니라, 좀더 작은 차원의 embedding으로 표현하는 방법들이 있습니다. one-hot 표현이 discrete한 sparse vector로 표현이 된다면, embedding은 보다 차원이 작은 dense vector로 표현이 됩니다. Embedding을 하는 기법으로는 word2vec이 가장 대표적이라 하겠습니다. 앞으로 설명하게 될 대부분의 방법들은 embedding layer를 거의 필수적으로 사용합니다. 때로는 pre-train된 데이터를 사용할 수도 있고, 데이터로부터 직접 학습시킬 수도 있습니다.

이렇게 Token represntation이 된 후 이를 다시 요약하여 sentence representation을 하는 방법도 다양한데요. 단순히 순서 상관없이 합하거나 평균을 내어서 표현할 수도 있는데, 방법의 단순함에 비해 text classification에서는 아주 잘 작동을 합니다. 여기까지는 단순히 문장의 의미를 이해하고 있다기 보다는 출현하는 단어의 빈도를 통해 패턴을 추출한다고 볼 수 있습니다. 보다 문장의 흐름을 이해하려고 한다면, 문장 속에 나타나는 단어와의 관계를 고려해야 할 것입니다. 이렇게 단어의 순서를 고려하기 위해 Language model에서는 RNN을 이용합니다. RNN은 hidden state에 단어들의 순서에 관한 정보를 저장합니다. $T$개의 단어로 이루어진 문장의 경우, $T$개의 time step을 가지고 있다고 생각할 수 있고, $d$ 차원으로 축약된 hidden state를 가지고 있다면, 하나의 문장은 다음과 같이 요약될 수 있습니다.

$$ H =  (h_1, \ldots, h_T), \quad h_i \in \mathbb R^d $$

하나의 문장을 $d\times T$ 행렬로 요약을 했습니다. 문장 분류에서는 마지막 hidden state값을 많이 사용하는데, 문장 분류에서는 충분할지 모르겠으나, 문장을 번역하기 위해서는 너무 많은 문장에 대한 정보를 하나의 벡터로 줄이려고 하니, 표현력에 문제가 생기기도 합니다.

$n$-gram의 아이디어를 차용해서, 문장의 일부분을 표현할 때, 문장 전체를 고려하지 않고 단순히 주변의 단어만을 고려한다면, 다음과 같이 1D CNN을 적용할 수 있습니다.

![1d_conv_1](/assets/1d_conv_1.png)

위의 그림은 2번의 1d convolution을 거치면서 데이터가 요약되는 과정을 표시했습니다. 만약 kernel의 크기가 3이고, padding을 1을 준 뒤에 채널을 $c_1$개로 준다면, 다음과 중간의 레이어와 같이 표현되고, 채널이 $c_2$인 커널의 크기가 3인 1d convolution을 적용하면 최종적으로 $c_2 \times T$의 행렬을 얻게 되고, 이것이 하나의 sentence representation이 됩니다.
만약 여러개의 1D CNN을 사용한다면, token을 종합해서 구절을 표현하고, 구절을 압축해서 문장을 표현하는 식의 일반적인 문장에 대한 직관을 표현할 수 있다는 장점이 있습니다. 그리고 RNN에 비해서 계산이 빠르다는 장점도 가지고 있습니다.

이 외에도, 두 token들의 관계의 내적을 통해 문장을 표현할 수도 있을 것입니다. 아래는 두 단어들의 조합들의 함수로 문장을 표현하는 것을 표현한 그림입니다.

![rel_net](/assets/rel_net.png)

각 token들에 함수 $f(\cdot, \cdot)$을 적용하고 이들의 합을 문장의 일부로 표현한 것입니다. 2개 이상의 token들 간의 관계를 보는 것도 가능할 것입니다.

위와 같은 세가지 경우 모두 다음과 같은 일반적인 함수의 형태로 나타낼 수 있습니다.

$$ h_t = f(x_t, x_{1}) + \cdots + f(x_t, x_{t-1})  + f(x_t, x_{t+1}) + \cdots + f(x_t, x_{T})$$

RNN으로 표현한 경우는, $x_t$ 이후의 함수값이 모두 0이어서 다음과 같이 쓸 수 있겠죠.

$$ h_t = f(x_t, x_{t-k}) + \cdots + f(x_t, x_{t-1})$$


반면, CNN의 경우는 다음과 일정 범위 내의 token만 관여하므로 다음과 같이 쓸 수 있습니다. 여기서 $k$는 kernel을 통해 고려되는 token의 개수입니다. 1개의 convolution만 사용했다면, $k$는 곧 커널의 크기가 될 것입니다.


$$ h_t = f(x_t, x_{t-k}) + \cdots + f(x_t, x_{t-1})  + f(x_t, x_{t+1}) + \cdots + f(x_t, x_{t+ k})$$

RNN을 이용한 representation은 CNN과 Relation network은 token의 고려되는 token의 개수 관점에서 양 극단이라고 할 수 있겠습니다. relation network는 전체 token을 사용해서 sentence representation을 하는 반면, CNN은 일정 부분의 token만을 사용해서 sentence representation을 진행합니다. RN의 경우 모든 가능한 두 token들의 조합을 사용하므로, 많은 정보를 포함할 수 있지만, 쓸데 없는 정보가 들어가 있을 수 있고, CNN의 경우는 가까운 token들의 정보만 이용하므로, 멀리 떨어져 있는 token의 관계를 반영하지 못합니다. 이 두가지를 절충한 것이 Self-Attention 개념입니다.


### Self-Attention

Self - attention mechanism은 다음과 같은 형태를 띕니다.


$$h_t = \sum_{t' = 1}^T \alpha(x_t, x_{t'}) f(x_t, x_{t'})$$

위에서 $\alpha(\cdot,\cdot)$은 해당 위치에 있는 단어의 관계를 얼마나 반영할지를 나타내는 정도입니다. 만약에 지금 5번째 단어를 예측하는 데 있어 2번째 단어의 중요도는 얼마나 되는지를 측정한다고 볼 수 있겠습니다. 결국 하고 싶은 것은 token의 조합이 text-classification에 영향을 미쳤나를 파악하는 것입니다.

Self-attention은 다음의 그림에서와 같이 구조화시킬 수 있습니다.

![sa_2](/assets/sa_2.png)

문장에 속에 있는 각 token의 정보를 압축해 놓은 어떤 형태의 embedding, $H_1, \ldots, H_T$,이 있다고 하면, 각 token의 정보는 $\alpha$로 표현되는 weight들에 의해 자기자신 혹은 문장 안에 있는 다른 token과의 관계가 결과에 미치는 중요도가 조정됩니다. Attention weight들은 다음의 조건을 만족합니다.

$\sum_{t = 1} ^T \alpha_{i, t} = 1$


결국 feed forward 단계를 통해 얻은 embedding에 weight matrix를 곱한 후에 dense network를 통과시킨 후 2개의 final node의 값을 얻습니다. Ground truth와의 비교를 통해 loss 값을 발생시킨 후, backward step을 통해 $\alpha$의 값을 학습을 시키게 됩니다. 결국 얻고 싶은 것은 token과 token의 관계의 정도를 표시하는 다음과 같은 attention matrix입니다.

![sa_1](/assets/sa_1.png)



원래 논문([A structured Self-Attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130.pdf))에서 소개되어 있는 self-attention은 bidirectional LSTM의 hidden layer에 SA mechanism을 적용했습니다.

![Structured Self-Attentive Sentence Embedding](/assets/structured_sa_sentence_embedding_fig1.png)

하지만, 꼭 lstm을 통해 sentence representation을 진행할 필요는 없습니다. 이 글에서는 relation network을 통해 token 간의 관계를 요약하여 sentence representation을 진행한 후에 이 위에 SA mechanism을 올려보겠습니다. 여기서 우리는 문장을 $D \times T$의 행렬로 표시했다고 가정하겠습니(참고로 원래 논문에서는 반대방향으로 진행하는 $u$ 차원의 LSTM hidden state 2개를 concat하여 문장은 $2u \times n$ 크기의 행렬로 표현했습니다. 여기서 $n$은 문장의 길이입니다.)

먼저 self attention은 입력되는 sentence에 있는 token의 함수가 token 자신의 attention값이 되는 모형입니다. Attention 행렬은 다음과 같은 수식으로 표현할 수 있습니다.


$$ A = softmax(W_2 tanh (W_1 H^T))$$

여기에서 $H$는 입력되는 문장의 sentence representation입니다. 위의 notation을 따르면 $D \times T$의 크기를 지닙니다.
$W_1$은 그 크기가 $d \times D$로서, $D$ 차원의 embedding을 $d$ 차원으로 줄여놓는 변환을 의미합니다. 이렇게 변환 된 embedding 값에 tanh 변환을 적용하여, -1에서 1 사이의 값을 갖는 $d\times T$ 행렬을 얻습니다. 여기에 다시 한번 선형 변환을 통해 $r\times T$의 행렬로 얻은후에 softmax 변환을 통해 확률값으로 각 dimension (총 r개의 dimension에 대해... ) $T$개의 원소에 대해 softmax를 적용하여 최종 attentive matrix를 얻습니다. $r$개의 attention vector를 고려하는 이유는 하나의 attention만으로는 담지 못하는 단어의 중요도가 있을지도 모르기 때문입니다. 그래서 r개의 attention 벡터를 구하고 나서 해석은 이들의 평균이나 아니면 가장 그럴 듯한 해석만을 사용합니다.

### Self-Attention with Gluon

여기에서는 relation network를 이용해서 sentence representation을 한후 여기에 attention을 적용시켜 보도록 하겠습니다.
* Multi width convolutional layers [Kim, 2014; Lee et al., 2017]
* Dilated Convolutioal layers [
