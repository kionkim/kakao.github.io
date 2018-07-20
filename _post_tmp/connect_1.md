
## Text Classification

---

## How to represent a sentence

* A sentence is a variable-length sequence of tokens: $X = (x_1, \ldots, x_T)$
* Each token could be any one from a vocabulary: $x_t \in V$

abstraction 되어 있어서 이미지나 시그널과는 다르게 분석해야 함

* onece the vocabulary is fixed and encoding is done,


## How to represent a token

관계를 잘 모른다는 사실을 인코딩하고 싶음 -> index

* A token is an integer index
* How do should we represent a token so that it reflects its 'meaning'
* First, we assume nothing is known: use an on-hot encoding

* Every token is equally distant away from others

* Second, the neural network capture the token's meaning as a vector
* This is done by a simple matrix multiplication
$$ Wx = W[\hat x], \hat x = \arg \max_j x_j$$
> Table lookup layer

## How to represent a sentence - CBoW

* After the table-lookup operation, the input sentence is a sequence of continuous, high-dimensional vectors:

* The sentence length T differs from one sentence to another

단어를 veector로 표현하고, 평균을 내어 문장으로 표현함
단어의 순서 무시
하지만 너무 잘 작동함

> In practice, use FastText[Bojanovski et al., 2017] => 가장 첫번째로 시도해야 할 방법

문제에 가장 저합한 representation인가?
sentiment -> CBOW 등... 모형을 어떻게 만드느냐에 따라서 representation이 따라온다.


## How to represent a sentence - Relation Network[ Santoro et al., 2017]: Skip Bigrams
  * Consider all posible pairs of tokens:
  * Cobine two token vectors with a neural network for each pairs
$$f(x_i, x_j) = W\phi(U_{left}e_i  +  U_{right}e_j)$$

$$RN(X) = \frac 1{2N(N-1)} \sum_{i=1}^{T-1}\sum_{j = i +1} ^T f(x_i, x_j)$$


## How to represnet as sentence - CNN

* Convolutional Networks [Kim, 2014; Kalchbrenner et al., 2015]
  - Captures $k$-grams hierarchically
  - One 1-D convolutional layer: consider all $k$-Bigrams
  - Stack more than one convolutional alyers: progressively-growing window
  - Fits our intuituion of how sentence is understood



CNN은 아주 구현이 잘되어 있어서 efficient하게 computing할 수 있다는 장점이 있음


## How to represent a sentence - Self Attention

* Can we combine and generalize the relation network and CNN?

* CNN as a weighted relation network:

  - Original:
  - Weighted:

RN은 모든 weight가 1인거고, CNN은 윈도우 밖의 weight가 0인 거니깐.. 데이터에서 학습을 할 수 있지 않을까?

* That is, compute the weight of each pair $(x_t, x_{t'})$

$$ h_t =  \sum_{t' = 1 }^T \alpha(x_t, x_{t'}) f(x_t, x_{t'})$$

* The weigght function could be yet another neural network.

Long range dependency를 찾을 수 있고, redundant relationship을 제거할 수 있다.
여기에서 더욱 복잡하게 발전할 수있음. multi headed..blabla.... 다시 art의 영역

* Weaknesses of self-Attention
  - Quadratic computational complexity $O(T^2)$
  - Some operations acannot be done easily: e.g., counting


* Recurrent neural network: Online compression of a sequence $O(T)$

$$h_t = RNN(h_{t-1}, x_t), where h_0 =0$$

* Bidirectional RNN to account for both sides.
* Inherently sequential processing
  - Less desirable for modern, parallized, distributed computing infrastructure


* In all but CBOW, we end up with a set of vector representations.
* These approaches could be "stacked" in an arbitrary way to improve performance
  - Chen, Firat, Bpana et al. [2018]
  - Lee et al. [2017]

----

Polysemi problem ; How to disembligation??
embedding 차원 공간에서는 neighborhood가 아주 많아질 수 있음. 따라서 여러가지 의미를 encode를 할 수 있음.
이렇게 embedding된 공간에서 어떤 의미를 뽑을 것이냐? 그건 문제를 푸는 데에서 필요한 sentence representation을 찾아가도록 하게 됨

Language modeling

새로운 단어가 생길 때 어떻게 처리하나? 어떤 레벨에서 token을 define하였느냐?
캐릭터 레벨에서는 새로운 단어를 생성하기도 함
하지만 단어단위로 하면.... 새로운 단어를 추가해서 fine tuning. 기존에 존재하는 embedding의 조합으로 설명할 수 있음. language에서

챗봇... hierarchical LSTM. 클래스가 추가가 되면 새로 데이터를 모아서 트레이닝?
원하는 파라미터의 node만 update할 수 있음. 다른 파라미터들과 관계가 optimal한가?
class의 description을 이용할 수 있을까? weight vector를 뽑아내서 network에 이용

WASAB James Western이 잘 동작함.
Few shot learning, one shot learning


Attention의 길이: Attention의 길이를 어떻게 줄일 것인가가 큰 문제
Google=> 일정비율로 window를 잡음
computational efficiency vs 성능 => trade-off임

attention류에서는 parameter추정이 어려워서 초기값 설정 등의 문제가 있음
잘 모르겠음....ㅠㅠ

---

## Language Modelling

아주 다른 문제 같지만, 문장 생성과 문장 scoring은 같은 문제로 생각할 수 있음

* Autoregressive sequence modeling
  * The distribution over the next token is based on all the previous tokens

* Unsupervised learning becomes a set of supervised problems

자기상관 모델리은 unsupervised를 supervised로 바꿀 수 있음. Sentence scoring은 text classification을 아주 많이 하는 것으로 이해할 수 있음

## N-Gram Language models

* What would we do without a neural network?
* We need to estimate n-gram probabilities: $p(x| x_{-N}, x_{-N+1} ,\ldots x_{-1})$

* N-Gram Language models

* N-Gram problems
  - Data sparsity
  - Inability to capture long-term dependencies

## Traditional Solutions

1. Data sparsity
  - Smoothing: add a small constant to avoid 0
  - Backoff: try a shorter window

The most widely used approach: Kneser-Ney smoothing/backoff
**KenLM** implements the efficient n-gram LM model

2. Long-term dependency
  - Increase $n$: not feasible as the data sparsity worsens
  - \# off all possible $n$-grapms grows exponentially
  -
