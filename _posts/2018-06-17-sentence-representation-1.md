---
layout: post
title: 'BoW를 이용한 text-classification'
author: kion.kim
date: 2018-06-01 17:00
tags: [deeplearning, statistics, generative-model]
---

## 시작하며

Attention mechanism은 자연어 처리 분야와 그 자연어 처리로부터 파생되는 수많은 영역에서 가장 활발하게 사용되고 있는 최근 2년간 발견된 architecture 중에서 가장 중요한 발견으로 여겨지고 있습니다. 어제 조경현 교수님 수업을 듣고 나니 attention mechanism의 사용은 이제 선택이 아니라 필수라는 생각이 들더군요. 언제나 느끼는 바이지만, 깔끔하게 정리된 수업을 듣다가 보니, 피상적으로 이해한다고 생각했던 것들은 십중팔구 이해했던 것이 아니었습니다. 그래서 또 저만의 flow로 다시 한번 구성해 보았습니다.

앞으로 몇개의 블로그를 더 쓸 예정인데, 여러가지 sentence representation으로 sentiment analysis를 하는 과정을 정리해 보려고 합니다. NLP 관련해서 첫 블로그인만큼, NLP에서 가장 기본이 되는 개념인 token representation과 sentence representation의 일반적인 이야기를 먼저 해보려 합니다. 그뒤에는 제일 간단한 형태의 sentence representation을 통해서 sentiment analysis를 진행하는 방법에 대해서 적어보겠습니다.

## Sentence represion의 일반론

먼저 문장은 컴퓨터가 이해할 수 있는 숫자로 바꾸어야 분석을 진행할 수 있을 것입니다. 문장을 숫자로 바꾸는 단계를 두가지로 나눠 볼 수 있는데요, 문장의 요소 (단어가 될 수도 있고, 문자가 될 수도 있습니다. 이 단위를 token이라고 이야기 합니다.)를 숫자로 표현하는 단계(token representation)와 숫자화 된 단어들을 요약해서 문장을 숫자로 표현하는 단계(sentence representation)으로 나눠볼 수 있겠습니다. Token을 숫자로 표현하는 가장 직관적인 방법은 token을 one-hot 벡터로 나타내는 것입니다. 물론, 단어를 잘 표현하기 위해서 word2vec 등의 embedding을 사용하기도 합니다. 하지만 오늘 글은 sentence representation이 그 목표이므로, 다루지 않겠습니다. 위에서 언급한 것과 같이 token이라는 개념은 다양하게 정의가 될 수 있습니다. 가장 일반적인 token의 단위는 단어(word)와 문자(character)입니다.  

단어를 one-hot으로 나타낸다는 것은 주어진 데이터 셋에 한번이라도 등장하는 단어를 모두 고려하여 벡터를 만들 수 있어야 하므로,  단어를 나타내기 위한 벡터의 차원이 아주 커진다는 단점이 있습니다. 특히 한국어의 경우, 단어의 변형이 너무나도 많습니다. 그래서 우리가 다루어야 할 embedding 공간이 아주 커지는 문제가 있습니다. 여기에다가 한국어에는 독특하게 단어들마다 띄워쓰기가 모두 다를 수가 있죠. '스타벅스'라고 쓰기도 하지만 '스타 벅스'라고 쓰기도 하면 이 것들을 하나의 단어로 인식해야 할 텐데 one-hot 벡터로는 이러한 차이를 반영할 수 없습니다. 다른 언어는 상대적으로 한국어에 비해 이러한 문제가 덜합니다. 또한 단순 오타 또한 차원을 높이는 데 큰 영향을 주기도 합니다. 잘못 쓴 단어는 새로운 단어로 인식을 하게 되니까요. 하지만, 이렇게 얻어진 token representation을 잘 이해한다고 하면, 인지적으로 의미있는 새로운 표현이 가능할 것입니다. '대학교', '중학교', '고등학교'는 embedding 공간에서 아주 비슷한 곳에 모여 있을 것이고, 사람이 그 유사성을 인지하기에 문제가 없다는 것입니다. 한국어 처리에서는 형태소 분석을 통해서 subword를 찾는 방법으로 단어의 변형을 하여주곤 합니다만, 여전히 단순 철자 오류는 다루기가 어렵습니다.

반면 문자 단위의 representation은 상대적으로 작은 차원으로 표현을 할 수가 있습니다. 한국어로 표현되는 문자는 여전히 많은 수준이기는 하지만, 영어의 경우에는 알파벳 개수와 몇몇 문장부도 정도가 전부입니다. 그래서 문자를 표현하기 위한 one-hot 벡터도 고작 몇십 차원에서 표현이 가능합니다. 한국어도 역시 데이터셋에 있는 서로 다른 단어의 개수에 비해서는 한국어 문자의 개수가 그렇게 많지 않습니다. 따라서 단어를  처리하는 것보다는 문자를 처리하는 것이 훨씬 효율적일 수 있습니다. 하지만, 이렇게 문자 단위의 representation을 해놓게 되면, representation 공간 안에서 단어의 유사성들을 잘 표현할 수 있는가? 하는 질문이 당연히 생기게 됩니다. 아주 잘 된다는 것이 지금까지 스터디 결과라고 하는군요. (자세한 내용은 저도 잘 모르겠습니다만, 모형의 성능이 그다지 떨어지지 않는다는 결과가 있었습니다.)
 
그 다음 필요한 것이 이렇게 숫자로 표현된 token representation을 바탕으로 문장을 표현하는 sentence represntation 단계입니다. 여기에서는 한 5가지 정도의 sentence representation이 존재합니다. 

* CBoW (BoW 포함)
* CNN
* Relation Network (Skip-gram)
* Self-attention
* RNN

물론, 이 외에도 수많은 variant들이 있겠죠. 제가 알고 있는 정도가 이정도입니다. 하지만 이 정도 표현을 벗어나지는 않는 것 같습니다. 가장 쉬은 것이 CBoW라고 해서 token representation을 순서에 상관없이 모두 평균을 내버리는 것입니다. token을 one-hot으로 표현하느냐 아니면, embedding으로 표현하느냐에 따라 BoW와 CBoW로 나뉩니다. 하나의 token에 대한 representation은 차원이 동일하므로, representation 벡터의 원소끼리 모두 더해서 평균을 낼 수 있습니다. 그렇게 얻은 평균을 해당 문장의 대표값으로 사용하는 것입니다. 당연히 단어의 순서는 전혀 고려되지 않습니다. 비록 순서를 무시하기는 하지만, CBoW 방법은 sentimental analysis와 같은 text classification에서는 아주 잘 작동을 합니다. 하지만, 단어의 순서가 중요한 machine translation이나 sentence generation에 적합한 방법은 아닐 것입니다. 이 블로그에서는 제일 단순한 형태인 BoW를 이용해서 sentiment analysis를 진행하는 과정을 나타내고자 합니다.

## BoW를 이용한 sentiment analysis

Sentiment analysis는 문장을 보고 그 문장에 나타나 있는 감정이 어떤 것인지를 예측하는 문제로 text classification의 가장 대표적인 문제입니다. 여기에서는 좀 식상하기는 하지만, umich-sentiment-train.txt로 제공되는 영화 평점 데이터에서 부정/긍정 대답을 구분하는 문제에 적용해 보겠습니다. 참고로 말씀드리자면, 워낙 쉬운 문제라 거의 1의 정확도를 보입니다. 앞으로 조금씩 더 어려운 데이터셋에 방법들을 적용해 가는 과정을 보여주는 것도 이 블로그의 목표이기도 합니다.

먼저 BoW에 대해서 이야기 해보겠습니다.
수학적으로는 다음과 같이 단순하게 쓸 수 있겠지요. 먼저 문장마다 제각각이지만, 만약 하나의 문장에 들어 있는 token의 숫자가 $T$라 하면, 각 token을 $(e_1, \ldots, e_T)$로 나타냅니다. 그런다음 문장을 다음과 같이 표현한다는 것입니다.

$$\frac 1 T \sum_{t=1}^T e_t$$

token representation을 가지고 있기만 하면 한방에 문장의 표현을 얻을 수 있습니다.



~~~
import os
import pandas as pd
import numpy as np
import nltk
import collections
from sklearn.preprocessing import normalize
~~~

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
