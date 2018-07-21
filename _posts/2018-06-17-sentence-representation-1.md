---
layout: post
title: 'BoW를 이용한 text-classification'
author: kion.kim
date: 2018-06-17 17:00
tags: [deeplearning, statistics, nlp models, text classification, BoW, machine learning]
---

## 시작하며

Text classification은 주어진 NLP에서 가장 쉽고 기본적인 task에 해당합니다. 문장에서 단어들의 pattern을 찾아 그 문장이 어느 범주에 속하는 것이 가장 기본적인 접근 방법이라고 한다면, 문장의 정보를 이해해서 문장을 분류해 낼 수 있다면 더욱 좋은 결과를 낼 수 있을 것입니다. 그 과정이 자연어 처리가 발전하는 과정일텐데요. 이번 블로그 시리즈에서는 아주 빠른 속도로 발전해 가는 자연어 처리 기법들을 적어보려고 합니다. 저도 전문가는 아니므로 초보자 눈높이에 맞춰 (곧 저의 눈높이), 글을 적어보려구요. 다양한 형태와 기법의 text classification부터 시작해서 결국에는 Neural Machine Tranlation(NMT)의 기법까지 다루어 보는 것이 목표입니다.

이 블로그는 6월 말에 있었던 조경현 교수님의 connect 재단 초청 강연을 보고 이제는 스스로 정리를 해야될 때가 되었다 싶어 쓰기 시작합니다. 언제나 느끼는 바이지만, 깔끔하게 정리된 수업을 듣다가 보니, 피상적으로 이해한다고 생각했던 것들은 십중팔구 이해했던 것이 아니었습니다. 그래서 또 저만의 flow로 다시 한번 구성해 보았습니다.

이 글에서는 가장 기본적인 Back of Word(BoW)를 이용한 text classification에 대해서 이야기 해 보겠습니다.

## 서설... Attention mechanism을 향해

이 글에서 소개해 드릴 BoW를 이용한 text classification은 문장을 분류함에 있어서 문장을 이해하여 해당 카테고리를 찾아간다고 볼 수 는 없습니다. 단순히 문장에 속해 있는 token의 패턴을 통해서 분류를 하는 형태입니다. 아주 초보적인 수준의 분류 기법입니다. 하지만, deep learning 기법들이 발달하면서 단어의 단순한 출현 빈도나 패턴 보다는 좀더 문장의 정보를 이해하기 시작했습니다. NLP에서 가장 먼저 각광받은 신경망이 Recurrent Neural Network(RNN)이었습니다. Convolutional Neural Network(CNN)은 이미지 처리에서 큰 활약을 하고 있었는데요. 그림의 지역적인 특성을 필터링하는 데 사용했던 Convolution 기법을 문장의 지역적인 특성과 문맥을 파악하는데 활용하기 시작했습니다. 최근에는 CNN을 활용한 처리 기법도 많이 나오는 상태구요. 하지만 정보를 어떤 형태로 축약하는지는 그렇게 알기 쉽지 않았습니다. 그리고 아주 긴 문장에 대한 처리는 여전히 문제였습니다. 이런 상황에서 최근에 아주 주목받고 있는 것이 attention mechanism입니다.

Attention mechanism은 자연어 처리 분야와 그 자연어 처리로부터 파생되는 수많은 영역에서 가장 활발하게 사용되고 있는 architecture로서 빠른 속도로 많은 모형에 적용되고 있습니다. 이제 모형의 성능 관점은 물론 해석 관점에서도 아주 중요한 요소로 여겨지고 있습니다. 특히 어제 조경현 교수님 수업을 듣고 나니 attention mechanism의 사용은 이제 선택이 아니라 필수라는 생각이 들더군요.

NLP 관련해서 첫 블로그인만큼, NLP에서 가장 기본이 되는 개념인 token representation과 sentence representation의 일반적인 이야기를 먼저 정리한 후에, 오늘 이야기할 BoW를 이용한 text classification을 진행해 보도록 하겠습니다.

> NOTE: 아래의 분석 과정은 [Text_Classification](http://210.121.159.217:9090/kionkim/stat-analysis/blob/master/nlp_models/notebooks/text_classification_CBOW_representation.ipynb)에 자세하게 나타나 있습니다. 참고하시기 바랍니다.


## Sentence representation의 일반론

먼저 문장은 컴퓨터가 이해할 수 있는 숫자로 바꾸어야 분석을 진행할 수 있을 것입니다. 문장을 숫자로 바꾸는 단계를 두가지로 나눠 볼 수 있는데요, 문장의 요소 (단어가 될 수도 있고, 문자가 될 수도 있습니다. 이 단위를 token이라고 이야기 합니다.)를 숫자로 표현하는 단계(token representation)와 숫자화 된 단어들을 요약해서 문장을 숫자로 표현하는 단계(sentence representation)으로 나눠볼 수 있겠습니다.

### Token representation

Token을 숫자로 표현하는 가장 직관적인 방법은 단어에 순서를 부여하는 것이겠지요. '너의 목소리가 들려'라는 문장이 나타나면, '너의'는 1번, '목소리가'는 2번, '들려'는 3번, 이런 식으로 번호를 붙여나가는 방법일 것입니다. 전형적인 숫자로 표현되는 범주형 자료형으로서, 이러한 데이터를 분석할 때, 숫자화된 token을 one-hot 벡터로 나타냅니다. 통계학에서는 '지시변수' 혹은 indicator라고 부릅니다. 단어를 잘 표현하기 위해서 word2vec 등의 embedding을 사용하기도 합니다. 이 방법은 token representation을 소개하는 블로그에서 따로 다룰 예정이구요. (열심히 정리 중입니다. 해야할 게 너무 많네요..ㅠㅠ)

위에서 언급한 것과 같이 token이라는 개념은 다양하게 정의가 될 수 있습니다. 가장 일반적인 token의 단위는 단어(word)와 문자(character)입니다.  

단어를 one-hot으로 나타내는 방법은 주어진 데이터 셋에 한번이라도 등장하는 단어를 모두 카운트 해서, 각 단어를 하나의 차원으로 보고 벡터를 만드는 것입니다. 이렇게 하다 보니 수많은 단어가 포함되어 있는 데이터셋을 단어의 one-hot 벡터로 만들려면 벡터의 차원이 아주 커진다는 단점이 있습니다. 보통 수천에서 수만 차원에 이릅니다. 특히 한국어의 경우, 단어의 변형이 너무나도 많습니다. 그래서 우리가 다루어야 할 embedding 공간이 아주 커지는 문제가 있습니다. 이 문제는 stem 단어 혹은 어근을 찾음으로써 어느정도 해결이 됩니다. 해결이 된다기 보다는 그런 식으로 처리를 합니다.

여기에다가 한국어에는 독특하게 단어들마다 띄워쓰기가 모두 다를 수가 있죠. '스타벅스'라고 쓰기도 하지만 '스타 벅스'라고 쓰기도 하면 이 것들을 하나의 단어로 인식해야 할 텐데 one-hot 벡터로는 이러한 차이를 반영할 수 없습니다. 다른 언어는 상대적으로 한국어에 비해 이러한 문제가 덜하다고 하는군요. 또한 단순 오타 또한 차원을 높이는 데 큰 영향을 주기도 합니다. 잘못 쓴 단어는 새로운 단어로 인식을 하게 되니까요.

하지만, 이렇게 얻어진 token representation은, 인지적으로 이해하기는 쉬울 것입니다. '대학교', '중학교', '고등학교'는 embedding 공간에서 아주 비슷한 곳에 모여 있을 것이고, 사람이 그 유사성을 인지하기에 문제가 없다는 것입니다.

반면 문자 단위의 representation은 상대적으로 작은 차원으로 표현을 할 수가 있습니다. 한국어로 표현되는 문자는 여전히 많은 수준이기는 하죠. 자음과 모음의 조합으로 아주 다양한 발음을 표현할 수 있는 특성 때문이죠. 반면, 영어의 경우에는 알파벳 개수와 몇몇 문장부호 정도가 전부입니다. 그래서 문자를 표현하기 위한 one-hot 벡터도 고작 몇십 차원에서 표현이 가능합니다. 따라서 단어를  처리하는 것보다는 문자를 처리하는 것이 훨씬 효율적일 수 있습니다. 하지만, 이렇게 문자 단위의 representation을 해놓게 되면, representation 공간 안에서 단어의 유사성들을 잘 표현할 수 있는가? 하는 질문이 당연히 생기게 됩니다. '대학교'의 '대'가 '대구'의 '대'와 같은 '대'일까요? 다른 '대'일까요? 가까운 '대'일까요? 먼 '대'일까요? 하지만, 최고 난이도의 번역에서조차 아주 잘 된다는 것이 지금까지 스터디 결과라고 하는군요. (자세한 내용은 저도 잘 모르겠습니다만, 모형의 성능이 그다지 떨어지지 않는다는 결과가 있다고 합니다. 저도 정확한 결과가 궁금하군요.) 성능이 잘 나오는 이유로는 고차원의 데이터 공간에서는 어떤 데이터 포인트의 주변에 있는 neighborhood가 충분히 많다는 것입니다.


### Sentence representation

그 다음 필요한 것이 이렇게 숫자로 표현된 token representation을 바탕으로 문장을 표현하는 sentence representation 단계입니다. 약 5가지 정도의 sentence representation이 있는데요. 다음과 같습니다.

* CBoW (BoW 포함)
* CNN
* Relation Network (Skip-gram)
* Self-attention
* RNN

물론, 이 외에도 수많은 variant들이 있겠죠. 제가 알고 있는 정도가 이정도입니다. 하지만 이 정도 표현을 벗어나지는 않는 것 같습니다. 오늘은 BoW만 다루어 보겠습니다. token을 one-hot으로 표현하느냐 아니면, embedding으로 표현하느냐에 따라 BoW와 CBoW로 나뉩니다.

## BoW를 이용한 sentiment analysis

Sentiment analysis는 문장을 보고 그 문장에 나타나 있는 감정이 어떤 것인지를 예측하는 문제로 text classification의 가장 대표적인 문제입니다. 여기에서는 좀 식상하기는 하지만, umich-sentiment-train.txt로 제공되는 영화 평점 데이터에서 부정/긍정 대답을 구분하는 문제에 적용해 보겠습니다. 참고로 말씀드리자면, 워낙 쉬운 문제라 accuracy는 거의 1의 정확도를 보입니다. 앞으로 조금씩 더 어려운 데이터셋에 방법들을 적용해 가는 과정을 보여주는 것도 이 블로그의 목표이기도 합니다.

각각의 token에 대한 representation은 차원이 동일하므로, representation 벡터의 원소끼리 모두 더하거나 평균을 낼 수 있습니다. 이렇게 얻은 하나의 벡터를 해당 문장의 대표값으로 사용하는 것입니다. 당연히 단어의 순서는 전혀 고려되지 않습니다. 비록 순서를 무시하기는 하지만, CBoW 방법은 sentimental analysis와 같은 text classification에서는 아주 잘 작동을 합니다. 하지만, 단어의 순서가 중요한 machine translation이나 sentence generation에 적합한 방법은 아닐 것입니다.

수학적으로는 다음과 같이 단순하게 쓸 수 있겠지요. 먼저 문장마다 제각각이지만, 만약 하나의 문장에 들어 있는 token의 숫자가 $T$라 하면, 각 token을 $(e_1, \ldots, e_T)$로 나타냅니다. 그런다음 문장을 다음과 같이 표현한다는 것입니다.

$$\frac 1 T \sum_{t=1}^T e_t$$

혹은

$$\sum_{t=1}^T e_t$$

너무 쉽죠? 이게 전부입니다. 이제 숫자로 표현된 문장을 분류하기만 하면 됩니다.


## Python을 이용한 구현


Python에서는 어떤 형태로 구현이 될까요? 먼저 다음의 module을 불러옵니다.

~~~
import os
import pandas as pd
import numpy as np
import nltk
import collections
~~~

네번째 줄에 있는 nltk 모듈은 가장 많이 사용되는 nlp를 위한 전처리 모듈로서 이 블로그에서는 문장을 단어 단위로 tokenize하기 위해서 사용됩니다.
다섯번째 줄에 있는 collections는 container에 들어있는 원소들의 count를 보다 쉽게 하기 위해 사용하는 module입니다. 기본적으로 python에서 제공해 주는 collection 기능보다 더욱 많은 기능을 제공합니다. 이제 본격적으로 데이터를 불러오도록 하겠습니다.

~~~
word_freq = collections.Counter()
max_len = 0
num_rec = 0

with open('../data/umich-sentiment-train.txt', 'rb') as f:
    for line in f:
        label, sentence = line.decode('utf8').strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > max_len:
            max_len = len(words)
        for word in words:
            word_freq[word] += 1
        num_rec += 1
~~~

Collections에 있는 Counter class는  Container에 동일한 값의 자료가 몇개 있는지를 확인하는 객체입니다. 데이터가 저장되어 있는 file을 열고 개별 라인을 읽어오면서 label과 sentence를 분리합니다. 이렇게 얻어진 각각의 데이터에서 문장의 최대 길이와 전체 데이터셋에 등장하는 단어의 갯수를 셉니다.

이 분석에서는 최대 2000개의 단어만 고려할 것입니다. Word_freq에 저장되어 있는 정보를 활용해서 데이터에 가장 많이 등장하는 2000개 단어를 사용할 것입니다. 그런 다음, 한 문장을 구성하는 단어의 개수는 40개로 제한합니다. 만약 어떤 문장이 40개 이상의 단어로 구성이 되어 있으면, 최초 40개만 분석에 활용하고 그보다 짧은 문장이면 0을 채워 넣어서 40개 단어를 맞춥니다. 만약 2000개 안에 속하지 않는 단어를 만나면 1을 대입해서, 모르는 단어(Unknown)임을 표시합니다.

~~~
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
# most_common output -> list
word2idx = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(MAX_FEATURES - 2))}
word2idx ['PAD'] = 0
word2idx['UNK'] = 1
~~~

{단어: 인덱스} 자료를 바탕으로 {인덱스: 단어} 자료도 만들어둡니다. 숫자로 표현된 자료를 원래 자연어로 돌리는데 사용하는 등 나중에 여러모로 활용할 수 있습니다.

~~~
idx2word= {i:v for v, i in word2idx.items()}
vocab_size = len(word2idx)
~~~

다음 코드에서는 위에서 정의된 유효한 단어, 한 문장을 이루는 단어의 개수, 유효하지 않은 단어의 처리 방법 등을 바탕으로 실제 데이터를 불러옵니다. 숫자화된 데이터는 `x`와 `y`에 저장을 하고 원래 문장 데이터는 `origin_txt`라는 변수에 저장이 됩니다.    

~~~
y = []
x = []
origin_txt = []
with open('../data/umich-sentiment-train.txt', 'rb') as f:
    for line in f:
        _label, _sentence = line.decode('utf8').strip().split('\t')
        origin_txt.append(_sentence)
        y.append(int(_label))
        words = nltk.word_tokenize(_sentence.lower())
        _seq = []
        for word in words:
            if word in word2idx.keys():
                _seq.append(word2idx[word])
            else:
                _seq.append(word2idx['UNK'])
        if len(_seq) < MAX_SENTENCE_LENGTH:
            _seq.extend([0] * ((MAX_SENTENCE_LENGTH) - len(_seq)))
        else:
            _seq = _seq[:MAX_SENTENCE_LENGTH]
        x.append(_seq)
~~~

실제 데이터에 부정과 긍정의 평가가 어떤 빈도로 나타나는지 보면, 부정이 3091개 긍정이 3995개입니다.

~~~
pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index()
~~~

이렇게 얻어진 단어의 index 벡터를 one-hot 벡터로 바꾸기 위해서 다음과 같은 함수를 정의합니다.

~~~
def one_hot(x, vocab_size):
    res = np.zeros(shape = (vocab_size))
    res[x] = 1
    return res
~~~

위에 정의된 함수를 바탕으로 다음을 실행하면, 이제 행의 갯수가 2000이고 열의 개수가 40인 one-hot 벡터로 구성된  행렬이 문장의 개수만큼 만들어집니다.

~~~
x_1 = np.array([np.sum(np.array([one_hot(word, MAX_FEATURES) for word in example]), axis = 0) for example in x])
~~~

머신러닝에서 가장 중요한 개념 중의 하나는 bia-variance trade-off입니다. 이를 잘 극복하기 위해서 사용하는 방법이 데이터를 training set과 validation set으로 나누고, 학습은 training set에서 시키고, 학습된 결과가 일반화될 수 있는지를 가늠하기 위해서 validation set에서 학습 결과를 평가합니다. 기존의 통계적인 방법론에서도 일반화를 고려하기 위해서요사용되던 개념이었지만, 머신러닝에서는 훨씬 더 그 중요도가 높아졌는데요. 통계적인 모형은 모형이 단순하여, 모형의 표현력이 떨어지는만큼, training set만 잘 맞추게 되는 over-fitting 문제가 크지 않은 반면, 머신러닝 모형은 over-fitting 문제가 훨씬 심합니다. 그래서 training set과 validation set을 나누는 것, 그리고 더 나아가서는 test set까지, 이렇게 3개의 데이터셋으로 나누는 것은 아주 중요한 과정입니다.

다음은 지금까지 처리된 데이터를 training set과 validation set으로 나누는 작업입니다.

~~~
tr_idx = np.random.choice(range(x_1.shape[0]), int(x_1.shape[0] * .8))
va_idx = [x for x in range(x_1.shape[0]) if x not in tr_idx]

tr_x = x_1[tr_idx, :]
tr_y = [y[i] for i in tr_idx]
va_x = x_1[va_idx, :]
va_y = [y[i] for i in va_idx]
~~~

여기까지 진행하면 classification을 위한 데이터는 모두 준비가 되었습니다. 이제 classifier로 분류 문제를 풀어볼 차례입니다.

### Classification

몇가지 classifier를 적용해 보겠습니다. 먼저 xgboost를 적용하기 위해서는 다음의 library들이 필요하네요.

~~~
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
~~~

마지막 줄은 성능을 평가하기 위한 준비작업으로 accuracy score를 구하기 위해 필요한 모듈입니다. xgboost 모형을 학습하기 위한 코드는 아주아주 간단합니다. 다음과 같이 클래서 선언 후, fit method를 호출하면 끝입니다. 물론 다양한 모수가 있기는 하지만, 여기에서는 default로 갑니다. 아주 잘 나옵니다.
~~~
xgb = XGBClassifier()
xgb.fit(tr_x, tr_y)
~~~

다음은 추정된 모형에 validation set을 대입해서 실제 예측치를 얻고 그 예측치에 대한 accuracy를 구하는 과정입니다.

~~~
y_pred_xgb = xgb.predict(va_x)
pred_xgb = [round(val) for val in y_pred_xgb]

accuracy_xgb = accuracy_score(va_y, pred_xgb)
print('Accuracy: %.2f%%'%(accuracy_xgb * 100.0))
~~~
위의 결과를 돌려보면,  validation set의 accuracy가 0.98에 육박함을 알 수 있습니다. 그러니깐.. 대책없이 쉬운 문제라고 볼 밖에요.

이제 비슷한 scheme으로 DNN을 적용해 보겠습니다. 여러가지 많이 사용되는 framework 중에 후발주자로서 가장 늦게 시작하기는 하였지만, 늦게 출발한 만큼 기존에 활발하게 사용되고 있는 deep learning framework인 tensoflow, pytorch, keras의 장점만을 모아서 만들어 놓은 mxnet이라는 framework이 있습니다. 그리고 mxnet을 보다 쉽게 사용하기 위해 mxnet을 wrapping한 gluon이라는 framework는 쉽기도 하고 유연하기도 해야 하는 deep learning framework의 요건을 모두 만족시키는 framework로 생각됩니다. Keras는 쉽기는 하나 점점 복잡해지는 deep learning network을 표현하기에는 좀 부족해 보이는 반면, gluon은 keras만큼 쉬우면서도 유연한 tool이라는 게 개인적인 생각입니다. mxnet과 gluon을 사용하기 위해서는 다음과 같이 필수 module을 import합니다.

~~~
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
context = mx.gpu()
~~~

tensorflow에서는 GPU와 CPU를 오가는 것이 그렇게 user-friendly 정의가 되어있지 않았습니다. 어떤 장비를 사용하는지에 따라 graph가 꼬이기도 하고, 뭔가 말로 표현하기는 좀 어렵지만, 굉장히 사람을 신경쓰이게 하는 면이 있었습니다. 하지만, gluon에서는 context를 지정하는 것으로 어떤 resource를 이용하는가에 대한 고민은 크게 하지 않아도 됩니다. GPU를 사용하고 싶으면 mx.gpu(0)를 CPU를 사용하고 싶녀면 mx.cpu(0)을 지정하면 됩니다.

Gluon의 기본적인 programming style은 pytorch를 따릅니다. 다음과 같이 `nn.Block` class를 상속 받아서 구현하고자 하는 network를 정의하는 형태인데요. name_scope 안에 network에서 사용할 weight값이 들어 있는 layer들을 정의합니다. 그리고 실제 feed-forward 계산은 `forward` method에서 이루어집니다. Class 내에서 network를 쌓아가는 모습은 pytorch의 형태와 비슷하기는 하지만, 이를 Keras의 형태로 network를 정의할 수도 있습니다. 아마도 앞으로 nn.Sequence를 상속받은 class를 통해 network를 정의하는 예제도 따라 나올 것 같습니다.


~~~
class MLP(nn.Block):
    def __init__(self, input_dim, emb_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Embedding(input_dim = input_dim, output_dim = emb_dim)
            self.dense1 = nn.Dense(64)
            #self.dense2 = nn.Dense(32, activation = 'relu')
            self.bn = nn.BatchNorm()
            self.dense2 = nn.Dense(2)

    def forward(self, x):
        x = self.embed(x)
        x = self.dense1(x)
        x = self.bn(x)
        x = nd.relu(x)
        x = self.dense2(x)
        return x
~~~

위에서는 1개의 dense layer를 이지고 있는 간단한 network를 정의했습니다. Batch-normalization을 적용했고, RELU activation 함수를 사용했죠. 그 결과 나오는 마지막 output은 2개의 node를 가지게 됩니다. softmax를 사용해서 이 두개의 node값을 긍정과 부정의 확률값으로 표현하게 될 것입니다. 다음과 같이 실제로 class를 객체화시킨 후에 mlp라는 객체가 가지고 있는 모수들, 다시 말하면 weight값,들을 초기화하는 단계를 거칩니다. Xavier initializer를 사용합니다. 물론, GPU를 사용하기 위해 context지지정해 줍니다.

~~~
mlp = MLP(input_dim = MAX_FEATURES, emb_dim = 50)
mlp.collect_params().initialize(mx.init.Xavier(), ctx = context)
~~~


학습을 시키기 위해서는 loss 함수와 optimizer를 정해야 하는데요. gluon에서는 간단하게 다음과 같이 정의할 수 있습니다.

~~~
loss = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(mlp.collect_params(), 'adam', {'learning_rate': 1e-3})
~~~

SoftmaxCELoss는 Softmax를 적용한 후 Cross Entropy Loss를 적용하라는 의미입니다. 여기에서 softmax가 적용될 것이기 때문에 위에서 network를 정의할 때에는 최종 output layer의 마지막에는 따로 activation 함수를 지정하지 않았습니다. trainer에는 학습을 해야할 모수와 optimizer의 종류, 그리고 optimize에 필요한 hyperparameter를 넣어주어야 합니다. 이제 학습을 하기 위해 모형 관련된 내용들은 모두 정의가 된 상태입니다. 요약하면, DNN을 수행하기 위해서는 다음의 4가지 정도를 꼭 정해주어야 한다는 거죠.

* Network architecture
* Optimizer
* Loss Function
* Hyper parameter

Data를 network에 feeding할 때, deep learning에서는 mini batch를 사용하게 됩니다. 큰 데이터를 메모리에 담을 수 없어 나오게 된 현실적인 고려인데, 이렇게 데이터의 일부분만으로 모수를 업데이트 해도 평균적으로 잘 된다는 게 알려진 사실이고, 그래서 이렇게 임의로 뽑은 일부의 데이터만 활용해서 모수를 갱신하는 방법을 SGD 방법이라고 하죠. SGD의 여러가지 변종들이 Adam이니 Adadelta니 하는 optimizer입니다. 따라서 mini-batch의 크기만큼 데이터를 계속 잘라서 network에 넣어주어야 하는데요, 이런 작업들을 쉽게 할 수 있도록 gluon에서는 NDArrayIterator라는 class를 제공합니다. 다음과 같이 사용합니다.

~~~
train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False)
valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False)
~~~

이렇게 하면 iterator로 정의한 것이므로, 메모리에 대한 걱정도 사라지게 됩니다.

이제 모형 관련된 준비 사항 및 데이터 관련 준비 사항도 모두 끝이 났습니다. 이제 이런 설정들을 바탕으로 실제 학습을 진행하면 됩니다. 다음은 실제 코드입니다.

~~~
for epoch in tqdm_notebook(range(n_epoch), desc = 'epoch'):
    ## Training
    train_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(train_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        with autograd.record():
            _out = mlp(_dat)
            _los = nd.sum(loss(_out, _label)) # 배치의 크기만큼의 loss가 나옴
            _los.backward()
        trainer.step(_dat.shape[0])
        n_obs += _dat.shape[0]
        #print(n_obs)
        _total_los += nd.sum(_los).asnumpy()
        # Epoch loss를 구하기 위해서 결과물을 계속 쌓음
        pred.extend(nd.softmax(_out)[:,1].asnumpy()) # 두번째 컬럼의 확률이 예측 확률
        label.extend(_label.asnumpy())
    #print(pred)
    #print([round(p) for p in pred]) # 기본이 float임
    #print(label)
    #print('**** ' + str(n_obs))
    #print(label[:10])
    #print(pred[:10])
    #print([round(p) for p in pred][:10])
    tr_acc = accuracy_score(label, [round(p) for p in pred])
    tr_loss = _total_los/n_obs

    ### Evaluate training
    valid_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(valid_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        _out = mlp(_dat)
        _pred_score = nd.softmax(_out)
        n_obs += _dat.shape[0]
        _total_los += nd.sum(loss(_out, _label))
        pred.extend(nd.softmax(_out)[:,1].asnumpy())
        label.extend(_label.asnumpy())
    va_acc = accuracy_score(label, [round(p) for p in pred])
    va_loss = _total_los/n_obs
    tqdm.write('Epoch {}: tr_loss = {}, tr_acc= {}, va_loss = {}, va_acc= {}'.format(epoch, tr_loss, tr_acc, va_loss, va_acc))
~~~

## 마치며

단순히 단어의 one-hot representation만으로도 성능이 높은 모형을 구축할 수 있었습니다. 이 데이터셋이 가장 entry level의 쉬운 데이터셋이기도 합니다만, 사실 text classification에서는 이정도의 technique로도 충분히 좋은 성능을 얻을 수 있다고 합니다. 다음 글에서는 CNN을 이용한 sentence representation을 해보도록 하죠.
