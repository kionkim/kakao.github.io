---
layout: post
title: 'Sentence representation - RNN'
author: kion.kim
date: 2018-0x-xx 17:00
tags: [deeplearning, self-attention, nlp, sentence representation]
---

## 시작하며

RNN은 주로 NLP에서 많이 사용되는 모형입니다. RNN은 이외에도 여러가지 이전 관측치의 값이 다음 관측치의 값에 영향을 미치는 시계열류의 데이터를 모델링하기 위해 많이 사용됩니다. RNN 이전에는 주로 ARIMA, Markov Random Field 등으로 풀던 문제였습니다. 문장을 하나의 숫자열로 표현하는 것은, 앞에서도 언급한 바 있지만, 어떻게든 token을 숫자화시키고 그 token을 하나의 값으로 나타내는 과정입니다. 어떠한 방법이든 token의 정보, 그리고 그 token들이 가지고 있는 여러가지 관계성 등이 유지가 되기만 한다면, 어떠한 방법도 사용할 수 있습니다. 그 방법들 중에서 인간이 문장을 인지하는 방식을 묘사하는 방식으로 모형을 고안된 방법들이 있는데, 대표적인 예가 RNN과 CNN, 그리고 최근에 각광받고 있는 Attention mechanism입니다. 문장에 등장하는 embedding된 단어의 요약을 하나의  pattern으로 보고 그것을 인식하여 단순히 분류하고 있다고 한다면, 보다 인간이 문장을 이해하는 방식을 따라하므로써, RNN과 CNN은 더욱 성능이 좋은 모형을 만들어 낼 수 있습니다. Sentiment analysis를 넘어선 neural translation에서는 보다 복잡한 모형들이 필요한 이유이기도 합니다.
RNN은 주로 NLP에서 많이 사용되는 모형입니다. RNN은 이외에도 여러가지 이전 관측치의 값이 다음 관측치의 값에 영향을 미치는 시계열류의 데이터를 모델링하기 위해 많이 사용됩니다. RNN 이전에는 주로 ARIMA, Markov Random Field 등으로 풀던 문제였습니다. 문장을 하나의 숫자열로 표현하는 것은, 앞에서도 언급한 바 있지만, 어떻게든 token을 숫자화시키고 그 token을 하나의 값으로 나타내는 과정입니다. 어떠한 방법이든 token의 정보, 그리고 그 token들이 가지고 있는 여러가지 관계성 등이 유지가 되기만 한다면, 어떠한 방법도 사용할 수 있습니다. 그 방법들 중에서 인간이 문장을 인지하는 방식을 묘사하는 방식으로 모형을 고안된 방법들이 있는데, 대표적인 예가 RNN과 CNN, 그리고 최근에 각광받고 있는 Attention mechanism입니다. 문장에 등장하는 embedding된 단어의 요약을 하나의  pattern으로 보고 그것을 인식하여 단순히 분류하고 있다고 한다면, 보다 인간이 문장을 이해하는 방식을 따라하므로써, RNN과 CNN은 더욱 성능이 좋은 모형을 만들어 낼 수 있습니다. Sentiment analysis를 넘어선 neural translation에서는 보다 복잡한 모형들이 필요한 이유이기도 합니다.
RNN은 주로 NLP에서 많이 사용되는 모형입니다. RNN은 이외에도 여러가지 이전 관측치의 값이 다음 관측치의 값에 영향을 미치는 시계열류의 데이터를 모델링하기 위해 많이 사용됩니다. RNN 이전에는 주로 ARIMA, Markov Random Field 등으로 풀던 문제였습니다. 문장을 하나의 숫자열로 표현하는 것은, 앞에서도 언급한 바 있지만, 어떻게든 token을 숫자화시키고 그 token을 하나의 값으로 나타내는 과정입니다. 어떠한 방법이든 token의 정보, 그리고 그 token들이 가지고 있는 여러가지 관계성 등이 유지가 되기만 한다면, 어떠한 방법도 사용할 수 있습니다. 그 방법들 중에서 인간이 문장을 인지하는 방식을 묘사하는 방식으로 모형을 고안된 방법들이 있는데, 대표적인 예가 RNN과 CNN, 그리고 최근에 각광받고 있는 Attention mechanism입니다. 문장에 등장하는 embedding된 단어의 요약을 하나의  pattern으로 보고 그것을 인식하여 단순히 분류하고 있다고 한다면, 보다 인간이 문장을 이해하는 방식을 따라하므로써, RNN과 CNN은 더욱 성능이 좋은 모형을 만들어 낼 수 있습니다. Sentiment analysis를 넘어선 neural translation에서는 보다 복잡한 모형들이 필요한 이유이기도 합니다.

Gluon에서 LSTM을 어떻게 사용하는지에 대한 내용을 찾아보기는 쉽지 않습니다. 그리고 API의 document 자체도 그리 훌륭하지는 않지만, 예제도 거의 찾아볼 수 없습니다. RNN에 대한 기본적인 내용들은 이미 많은 곳에서 알려져 있으니, sentiment analysis 과정을 통해 gluon에서 어떻게 LSTM 등 RNN을 사용하는지를 중심으로 알아보겠습니다.


## Architecture

상상할 수 있는 구조는 아주 다양합니다. 단순히 hidden layer를 쓸 수도 있고, hidden layer를 여러층 사용할 수도 있을  것입니다. 각 time step의 output을 평균해서 사용할 수도 있을 것입니다. 본인의 기억을 위해 각각의 구조를 gluon으로 어떻게 반영하는지 정리해 보겠습니다. 그러면서 gluon LSTM API의 사용법에 대해서 자세히 기록하겠습니다.

## LSTM cell의 구조

많은 곳에서 이미 LSTM의 구조에 대한 정보는 얻을 수 있습니다. 그럼에도 불구하고 한번 주지해야 할 사실은 LSTM에는 이전 time step에서 2개의 정보를 활용한다는 사실입니다. 

### single hidden State

가장 먼저 알아볼 기본 구조는 다음과 같습니다. Token을 embedding 한 후에 이를 LSTM layer를 통과시킨 후, 그 결과물을 classifier의 입력으로 사용하는 구조입니다. 

![lstm_structure](/assets/lstm_structure.png)

최초의 cell state와 hidden state의 



RNN은 token을 순차적으로 만들어 나가는 방식으로 특정 단어를 20개 하면 될 모든 단어에 동일한 영향을 token(여기서는 단어)을 순서대로 다음 단어를 처리할 때 사용할 것입니다. recognition이는 단순한 token의 평균이나 합을 이용하는 간단한 방법들을 점점 사용 가장 단순하게 token을 평균내는 방법이 있습니다. 하지만, 