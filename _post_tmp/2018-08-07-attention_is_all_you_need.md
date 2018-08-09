---
layout: post
title: 'Transformer- Attention is All you need'
author: kion.kim
date: 2018-0x-xx 17:00
tags: [deeplearning, nlp, attention, skip connection, NMT]
---

## 기계 번역의 trend

기계 번역 전문가는 아니지만, 워낙 서로 다른 domain에서 개발된 방법론들이 융합하고 발전하고 있는 deep learning 세상에서 뒤쳐지지 안기 위해 여러가지 분야를 보고 있습니다. Facebook에서 개발된 neural machine translation 방법인 transformer를 이야기 하기 전에, 간단하게 trend를 말해보면... 최근 가장 많이 쓰이는 통계적 번역 방법이 seq2seq입니다. 이 seq2seq 방법은 RNN에 기반하고 있습니다. 입력된 문장을 thought vector로 압축시킵니다. Seq2seq 모형에서는 이렇게 만들어진 thought vector를 시작점으로 해서 출력 문장의 token을 하나씩 생성해 가는 형태를 지니고 있습니다. 최초에는 thought vector와 '<SOS>' 정보를 가지고, 첫번째 token을 만들고, 만들어진 token과 thought vector를 가지고 그 다음 token을 생성하고, 마지막에 '<EOS>' token이 나오면 문장생성을 마치는 그런 형태의 모형입니다. 인터넷에서 가지고 온 아주 유명한 그림입니다. 

![seq2seq_thought_vector](/assets/seq2seq_thought_vector.png)

문제는 thought vector입니다. 이 thought vector가 모든 정보를 충분히 담고 있느냐? 문장이 아주 길어지거나 복잡해지면, 수많은 문장들에 담겨 있는 서로 다른 모든 정보를 하나의 vector에 녹이는 것은 어려울 것입니다. 조경현 교수님의 논문에서 볼 수 있는 것처럼, 문장이 길어질수록 그 성능이 많이 떨어지는 것을 볼 수 있습니다.

![Bahdandau_attention](/assets/Bahdandau_attention_fllo8qp5n.png)

이런 문제를 해결하기 위해 제안 된 것이 attention mechanism입니다. 하나의 thought vector를 사용하는대신, 매 token을 생성함에 있어서 입력된 문장에서 어떤 부분을 더 중요하게 볼 지를 context vector로 표현을 하는 것입니다. 그 context vector를 생성하는 mechanism이 attention mechanism이라고 생각하면 될 것 같습니다. 아래의 그림은 번역을 하는 건 아니구요. 날짜의 format을 바꾸는 network인데요. 그 mechanism이 잘 표현되어 있습니다.

 ![seq_seq_att_example](/assets/seq_seq_att_example_6p4s05mu0.png)

Encoding에 RNN을 사용할 필요가 있을까? 이게 이 approach의 근본적인 질문입니다. 그래서 위의 seq2seq에서 RNN을 과감히 빼버림으로써, 번역의 성능은 그대로 유지하면서 속도를 아주 빠르게 하는 transformer라는 network를 구축할 수 있었던 것입니다. 최초의 attention mechanism은 극단적으로 문장의 전체를 보는 skip bigram과 문장의 일부분만을 보는 $n$-gram를 일반화하는 형태라는 motivation이 있었지만, transformer는 attention이라는 아이디만 차용한후 극도로 복잡하게 만들어버리는 computer scientist들의 특성이 반영되어, 결국에는 왜 잘 되는지는 잘 모르는 그러한 art의 영역으로 넘어가게 되어버렸습니다. 그러면 어떻습니까? 성능은 잘 나오잖아요? 세상은 바뀌는 거구요.


## Transformer의 building block

Transformer를 이해하기 위해서는 한 2가지 정도만 이해하고 있으면 됩니다. Attention과 positional encoding입니다. 이렇게 두개의 component만 이해하고 있으면, 나머지는 이들을 아주 복잡하게 쌓아올리는 그러한 단계입니다. 이때부터는 architecture와 computing power로 승부를 하는 것이죠.

### Multi headed Attention

Multi headed attention은 self-attention의 확장된 형태입니다. Multi headed는 self-attention이 여러번 한다고 생각하시면 되겠습니다. attention을 만들어내는 weight vector가 random하게 초기값이 설정될 것이고, 같은 input에 대해 attention network를 여러개 모형에서 학습시키면, 각각의 attention network는 문장의 서로 다른 특성을 반영할 것이라는 것입니다. 여러개의 attention을 합할 때에는 맨날 하는 것처럼 벡터를 이어붙인 후에 또 한번 network를 태우는 과정을 거칩니다.

![multi_headed_attention](/assets/multi_headed_attention_wacg2x7o4.png)

그러니깐 이를 이해하기 위해서는 self-attention에 대해서 잘 알면 될 것 같습니다. self attention은 여러가지 형태가 있습니다. 여기에서 사용되는 attention 방법은 scaled dot-product attnetion입니다.

![scaled_dot_product_attention](/assets/scaled_dot_product_attention.png)