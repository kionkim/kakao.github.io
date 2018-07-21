---
layout: post
title: 'Sentiment Analysis- Self Attention'
author: kion.kim
date: 2018-07-12 17:00
tags: [deeplearning, LSTM, nlp, sentence representation]
---

## 시작하며

LSTM을 이용해서 문장의 여러 특성들을 뽑을 수 있습니다. 지난 블로그들에서는 주로 hidden state의 정보를 이용해서 문장을 표현하는 코드들을 짜보았는데, 사실 hidden state의 정보 이외에도 각 time step의 ㅡ로을 이용해서 문장을 요약할 수도 있을 것 같습니다. 하지만 각 time step의 output은 seq2seq 문제에서 실제 그 진가를 발휘합니다. 이전 단계의 output이 그 다음 time step의 input으로 들어감으로써, 순차적으로 문장을 생성할 때 유용하게 사용됩니다. 나중에 nmt 쪽에서 살펴 보려고 합니다. 


## Self Attention

지금까지 여러가지 모형을 만들어 보면서 가장 아쉬웠던 점은 그동안의 모형들을 과연 그 모형이 어떠한 mechanism에 의해 작동하는지 확인을 할 수 없었다는 것입니다. 지금까지 다룬 데이터는 더군다나 너무 성능이 좋은 모형들로 구성이 되어 있어서 어느 모형이 어떻게 동작하는지 잘 알 수 없었습니다. 이런 문제점을 해결하기 위해 좋은 tool이 있습니다. 2017년 Zhouhan Lin 외 다수가 작성한 논문에서 최초로 소개된 Self-attention이라는 개념은, seq2seq에서 사용된 attention을 변형한 형태로, 각 token이 실제 분류 결과에 어떠한 영향을 미쳤는지를 attention의 weight 형태로 보여줍니다.


## Self Attention의 구조

논문의 아이디어는 대부분 하나의 그림으로 이야기됩니다. 다음은 원래 논문에 수록되어 있는 전반적인 아이디어를 나타내는, 논문 전체를 대변하는, 그림입니다.

 ![Zhouhan et al. (2017)](/assets/self-attention-structure.png)

결국 classifier의 입력으로 들어가는 데이터는 그림 상에서 M이라는 행렬입니다. M 행렬은 $r \times (2\cdot u)$의 크기를 지닙니다. 여기서 $r$은 우리가 몇개의 관점에서 문장을 요약할지를 나타내고, $u$는 LSTM layer의 hidden dimension입니다. Bidirectional LSTM이므로 $2\cdotu u$만큼의 크기가 된 것입니다. 최종적으로 $M$이라는 결과물을 얻기 위해서는 다음의 변환 과정을 거칩니다.

$$ M = A H = \left\{ softmax(W_2 \tanh (W_1 H^T)\right\}\cdot [H_{forward}, H_{backward}]$$

논문에 나와 있는 식을 한꺼번에 쓴 식이고, 이는 위의 그림에 이미 충분히 설명되어 있습니다. 그래도 다시 한번 설명해 보자면...

Embedding 과정은 NLP에서는 거의 필수 요소가 되어가고 있습니다. 미리 train되어 있는 embedding을 쓰는지, 해당 문제에 맞는 embedding을 network 안에서 직접 학습해서 쓰는지의 차이인 것 같습니다. Embedding의 dimension을 $e$라고 하겠습니다. 먼저 embedding layer를 통과히여 $e$ 차원의 벡터들로 표현된 token들은 LSTM layer를 2번 거칩니다. 하나는 순방향의 문장 의미를 해석하는 forward LSTM이고, 다른 하나는 역방향의 문장 의미를 해석하는 backward LSTM입니다. 두 LSTM layer는 각각 $u$ 차원의 hidden vector를 갖고 있습니다. 각각의 time step에서 문장을 요약하기 위해서 두개의 hidden 벡터를 이어 붙입니다 그래서 각각의 time step의 정보는 $2u$ 크기의 벡터로 요약이 되는 것이지요. 모든 time step을 이어붙여서 하나의 행렬 형태로 만들어진 것이 $H$라는 행렬입니다. $H$ 행렬의 크기는 $n \times 2u$가 되겠습니다. 결국 문장은 $n$개의 token으로 이루어져 있다고 가정하는 셈입니다.

> 개인적인 소회를 쓰자면, Computer science에서는 행과 열에 대해 신경을 많이 쓰지 않는 것 같습니다. 행과 열이 정확하게 일치하는 논문이나 글들을 본 적이 없는 것 같습니다. 그냥 찰떡같이 알아먹나 봅니다. 엄밀하게 행과 열을 맞추다 보면 일치하지 않아서 시간을 많이 보낸 경험이 참 많습니다. 

이런 $H$를 바탕으로 이제 각 time step에서 $2u$ 크기의 벡터로 요약된 정보들에 중요도(attention이라고 하면 이상할까요?)를 어떻게 주는지를 결정하기 위한 attention vector를 계산합니다.

Attention 벡터는 일단 $W_1$이라는 행렬을 곱해서 선형 변환을 진행합니다. 각각의 time step에서 Hidden state인 $2u$ 크기의 벡터 에서 $d_a$의 크기로 줄이는 작업을 진행합니다. 이렇게 $d_a$ 차원으로 줄어든 벡터에 비선형 변환 $\tanh$를 적용한 후에 다시 한번 $r$ 차원으로 줄입니다. 이렇게 $r$ 차원으로 줄어든 벡터는 주어진 time step에서의 token이 지니는 $r$개의 서로 다른 관점에서의 주요도라고 볼 수 있습니다. 이렇게 얻어진 $r\times n$ 행렬은 열의 방향으로 softmax를 적용시켜 결국 최종적인 $r$-hop self-attention vector를 얻게 됩니다.


## 실제 구현에서는...

Self attention을 구하기 위해서는 2개의 새로운 행렬이 등장합니다. $W_1$과 $W_2$는 각각 $d_a \times 2u$ $r \times d_a$의 크기를 지닙니다. 이 두 행렬은 각각 $2u$와 $d_a$를 input으로 하고, $d_a$와 $r$을 output으로 하는 dense layer의 weight값으로 볼 수 있습니다. 이 때 두 dense layer는 모두 bias term은 없어야겠죠.

이렇게 놓고 보면, 각각의 time step은 모두 독립적으로 볼 수 있습니다. $W_1$과 $W_2$는 각각의 time step에 따로 적용되는 것입니다. 구현 상에서는 $n$개의 time step을 example 개념으로 볼 수도 있을 것 같습니다. 그래서 다음과 같이 구현을 하게 됩니다.

```
class Sentence_Representation(nn.Block):
    def __init__(self, **kwargs):
        super(Sentence_Representation, self).__init__()
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        self.att = []
        with self.name_scope():
            self.f_hidden = []
            self.b_hidden = []
            self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
            self.drop = nn.Dropout(.2)
            self.f_lstm = rnn.LSTMCell(self.hidden_dim // 2)
            self.b_lstm = rnn.LSTMCell(self.hidden_dim // 2)
            self.w_1 = nn.Dense(self.d, use_bias = False)
            self.w_2 = nn.Dense(self.r, use_bias = False)

    def forward(self, x, _f_hidden, _b_hidden):
        embeds = self.embed(x) # batch * time step * embedding
        f_hidden = []
        b_hidden = []
        self.f_hidden = _f_hidden
        self.b_hidden = _b_hidden
        # Forward LSTM
        for i in range(embeds.shape[1]):
            dat = embeds[:, i, :]
            _, self.f_hidden = self.f_lstm(dat, self.f_hidden)
            f_hidden.append(self.f_hidden[0])
        # Backward LSTM
        for i in np.arange(embeds.shape[1], 0, -1):
            dat = embeds[:, np.int(i - 1), :] # np.int() necessary
            _, self.b_hidden = self.b_lstm(dat, self.b_hidden)
            b_hidden.append(self.b_hidden[0])
        
        _hidden = [nd.concat(x, y, dim = 1) for x, y in zip(f_hidden, b_hidden)]
        h = nd.concat(*[x.expand_dims(axis = 0) for x in _hidden], dim = 0)
        h = nd.transpose(h, (1, 0, 2)) # Batch * Timestep * (2 * hidden_dim)
        
        # get self-attention
        _h = h.reshape((-1, h.shape[-1]))
        _w = nd.tanh(self.w_1(_h))
        w = self.w_2(_w)
        _att = w.reshape((-1, h.shape[1], w.shape[-1])) # Batch * Timestep * r
        att = nd.softmax(_att, axis = 1)
        self.att = att # store attention 
        x = gemm2(att, h, transpose_a = True)  # h = Batch * Timestep * (2 * hidden_dim), a = Batch * Timestep * r
        return x, att
```
위에서 $h$는 LSTM에 의해서 요약되어 나온 hidden state을 의미하고 $n$개의 $2u$ 벡터가 됩니다. 이 벡터를 바탕으로 `w_1` layer를 적용시킬 때에는 각각의 time step을 하나의 example로 보기 위해 `reshape`을 진행해 줍니다. attention을 구할 때는 Batch size($B$라고 표현하겠습니다.!)만큼의 example이 있는 것이 아니라, $B\times n$만큼의 example을 network의 input으로 활용합니다.

## 결과물

Attention을 시각해 하기 위해서 다음의 함수를 정의합니다.

```
def plot_attention(net, n_samples = 10, mean = False):
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    idx = np.random.choice(np.arange(len(va_x)), size = n_samples, replace = False)
    _dat = [va_x[i] for i in idx]
    
    w_idx = []
    word = [[idx2word[x] for x in y] for y in _dat]
    original_txt = [va_origin[i] for i in idx]
    out, att = net(nd.array(_dat, ctx = context)) 

    _a = []
    _w = []
    for x, y, z in zip(word, att, original_txt):
        _idx = [i for i, _x in enumerate(x) if _x is not 'PAD']
        _w.append(np.array([x[i] for i in _idx]))
        _a.append(np.array([y[i].asnumpy() for i in _idx]))
        
    _label = [va_y[i] for i in idx]
    _pred = (nd.sigmoid(out) > .5).asnumpy()
    
    fig, axes = plt.subplots(np.int(np.ceil(n_samples / 4)), 4, sharex = False, sharey = True)
    plt.subplots_adjust(hspace=1)
    if mean == True:
        fig.set_size_inches(20, 4)
        plt.subplots_adjust(hspace=5)
    else:
        fig.set_size_inches(20, 20)
        plt.subplots_adjust(hspace=1)
    cbar_ax = fig.add_axes([.91, .3, .04, .4])
    
    
    
    for i in range(n_samples):
        if mean == True:
            _data = nd.softmax(nd.array(np.mean(_a[i], axis = 1))).asnumpy()
            sns.heatmap(pd.DataFrame(_data, index = _w[i]).T, ax = axes.flat[i], cmap = 'RdYlGn', linewidths = .3, cbar_ax = cbar_ax)
        else:
            sns.heatmap(pd.DataFrame(_a[i], index = _w[i]).T, ax = axes.flat[i], cmap = 'RdYlGn', linewidths = .3, cbar_ax = cbar_ax)
        axes.flat[i].set_title('Label: {}, Pred: {}'.format(_label[i], np.int(_pred[i])))
```
$r$개의 attention을 각각 plotting할 수도 있지만, $r$개의 attention을 모두 더한 후에 softmax를 다시 적용해서 새로운 attention 벡터를 구할 수도 있게 했습니다. $r$ 개의 벡터를 그려봤는데, 워낙 짧은 문장이어서 그런지 별로 insight가 보이지는 않았습니다. $r$개의 attention 벡터를 구했던 이유도 애초에 문장이 길어지거나 복잡해질 때 여러가지 관점에서 문장을 요약하는 목적이었습니다. 이 문제에는 해당하지 않는다고 보이네요.

어쨌든 random하게 고른 12개의 결과는 다음과 같습니다.

![self-attention-result-example](/assets/self-attention-result-example.png)

몇몇 결과에서 `hate`, `suck`라는 단어가 나오면 negative sample로 분류하는 것을 볼 수 있습니다. 반면 positive sample로 분류할 때는 그렇게 큰 특징을 볼 수 없습니다. 몇번의 다른 시도에서는 `awesome`, `want`등의 단어들이 positive sample로 분류될 때 높은 attention을 보이는 것을 보기도 했습니다. 보다 많은 샘플들을 이용해서 오랫동안 학습하면 좀더 나아질까요?

다음 글에서는 relation network에 근거한 self-attention에 대해서 구현해보고 알아보겠습니다.


