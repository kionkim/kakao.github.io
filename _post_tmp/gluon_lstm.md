NOTE
모든 time step의 hidden state값을 알기 위해서는 rnn.LSTM을 사용할 수 없음
rnn.LSTM은 매 time step의 output을 돌려주나
매 time step의 hidden을 돌려주지는 않음 ( 맨 마지막 time step의 hidden layer만 돌려줌)
LSTMCell을 사용하면,
loop을 돌려서 매 time step의 LSTM을 구현해 주어야 하고
layer를 여러개 주려면, 또다시 loop을 구현해야 함



Actually, what I meant was about positive parts not negative. There may be performance improvements after the version updated. But, It depends on your models.
For example, There was Improved sparse SGD, sparse AdaGrad and sparse Adam optimizer speed on GPU by 30x.
Although there are many other improvements, it’s hard to figure out the total impact on your model reading the release notes. If you have the benchmark code, it is easy to see. (edited)



I meant the situation when it gets improved as version goes up. we need to compare versions of framework not to guarantee that there were bugs in newer version, but to prove how much my model gets imporved. Of course mxnet developers always release new version with technical release notes, it is not so easy to figure out how much it will improve my model. Once you have te benchmark code, there is no doubt about it. For example, you can say that, sparse SGD, sparse AdaGrad, and sparse Adam optimizer is 30x faster on gpu, but what if we didn't use one of those optimizers? It's up to you where you put your comments in. But I want to be clear about this point.