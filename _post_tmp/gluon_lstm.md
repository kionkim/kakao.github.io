NOTE
모든 time step의 hidden state값을 알기 위해서는 rnn.LSTM을 사용할 수 없음
rnn.LSTM은 매 time step의 output을 돌려주나
매 time step의 hidden을 돌려주지는 않음 ( 맨 마지막 time step의 hidden layer만 돌려줌)
LSTMCell을 사용하면,
loop을 돌려서 매 time step의 LSTM을 구현해 주어야 하고
layer를 여러개 주려면, 또다시 loop을 구현해야 함