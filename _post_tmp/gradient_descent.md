

## 시작하며



Deep learning을 공부하면서 별 고민 없이 당연히 받아들이는 개념들이 참 많습니다. 지난 포스트에서 살펴본 convolution kernel에 대한 내용이 그랬습니다. 당연히 convolution은 kernel을 한칸씩 혹은 몇칸씩 움직이면서 convolution 연산을 차례대로 해나가는 것이라고 막연히 생각했지만, convolution의 개념을 확장해 가는 과정 (예를 들면 deconvolution, dilated convolution 등)에서 점점더 기본적인 convolution의 정의를 다시 되짚어야 했던 경험이 있습니다. 결국 한발짝 나서기 위해서는 기본적인 정의에 대한 이해가 가장 중요한 것이죠.

지난 주말에 다시 공부하다가 graident descent를 살펴보면서 문득 왜 Gradient vector는 목적함수를 가장 가파르게 증가시키는 방향을 의미하는지가 궁금했습니다. 사실, 이렇게 알려진 것은 하나의 정리로, 과연 어떤 reasoning을 통해 gradient 방향이 왜 그러한 방향인지에 대해서 생각해보다가 처음부터 끝까지 정리된 자료가 그렇게 많지는 않다는 것을 깨달았습니다. 자세히 생각해 보면, 함수의 input이 있고, 함수를 vector 각각의 원소에 대해 편미분을 한 후에 원래의 point에서 적당히(learning rate를 적용합니다.) 빼주면 loss function을 작게 하는 방향으로 input이 변화한다는 게 별로 당연해 보이지는 않습니다. 이에 대해서 좀더 자세하게 알아보고 싶습니다.



## 적절한 setup

기본적으로 $p$개의 feature가 있고, 그 feature의 함수가 우리가 최적화하고자 하는 목적함수(Object function)입니다. 함수의 결과물도 하나의 실수로, 결국 우리는 $p+1$ 차원의 vector 공간을 상정합니다. 
$ y = L(x_1, \ldots, x_p), \textrm{ , where } \{x_1, \ldots, x_p, y \} \in \mathbb{R^{p+1}}$

수학에서는 hyperplane이라는 개념이 있습니다. Wiki에는 다음과 같이 정의되어 있습니다.

> In geometry, a hyperplane is a subspace whose dimension is one less than that of its ambient space.

단지 전체 차원에서 1차원 작은 subspace를 hyperplane입니다. 3차원 공간에서의 hyperplane은 2차원입니다. 2차원 평면에서는 line이 hyperplane이 됩니다. subspace가 되기 위해서는 꼭 원점을 포함해야 합니다. 

## 2차원 공간부터...

우리가 제일 익숙한 것이 $x$, $y$ 축으로 표현된 직교 좌표계입니다. 이 좌표계에서 가장 먼저 시작해 보죠. 반복해를 찾아가는 방법을 이해할 수 있습니다. 

2차원 공간을 정의하는 방법은 무수히 많습니다. 원점 0을 지나는 겹치지 않는 어떤 2개의 직선으로도 2차원 공간을 표현할 수 있습니다. 아래의 그림에서 원점을 지나는 두 직선 $a$와 $b$는 그 둘의 선형 결합으로 평면위의 어떠한 점도 표현할 수 있습니다.

![2d_coord](/assets/2d_coord.png)

그림으로야 직선을 $a$와 $b$%로 표시할 수 있지만 숫자로는 어떻게 표현할까요? 그렇게 하기 위해서는 어떤 기준이 필요합니다. 그 기준을 표준 기저라고 이야기 합니다. 2차원 평면에서 표준 기저는 $[0,1], [1, 0]$이 됩니다. 그리고 이렇게 표준기저로 두면 어떠한 직선이든 다음과 같이 표현할 수 있습니다.
$$a \cdot [1,0] + b \cdot [0,1] = [a,b]$$



선형대수에서는 직선을 원점으로부터 시작되는 벡터로 표현할 수 있습니다. 예를 들어, 직선 $a$는 다음과 같이 
직교 좌표계에서는 서로 직각인 2개의 직선으로 표현이 됩니다. 2차원 공간에서의 hyper-plane은 정의에 따라 1차원 직선입니다.  여기서는 Newton-Raphson method의 idea를 얻을 수 있습니다. 
## 3차원 공간에서...

Linear algebra는 3차원 공간에서의 현상을 바탕으로 이해하면 쉽습니다. 그 이상의 고차원은 우리가 상상할 수 없기 때문입니다. 따라서 많은 예제들이 3차원 공간에서 2차원 hyper-plane을 상정합니다. 3차원 공간에서 생각해보도록 하겠습니다.




### 등위 공간, hyper-plane, 법선


## $n$ 차원 공간에서는



## Taylor expansion


