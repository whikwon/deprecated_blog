---
title:  "Batch Normalization"
date: 2017-07-17 00:00:00
comments: true
---
- 2015년 ICML 2015에 publish된 논문이다.

***
#### 1. introduction <br>
Deep learning에서 모델을 학습시킬 때 parallelism의 효율성을 살리기 위해 training시
batch 대신 Stochastic Gradient Descent(SGD)를 사용한다.
SGD는 간단하고 효과적인 대신 세심한 hyperparameter tuning이 필요한데 특히나 각 layer의
input이 모든 이전 layer의 parameter에 의한 영향을 받기 때문이다.
이렇게 layer내 input값이 영향을 받아 지속적으로 분포가 변해서 문제가 발생하는 것(saturation)
을 ***internal covariate shift*** 라고 한다.
쉽게 예시를 들자면 sigmoid activation function의 경우를 생각해보자. <br>
($$z=g(Wu+b)$$, $$g(x) = {1\over {1+exp(-x)}}$$, $$W$$는 Weight matrix, $$b$$는 bias,
$$u$$는 layer input이다.)
여기서, $$x = Wu + b$$ 만약 $$x$$가 0보다 커지거나 작아지면 $$g'(x)$$는 0으로 가고, layer가 쌓이면 마찬가지로
vanishing gradient에 의해 $$x$$가 saturated regime으로 이동하게 된다. 깊은 layer 학습이 진행됨에 따라 이렇게 input값의
분포가 saturated regime으로 $$x$$의 차원이 이동해서 convergence에 느리게 도달하게 된다.
그래서 이런 nonlinearity input의 분포를 안정되게 해줄 수 있으면 optimizer가 saturated regime에 빠질 일도
드물 것이고 training도 빨라질 것이라는게 논문의 주된 내용이다.



Reference: <br>
Ioffe, Sergey, and Christian Szegedy. ["Batch normalization: Accelerating deep network training by reducing internal covariate shift."](https://arxiv.org/pdf/1502.03167.pdf) arXiv preprint arXiv:1502.03167 (2015)
