---
title: "WSABIE: Scaling Up To Large Vocabulary Image Annotation"
date: 2017-11-14 00:00:00
layout: post
excerpt: Implicit Feedback을 학습시키는 방법인 WSABIE를 소개한다.
categories: [Recommender system, Paper]
comments: true
---
먼저, 알고리즘에 나타난 중요 항들을 먼저 살펴보자.
- $$x_i$$: user 정보
- $$y_i$$: item 정보(e.g. 마우스 클릭 수)
- $$\Phi_I(x):\ \mathbb{R}^d \rightarrow \mathbb{R}^D$$: mapping 함수
- $$\Phi_W(i):\ \{1, ..., Y\} \rightarrow \mathbb{R}^D$$: mapping 함수
- $$f_i(x) = \Phi_W(i)^T \Phi_I(x) = W_i^T V x$$: user, item의 정보를 각각 mapping해서 D차원의 factor를 가지게 만든 뒤 둘을 내적하면
우리가 원하고자 하는 item에 대한 user의 선호도를 구할 수 있다는 의미.

아래 알고리즘을 보자. 1개 cycle만 보도록 하자.

1. 우리가 가진 rank가 높은 labeled data $$(x_i, y_i)$$로부터 random sampling 한다. $$y_i$$는 단지 $$y_i \in \{1, ... Y\}$$중에 label을 아는 하나의 데이터만 뽑은 것이다.
그리고 모델로 계산해서 $$f_{y_i}(x_i) = \Phi_W(y_i)^T \Phi_I(x_i)$$값을 구하자.
2. $$\bar y \in \{1, ... Y\} \\$$에서 annotation을 하나 random sampling 한다. 그리고 $$f_{\bar y}(x_i) = \Phi_W(\bar y)^T \Phi_I(x_i)$$값을
구한다.
3. labeled data는 항상 rank가 높으므로 $$f_{\bar y}(x_i) > f_{y_i}(x_i) - 1$$이거나 $$N \geq	Y - 1$$(*한 user에 대한 모든 item 정보를 다 본 것*)일 때까지 학습을 진행한다.
4. $$f_{\bar y}(x_i) > f_{y_i}(x_i) - 1$$인 조건에 부합하면 $$L \big(\big [\frac {Y-1} N \big]\big) \lvert 1 - f_y(x_i) + f_{\bar y}(x_i) \rvert_+$$ 의 gradient에
해당되는 값을 $$W_i, V$$에 update해준다. (*regularization term이나 bias도 추가될 시에 update*)

![algorithm](https://whikwon.github.io/images/rec_wsabie.png)



## 코드 분석
lightfm 패키지를 사용해서 구현했고 내부 코드에 어떤 내용이 있는지 살펴보도록 하자.

공통적으로 입력해줘야 하는 factor의 수나 regularization term을 제외했을 때 max_sampled라는 항만 추가된다.
이는 위 알고리즘에서는 $$y_i$$가 속한 집합의 개수에 해당되는 $$N$$을 hyperparameter로 두어서 더 적은 샘플링 수에 학습을 그만한다는 의미이다.
model의 학습이 어느 정도 되면 $$f_{\bar y}(x_i) > f_{y_i}(x_i) - 1$$인 값을 찾을 확률이 매우 낮아지고 계속 학습하면 overfitting도 생기고
학습 시간도 줄이기 위해서 추가한다고 볼 수 있다.

패키지에서 한 가지 특이한 점은 user_feature나 item_feature를 추가해줬을 때인데 살펴보자.

https://github.com/lyst/lightfm/blob/master/lightfm/_lightfm_fast.pyx.template 분석 필요


## Learning to Rank - BPR

implicit feedback matrix factorization 문제를 optimizing 하는 방법으로 BPR을 소개한다. BPR은 기존에 소개한 방법들과 조금 다른 접근 방식을 취하는데 user가 선호하는 item간의 ***rank*** 를 매긴다는 측면에서 그렇다. 직관적인 내용은 특정 방법을 통해 user, item를 latent factor로 나타냈을 경우에 상호 간의 score를 구할 수 있다. 이 score를 이용하면 item 간의 rank를 구할 수 있다는 내용이다.

일단 implicit feedback이 1 또는 0인 데이터로 나타난다고 가정하면 특정 user에 대해서 item 별로 관계를 알 수 있게 된다. 같은 값이면 missing value, 다른 값이면 1인 쪽이 positive 0인 쪽이 negative value를 갖게 한다.

BPR의 목적은 user에 대해 item i가 j에 비해 positive 일 확률을 구하는 것이다. ($p(i>_u j \lvert \Theta)$)

BPR은 몇 가지 문제점이 있어서 학습 알고리즘이 LearnBPR로 개선되었다.

$p(i>_u j \lvert \Theta)$은 0,1의 문제를 푸는 것이므로 sigmoid 함수를 사용하고 $\hat x_{uij}(\Theta)$에 대한 함수라고 정의해서
문제를 접근한다. 이 때, $\hat x_{uij}(\Theta)$를 kNN이나 matrix factorization에서 구한 score로 구성하는 것이 특징이다.

결국 SGD, ALS와 다른 기준인 BPR을 제시하는 것이고 score보다는 rank에 더 초점을 맞춰서 학습을 진행한다고 이해하면 되겠다.

학습 대상이 되는 데이터는 특정 user에 대해 item i가 j에 $$>_u$$한 데이터로 여기서 샘플링을 진행해서 학습시킨다.

정리하면 두 데이터 간 positive, negative를 결정하고 이를 바탕으로 score를 구하면 당연히 positive한 쪽이 더 크게 나올텐데 이 score를 가지고 rank가 더 높을 확률을 계산한다. 이 식을 가지고 학습시킨다.



Reference: <br>
