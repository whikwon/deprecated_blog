---
title:  "BPR: Bayesian Personalized Ranking from Implicit Feedback"
date: 2017-11-15 00:00:00
comments: true
---
- Implicit Feedback을 학습시키는 방법인 BPR를 소개한다.

## Introduction
논문에서 BPR이라는 ranking을 학습하는 알고리즘을 소개한다. BPR을 통해 아래 사항들에 대해서 기여를 할 수 있다.
1. 개인화된 ranking에 최적화된 optimization criterion(*목적 함수*)인 $$\normalsize {\text{BPR-O}} \small {\text{PT}}$$를 소개한다. $$\normalsize {\text{BPR-O}} \small {\text{PT}}$$가 AUC score의 최대화와 어떤 관련이 있는지 보일 예정이다.
2. $$\normalsize {\text{BPR-O}} \small {\text{PT}}$$를 최대화하기 위해서 $$\normalsize {\text{L}} \small {\text{EARN}} \normalsize {\text{BPR}}$$라는 학습 알고리즘을 제시한다.
$$\normalsize {\text{L}} \small {\text{EARN}} \normalsize {\text{BPR}}$$는 SGD와 bootstrap sampling을 이용해서 학습하는 방법이다.
3. $$\normalsize {\text{L}} \small {\text{EARN}} \normalsize {\text{BPR}}$$를 kNN과 Matrix Factorization 모델에 어떻게 적용할 수 있는지 살펴본다.
4. 개인화 ranking에서의 $$\text{BPR}$$의 성능이 기존 학습 방법 대비 좋다는 것을 보인다.

## 표기법 및 데이터 설정
$$U$$를 모든 user의 set, $$I$$를 모든 item의 set이라고 했을 때 implicit feedback은 $$S \subseteq U \times I$$라고 정의하자.
그리고 $$I$$에 속한 item들의 특정 feedback $$i, j$$ 값을 비교했을 때 $$i$$가 $$j$$보다 더 큰 경우 ***positive***, 작은 경우
***negative***, 같은 경우 ***missing value*** 라고 한다. (*이 세 가지로 모두 표현할 수 있다.*)
아래 그림에 잘 나타나있다. 왼쪽 그림처럼 user에 매칭되는 item이 matrix 형태로 주어지고 $$+$$는 값이 있음, $$?$$는 값이 없음을 나타낸다.
전체 matrix에 대해서 이제 user 별로 살펴 본다. 특정 user에 대한 전체 item 끼리의 선호도를 비교하면 위의 규칙에 따라 오른쪽 그림처럼 나타난다.
(*여기에서 item $$i$$를 $$j$$보다 선호한다라는 것을 수식으로 표현하면 $$i >_u J$$ 이다. 뒤에서 나오므로 기억해두자.*)

![BPR_figure2](https://whikwon.github.io/images/rec_BPR_figure2.png)
<center> <i> &lt;pairwise 선호도를 결정하는 방법 소개&gt;</i> </center> <br>

위 그림처럼 모든 user에 대해서 각 item의 다른 item들에 대한 선호도를 나타내면 우리가 학습 시에 사용할 데이터 $$D_S$$를 얻을 수 있다.
$$D_S := {(u,i,j) \lvert i \in I_u^+ \wedge j \in I \texttt{\\} I_u^+}$$

## BPR 목적 함수
우리가 구해야 할 것은 모든 item에 대한 개인화된 ranking이다. 이제 Bayes 정리로부터 어떻게 구할 수 있을지 살펴보자.
$$\Theta$$가 우리가 학습시킨 모델의 parameter를 나타낸다고 가정하면 Bayes 정리에 따라 $$p(\Theta \lvert >_u) \propto p(>_u \lvert \Theta) p(\Theta)$$
식을 얻을 수 있고 우리의 목표는 $$p(>_u \lvert \Theta)$$이므로 아래와 같이 정리가 가능하다.

$$\begin{align}
\prod_{u \in U} p(>_u \lvert \Theta) = &\prod_{(u,i,j) \in U \times I \times I} p(i >_u j \lvert \Theta)^{\delta((u,i,j) \in D_S)} \\ &\cdot (1 - p(i >_u j \lvert \Theta))^{\delta((u,j,i) \notin D_S)} \\
\end{align}$$

그리고 위 식에서 $$\delta$$는 indicator function로 우리는 $$D_S$$에 속할 경우 1, 나머지는 0으로 줘서 아래 식으로 정리가 가능하다.  
<center> $$\prod_{u \in U} p(>_u \lvert \Theta) = \prod_{(u,i,j) \in D_S} p(i>_u j \lvert \Theta)$$ </center>

그리고 이 확률 식을 $$\Theta$$에 대해 학습한 모델(kNN, MF...) 값의 sigmoid 식과 같다고 놓는다.
<center> $$p(i >_u j \lvert \Theta) := \sigma(\hat x_{uij}(\Theta))$$ </center>

이제 다시 처음으로 돌아가서 $$\normalsize {\text{BPR-O}} \small {\text{PT}}$$를 정의할 차례이다. 하나 빠진 부분이 있는데
prior인 $$p(\Theta)$$는 normal distribution으로 놓는다.($$p(\Theta) \sim N(0, \sigma_{\Theta}$$) 이제 구성 요소가 다 모였다.

$$\begin{align}
\normalsize {\text{BPR-O}} \small {\text{PT}} :&= ln\ p(\Theta \lvert >_u) \\
&= ln\ p(>_u \lvert \Theta)\ p(\Theta) \\
&= ln\ \prod_{(u,i,j) \in D_S} \sigma(\hat x_{uij})\ p(\Theta) \\
&= \displaystyle \sum_{(u,i,j) \in D_S} ln\ \sigma(\hat x_{uij}) + ln\ p(\Theta) \\
&= \displaystyle \sum_{(u,i,j) \in D_S} ln\ \sigma(\hat x_{uij}) - \lambda_{\Theta} \lVert \Theta \rVert^2
\end{align}$$

이대로 학습하면 개인화된 ranking에 최적화되게 학습할 수 있겠다.

## AUC optimization과의 논리적인 관계
AUC score를 구하는 식과 $$\normalsize {\text{BPR-O}} \small {\text{PT}}$$이 같은 논리를 가지고 있기 때문에 목적 함수를 학습시키면
최적의 AUC score를 구할 수 있다는 결론을 얻을 수 있다고 한다. 수식은 생략한다.
*찾다보니 AUC optimization가 많이 언급되는데 모델의 이게 무엇을 의미하는 지 모르겠다.*

![BPR_figure2](https://whikwon.github.io/images/rec_bpr_learnbpr.png)
<center> <i> &lt;LearnBPR을 통해 학습했을 때의 AUC score 변화&gt;</i> </center> <br>

## BPR Learning 알고리즘
위에서 목적 함수를 정의했으니 이제 학습 방법을 정해야 한다. 결론적으로 bootstrap sampling에 기반을 둔 SGD 알고리즘을 사용한다.
full GD를 사용하지 않는 이유는 연산량 문제와 skewness 문제로 인해서 사용하지 않는다고 한다. SGD를 사용했을 때 이들 문제가 해결되지만
user-wise나 item-wise로 샘플링했을 때 연속적으로 update되면서 나쁘게 수렴한다는 문제가 발생한다고 한다. 그래서 따로 user, item을 정하지
않고 전체에서 random bootstrap sampling으로 뽑는 방식을 취한다고 한다.

SGD를 이용해서 구한 $$\Theta$$에 대한 gradient는 아래 식과 같이 구해지고 이를 update하면서 학습이 진행된다.

$$\begin{align}
\dfrac {\partial \normalsize {\text{BPR-O}} \small {\text{PT}}} {\partial \Theta} &= \displaystyle \sum_{(u,i,j) \in D_S} \dfrac {\partial} {\partial \Theta} ln\ \sigma(\hat x_{uij}) - \lambda_{\Theta} \dfrac {\partial} {\partial \Theta} \lVert \Theta \rVert^2  \\
& \propto \sum_{(u,i,j) \in D_S} \dfrac {-e^{-\hat x_{uij}}} {1 + e^{-\hat x_{uij}}} \cdot \dfrac {\partial} {\partial \Theta} \hat x_{uij} - \lambda_{\Theta} \Theta
\end{align}$$

아직 모르는 내용이 하나 있다. 바로 $$\hat x_{uij}$$에 관한 내용이다. 이 부분은 앞서 간략하게 언급한 것처럼 kNN이나 MF와 같은 모델로부터 결정이
되며 그로부터 해당하는 식을 구할 수 있다. MF에 대해서만 살펴보도록 하자.

## Matrix Factorization으로 BPR 학습
먼저 $$D_S$$에 속하는 $$(u,i,j)$$ 대해서 $$\hat x_{uij} = \hat x_{ui} - \hat x_{uj}$$가 성립한다고 놓는다.
$$\hat x_{ui}, \hat x_{uj}$$는 특정 user의 특정 item에 대한 score의 예측 값이다. 즉, matrix factorization을 적용했을 때
학습 방식이 있고 $$\hat x_{ui}$$를 나타낼 수 있으면 $$\hat x_{uij}$$도 나타낼 수 있어서 이제 $$\normalsize {\text{BPR-O}} \small {\text{PT}}$$
학습하는 데에 문제가 없어진다.

## 학습 결과
마지막으로 학습 결과를 타 모델과 비교한 그래프이다. AUC score에서 BPR을 적용시킨 모델의 성능이 타 모델 대비 뛰어난 것을 확인할 수 있다.

![BPR_figure2](https://whikwon.github.io/images/rec_bpr_result.png)rec_bpr_learnbpr.png
<center> <i> &lt;BPR-모델 학습 결과&gt;</i> </center> <br>

Reference: <br>
[Steffen Rendle. BPR: Bayesian Personalized Ranking from Implicit Feedback. 2012.](https://arxiv.org/pdf/1205.2618.pdf)
