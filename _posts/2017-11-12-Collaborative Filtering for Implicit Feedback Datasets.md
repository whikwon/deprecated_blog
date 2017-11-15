---
title:  "Collaborative Filtering for Implicit Feedback Datasets"
date: 2017-11-12 00:00:00
comments: true
---

- Implicit Feedback에 Matrix Factorization을 적용한 논문이다. 학습 방법은 ALS를 사용하였다.

전체적인 내용이 앞서 정리한 [논문](https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E6%8E%A8%E8%8D%90/papers/Matrix%20Factorization%20Techniques%20for%20Recommender%20Systems.pdf)
의 내용과 많이 겹치며 explicit feedback 뿐 아니라 implicit feedback에도 성공적으로 적용한 내용을 주로 소개한다.

겹치는 내용은 제외하고 핵심적인 사항들만 적도록 하겠다.

## Implicit Feedback의 특징
1. **Negative feedback이 없다.**: user가 item에 얼마나 관심을 줬는지에 따라 선호도가 있는지 없는지를 알 수 있다고 가정한다.
그러므로 특정 item을 싫어한다는 feedback은 알 수 없다. 그래서 missing data에 대한 처리가 중요한데 이를
아직 보지 못한 item이라고 가정할 것인지 아니면 선호하지 않는 item인지에 따라 user에 대한 학습이 완전 달라질 수 있기 때문이다.
2. **Noisy하다.**: user의 행동이 일관되지 않고 상태에 따라 제각각이므로 noise가 데이터에 많이 존재하게 된다.
3. **값은 confidence를 의미한다.**: explicit feedback에서의 값은 *preference(선호도)* 를 의미했다. 하지만 implicit feedback에서의 값을
선호도로 놓기는 어려운데 noise가 많아 어느 정도 신뢰해야 할 지 판단이 어렵기 때문이다. 그래서 implicit에서는 *confidence(신뢰도)* 로 놓고
값이 높을 경우 선호도를 높은 확률로 가질 것으로 추측하고 낮을 경우 선호도를 낮은 확률로 가질 것이라고 추측한다.
4. **Implicit feedback 추천 시스템은 적절한 평가 방법이 필요하다.**: 평가하기 애매한 경우가 생긴다. 예를 들면 TV 프로그램에 대해서 조사했다고
했을 때 두 번 이상 본 프로그램이나 동시에 진행되는 프로그램은 어떻게 평가해야 할 지 애매하다. 이런 상황들에 대한 기준이 필요하다.
5. **데이터가 dense한 편이다.**: 엄청나게 많은 item 중 몇몇 개만 평가하는 explict에 비해 implicit은 더 많은 item들과 user가 상호작용을 하므로
데이터가 dense한 matrix 형태를 지니게 된다.

## 모델 소개
먼저 기존 목적 함수를 보면 latent factor로 user와 item의 특징을 나타내고 이 둘을 내적한 값을 예상 평점이라 하고 진짜 평점과의 차이를 최소화하는 방향으로
학습한다. 아래 식에서의 항은 $$r_{u,i}$$는 진짜 평점, $$x_u, y_i$$는 각각 latent factor로 이루어진 user, item vector이다.
<center> $$\underset {x_{\star}, y_{\star}} {min} \displaystyle \sum_{r_{u,i\ \text{is known}}} (r_{ui} - x_u^T y_i)^2 + \lambda (\lVert x_u \rVert^2 + \lVert y_i \rVert^2)$$ </center>

Implicit의 경우에는 평점이라는 개념이 없으므로 항을 약간 수정해야 한다. 먼저, 평점은 선호도($$p_{ui}$$)과 신뢰도($$c_{ui}$$)의 2개 항으로 나뉜다.
두 항 모두 우리가 얻은 feedback($$r_{ui}$$)로 부터 결정된다. $$r_{ui}$$는 구매한 횟수, 마우스 클릭 수, 브라우저에 머문 시간 등을 나타낸다.

선호도와 신뢰도를 살펴보자.
먼저, 선호도는 feedback의 유무로 결정된다. 있을 경우 1, 없을 경우 0으로 간주한다. 즉, user가 item에 관심을 보였을 때만 선호도가 있다고 가정하는 셈이다.
다음으로 신뢰도는 feedback의 크기에 따라 결정된다. 제품을 많이 구매했을 경우 여러 가지 noise(*선물을 주려고 구매하는 경우 등*)를 고려하더라도 선호도가 신뢰할만하다고 해석하는 셈이다.

이를 식으로 나타내면 아래와 같다. 위에서 해석한 대로 수식이 구성되며 신뢰도($$c_{ui}$$)의 경우는 hyperparameter인 $$\alpha$$가 추가되는데
이는 $$r_{ui}$$가 커짐에 따라 얼마나 가중치를 부여할 지 결정해준다. 논문에서는 $$\alpha = 40$$에서 좋은 성능을 낸다고 소개한다.

<center> $$p_{ui}=\left\{
            \begin{array}{ll}
              1\ \text{if}\ r_{ui} > 0 \\
              0\ \text{else}\ r_{ui} = 0
            \end{array}
            \right. \\c_{ui} = 1 + \alpha r_{ui}$$ </center>

목적 함수를 수정해서 다시 정리하면 아래와 같은 식을 얻을 수 있다.
<center> $$\underset {x_{\star}, y_{\star}} {min} \displaystyle \sum_{u, i} c_{ui}(p_{ui} - x_u^T y_i)^2 + \lambda (\sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2)$$ </center>

위에서 정의한 목적 함수를 Alternating Least Squares(ALS)를 사용해서 학습시킨다. $$x_u, y_i$$에 대해 update하는 식은

$$\begin{align} x_u &= (Y^T C^u Y + \lambda I)^{-1} Y^T C^u p(u) \\
y_i &= (X^T C^i X + \lambda I)^{-1} X^T C^i p(i)
\end{align}$$

으로 자세한 유도는 [stackoverflow](https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares/1073170#1073170)를 참조하자. 전체 $$m \times n$$의 matrix로 user-item 데이터가 이루어져 있다고 하고 각각의 user, item은 $$f$$개의 factor로 나타난다고 하자. $$\mathcal{N}$$은 데이터 중 non-zero observation의 개수이다.
그럼 ALS로 계산 시 총 연산량은 $$x_u, y_i$$에 대해 각각 $$O(f^2 \mathcal{N} + f^3 m), O(f^2 \mathcal{N} + f^3n)$$의 데이터의 size에 따라 linear한 연산량으로 계산이 가능하다.

## 정리
implicit feedback에 matrix factorization을 적용하기 위해 선호도($$p_{ui}$$)와 신뢰도($$c_{ui}$$)를 도입해서 학습을 시키는 것이 이 모델의
핵심이라고 할 수 있다.

Reference: <br>
[Yifan Hu. Collaborative Filtering for Implicit Feedback Datasets. 2008](http://yifanhu.net/PUB/cf.pdf)
