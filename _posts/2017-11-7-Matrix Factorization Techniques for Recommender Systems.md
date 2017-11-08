---
title:  "Matrix Factorization Techniques for Recommender Systems"
date: 2017-11-7 00:00:00
comments: true
---

- 추천 시스템 분야에서 널리쓰이는 Matrix Factorization(MF)에 대해 전반적으로 소개하는 2009년 논문이다.
- Yehuda Koren님이 작성하셨으며 추천 시스템 분야에서 매우 유명한 분이다.
- 작성된지 꽤 오래되어 현재는 어떤 알고리즘을 많이 사용하고 있는지 조사가 필요하다.

## 추천 시스템 전략
추천 시스템에 사용하는 접근법은 크게 2가지로 content filtering과 collaborative filtering이 있다.
각각의 개념을 살펴보자.

1) Content filtering: item의 각각을 특징들을 묶어줄 수 있는 범주를 정하고 얼마나
유사한지 비교하는 방식으로 이를 통해 user가 특정 item을 좋아하거나 구매했을 경우 비슷한 item을 추천해주는 방식이다.
영화를 예시로 들면 장르, 출연 배우, 감독, 제작사 등에 따라 비슷한 영화들을 묶을 수 있고 disney의 신데렐라를
좋아하는 10대 여성이 있을 경우 같은 회사에서 비슷한 내용의 애니매이션을 추천해줄 수 있다.

2) Collaborative filtering: user와 item 간의 상호 의존성을 나타내는 데이터를 통해 아직
평가되지 않은 user, item 간의 관계를 예측하는 방식이다. 장점은 domain free라서 특정한 지식, 정보를 공들여서 모으지 않아도 되며 대체로
content filtering 방식보다 성능이 좋다. 단점은 cold start 문제가 있어서 새로운 user, item에 대해서 적용할 수 없다.
방법론은 2가지 ***neighborhood methods*** 와 ***latent factor models*** 로 나뉘어지며 이 논문에서 소개하는
Matrix Factorization은 후자에 속한다.
  - neighborhood method: item 혹은 user 간의 이웃을 찾아서 특정 user가 좋아하는 영화와 가까운 item을 추천해주거나 (*item oriented approach*)
  특정 user에 가까운 user들을 찾아서 그들이 좋아하는 영화를 추천해주는 방식이다. (*user oriented approach*) 아래 그림은 user oriented approach에 대한
  내용이다. <br><br>
  ![neighborhood method](https://whikwon.github.io/images/rec_neighborhood.png)
  <center> <i> &lt;Neighborhood method 소개&gt;</i> </center> <br>
  - latent factor model: item, user의 특징들을 모두 설명할 수 있는 잠재적인 factor가 존재한다고 가정한 뒤 이를 구해 item, user를 모두 설명하는 방식이다.
  아래 예시를 보면 영화와 사용자에 대한 특징을 2가지로 압축해서 2차원 평면상에 점으로 나타냈다. 유사한 특징을 가졌을 경우에 서로 가까이 존재할 것이며
  사용자가 해당 영화를 좋아할 것이라고 예상할 수 있다. 아래 그림에선 Gus가 Indepencedence Day를 The Lion King보다 좋아할 것이라고 예상할 수 있다. <br><br>
  ![neighborhood method](https://whikwon.github.io/images/rec_latent_factor.png)
  <center> <i> &lt;latent method 소개&gt;</i> </center> <br>

## 추천 시스템 데이터 종류
- explicit feedback: user의 item에 대한 선호도를 명시적으로 드러낸다. 데이터에 바로 선호도가 드러나기 때문에 다루기 쉽고 user는 몇 개의 item에 대해서만
평가를 하므로 보통 sparse matrix의 형태를 지닌다. 예시로 영화 평점/좋아요 등이 있다.
- implicit feedback: user의 item에 대한 선호도를 내재적으로 드러낸다. 선호도를 데이터를 바탕으로 추정해야 해서 다루기 어려우며 user의 행동여부에
기반을 둬서 explicit보다는 dense한 matrix의 형태를 지닌다. 예시로 검색 이력, 마우스 클릭 등이 있다.

## Matrix Factorization 방법
matrix factorization은 위에서 간략하게 소개한 latent factor model에서 주로 사용되는 방법이다.
방법은 간단하다. user와 item 각각을 정해진 factor수의 벡터로 나타내는 것이다. (*위 예시에서는 2차원 벡터.*) 벡터로 나타낸 경우에
둘 사이의 유사도를 측정할 수 있고 유사도가 높은 item을 user에게 추천할 수 있다.

이 개념을 바탕으로 다양한 방법으로 발전했으며 이 때  ***scalability(확장성), predictive accuracy(예측 정확도)*** 를 고려해야한다.
이는 user나 item이 많아졌을 때에도 잘 다룰 수 있는지와 얼마나 정확하게 예측할 수 있는지에 대한 내용이다.
아래에서 다양한 모델들에 대해서 살펴보도록 하자.

## 기본적인 Matrix Factorization 소개
먼저, notation을 정의하면 아래와 같다. <br>
$$f: \text{latent factor 차원 수} \\
q_i: \text{i번째 item vector}\ (q_i \in \mathbb{R}^f)\\
p_u: \text{u번째 user vector}\ (p_u \in \mathbb{R}^f)\\
r_{ui}: \text{i번째 item에 대한 u번째 user의 실제 선호도} \\
\hat r_{ui} = q_i^Tp_u: \text{i번째 item에 대한 u번째 user의 예측 선호도} \\
$$ <br>
위 식을 보면 우리는 user의 item에 대한 실제 선호도와 예측 선호도가 같은 방향으로 $q_i, p_u$를 학습시키면 좋은 결과를 얻을 수 있다.
아직 MF의 목적 함수를 소개하기 전에 SVD와도 비슷한데 사용하지 않는 이유에 대해 짚고 넘어가겠다. (*SVD도 latent factor를 이용해서 데이터를 나타낼 수 있다.*)
이는 SVD가 결측값이 많은 sparse matrix형태의 데이터를 다룰 수 없기 때문인데 이를 해결하기 위해서
결측값을 여러 가지 방법으로 채울 경우에는 연산량이 많아지고 부정확해져서 사용하지 않는다.

다시 MF로 돌아와서 위 notation을 통해 본 내용을 식으로 나타내면 아래와 같은 목적함수가 생기며 이를 학습시키면 원하는 결과를 얻을 수 있을 것 같다. <br>
$$\underset {q^{\star} p^{\star}} {min} \displaystyle \sum_{(u,i)\in \kappa} (r_{ui} - q_i^Tp_u)^2 + \lambda_q \lVert q_i \rVert^2 + \lambda_p \lVert p_u \rVert^2: \text{목적 함수.} \\
\displaystyle \sum_{(u,i)\in \kappa} (r_{ui} - q_i^Tp_u)^2: \text{user의 item에 대한 선호도의 실제 값과 예측 값의 차이} \\
\lambda(\lVert q_i \rVert^2 + \lVert p_u \rVert^2): \text{L2 regularization term}
$$

## 학습 방법(알고리즘)
위에서 목적함수를 정의했으니 이제 학습 방법을 정할 차례이다. 주로 Stochastic Gradient Descent(SGD), Alternating Least Square(ALS) 2가지 방법 중에 하나를 사용한다.

- SGD: 목적 함수에 대한 변수 $$q_i, p_u$$ 각각의 gradient를 구한 뒤 일정 부분 update 해주는 방식을 취한다. 수식 전개는 간단하니 넘어가도록 하겠다.
error를 $$e_{ui} = r_{ui} - q_i^T p_u$$ 나타내면 update 식은 아래와 같이 나타낼 수 있다. $$\lambda$$는 learning rate이다.
  - $$q_i \leftarrow q_i + \gamma \cdot (e_{ui} \cdot p_u - \lambda_q \cdot q_i)$$
  - $$p_u \leftarrow p_u + \gamma \cdot (e_{ui} \cdot q_i - \lambda_p \cdot p_u)$$

- ALS: 목적 함수에 대한 변수가 $$q_i, p_u$$ 두 개이므로 이는 non-convex 하다. 그래서 convex로 문제를 풀기 위해서 둘 중에 하나를 고정한 상태에서 번갈아서
update를 해주는 방식을 취한다. 이 때, 단순한 least square를 푸는 문제로 바꿀 수 있다. ALS보다 SGD가 단순하고 연산량도 적지만 몇 가지 경우에 대해서는
ALS를 선호한다고 한다. 첫째는 병렬 연산이 가능한 경우이고 둘째는 implicit data를 다룰 때 사용한다고 한다.
  - $$q_i = r_{ui} \cdot P \cdot(P^T\cdot P + \lambda_q I)^{-1}$$
  - $$p_u = r_{ui} \cdot Q \cdot(Q^T\cdot Q + \lambda_p I)^{-1}$$

## biases 추가하기
MF의 장점 중에 하나로 다양한 데이터에 맞춘 목적 함수 수정을 통해서 좋은 성능을 낼 수 있는 flexibility를 언급하고 있다. 여기에선
*biases* 추가를 통해 어떤 특성을 추가할 수 있는지 소개하고자 한다. 예시를 들면, user별로 봤을 때 동일한 item이라도 누구는 평점을
대체적으로 높게 주지만 반대로 대체적으로 낮게 주는 사람도 있다. (*반대도 마찬가지이다.*) 이러한 특성을 *biases* 항을 추가함으로써 고려할 수 있다.

특정 user, item에 대한 biase로 전체 평균적인 값(기준), 특정 item이 다른 item들 보다 평점이 높은지 낮은지,
특정 user가 다른 user들에 비해 평점을 좋게 주는지 낮게 주는지를 고려해서 이를 모두 더해준 값으로 사용한다. 식으로 나타내면
$$b_{ui} = \mu + b_i + b_u$$로 이를 예상 평점에 더해 최종 평점을 구한다. $$\hat r_{ui} = \mu + b_i + b_u + q_i^Tp_u$$

최종적으로 목적 함수에 update를 하면 아래와 같은 식을 구할 수 있다.
$$\underset {q^{\star} p^{\star} b^{\star}} {min} \displaystyle \sum_{(u,i)\in \kappa} (r_{ui} - \mu - b_u - b_i - q_i^Tp_u)^2 + \lambda_q(\lVert q_i \rVert^2 + b_i^2)+ \lambda_p(\lVert p_u \rVert^2 + b_u^2)$$

## 부가적인 input sources
위에서 아주 간략하게 소개했듯 collaborative filtering에는 *cold start* 문제가 존재한다. *cold start* 문제는 user, item 간
관계를 나타내 줄 정보가 부족해서 특정 user의 item에 대한 선호도를 파악하기 어렵다는 것을 나타낸다. 이러한 문제를 implicit feedback을
통해 user의 선호도 예측을 보완한다. 예를 들면 user가 클릭을 했는지 안 했는지가 선호도에 영향을 미친다고 가정하는 것과 user의 프로필을 정보가
특정 item을 선호할 것이라고 예측을 하는 경우가 있다. 예측 선호도 값인 $$\hat r_{ui}$$의 식을 구성할 때 기존 구성 값들에 해당 implicit 정보를 벡터로 나타내서
더해준 뒤 목적 함수를 통해 학습하면 된다. (*논문의 수식이 항들 사이에 기호가 빠진 상태로 적혀있어 넘어가도록 하겠다.*)

## Temporal dynamics
지금까지 우린 static한 데이터에 대해서 얘기했다. 실제 세상에서는 유행이 돌고 돌듯 시간에 따라 계속해서 user의 선호도가 변한다.
이처럼 앞에서 우리가 본 예측 선호도 $$\hat r_{ui} = \mu + b_i + b_u + q_i^Tp_u$$의 각각 항들을 살펴보고 필요 시 시간에 관한 항으로 바꿔줄 필요가 있다.
먼저, user의 선호도($$p_u$$)와 biases($$b_u$$)는 모두 시간에 따라 변한다. 유행에 따라서 찾는 item들의 장르나 특징들이 계속 바뀔 것이기 때문이다. 그리고 item의 biases($$b_i$$)도
시간에 따라 변한다. 유행을 타고 있는 item들은 어느 정도 기대 심리가 있기 때문에 좋은 평을 받을 가능성이 높고 유행이 진 item은 반대이기 때문이다.
여기서 item의 특징을 나타내는 $$q_i$$는 시간에 따라 변하지 않는데, item 고유의 장르나 내용, 특징들은 시간에 영향을 받지 않기 때문이다.
위 내용을 고려해서 예측 선호도를 다시 나타내면 $$\hat r_{ui}(t) = \mu + b_i(t) + b_u(t) + q_i^Tp_u(t)$$이다.

## Inputs with varying confidence levels
같은 user가 평점을 매기는 경우라도 모두 다 같은 가중치를 갖는다고 할 수는 없다. user의 상태도 평점을 매길 때마다 매 번 바뀔 것이고 어느 정도 분포를 따른다고
볼 수 있겠다. 특히나 implicit feedback의 경우는 더 한데 사용자가 지금 하는 행동이 우연인지 좋아서 하는 것인지 확신하기 어렵기 때문이다.
그래서 confidence(신뢰도)라는 개념을 도입해서 행동의 빈도에 따라서 단발성으로 끝나는 사건인지 아니면 정말 선호도가 반영된 것인지를 고려해준다.
이 내용을 반영해서 목적 함수를 수정하면 $$\underset {q^{\star} p^{\star} b^{\star}} {min} \displaystyle \sum_{(u,i)\in \kappa} c_{ui}(r_{ui} - \mu - b_u - b_i - q_i^Tp_u)^2 + \lambda_q(\lVert q_i \rVert^2 + b_i^2)+ \lambda_p(\lVert p_u \rVert^2 + b_u^2)$$ 로 나타낼 수 있고 내용은 ***더 신뢰도가 높은 사건의 cost에 가중치를 부여하자*** 라는
의미로 받아들이면 되겠다.

## Netflix prize competition
마지막으로 소개할 내용은 저자가 Netflix prize competition에 참가했을 때의 결과에 대한 것이다. 앞에서 소개한 기법들을 전부다 적용시켰을 때
아래 그림처럼 성능이 점차적으로 좋아졌다고 한다. 특히 temporal component가 중요한 역할을 했다고 한다. <br><br>
![neighborhood method](https://whikwon.github.io/images/rec_netflix_performance.png)
<center> <i> &lt;Tuning에 따른 Matrix factorization 모델의 성능&gt;</i> </center> <br>


[Yehuda Koren. Matrix Factorization Techniques for Recommender System. IEEE. 2009.](https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E6%8E%A8%E8%8D%90/papers/Matrix%20Factorization%20Techniques%20for%20Recommender%20Systems.pdf)
