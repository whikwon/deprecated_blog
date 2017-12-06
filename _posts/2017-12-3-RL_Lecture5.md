---
title:  "RL_Lecture5: Model-Free Control"
date: 2017-12-5 00:00:00
comments: true
---

- David Silver의 Reinforcement Learning 5강을 정리한 내용이다.

이번 장은 MDP가 주어지지 않았을 때(*model-free*) 어떤 방식으로 문제를 해결할 지에 대해 다룬다. 앞 장에서 Policy evaluation에 대한 내용인 Prediction을 다루었고
이번 장에서는 Policy improvement에 대한 내용인 Control에 관해 다룬다. 4강과 마찬가지로 크게 Monte-Carlo 와 Temporal-Difference learning을 배우고 On-policy와
Off-policy의 차이점을 배우는 것이 목표이다.

***
### Model-Free Control?
엘리베이터, 로보컵 축구, 바둑, 헬리콥터, 바이오리액터 등 많은 것들이 MDP로 모델링될 수 있다. 근데 현실 세계의 대부분 문제들은 MDP model을 모르거나 알지만 너무 크다는 문제점이 있다.
이런 경우에 전체 state, action을 다 아는 상태로 문제를 푸는 것보다 더 효율적으로 sampling을 통해 푸는 방법을 찾아야 하는데 ***Model-free control*** 접근해서 풀 수 있다.

***
### On, Off-Policy Learning?
On-policy learning은 "Learn on the job", $$\pi$$로부터 sampling된 경험으로부터 $$\pi$$에 대한 policy를 배우는 과정이다. <br>
Off-policy learning은 "Look over someones's shoulder", $$\mu$$로부터 sampling된 경험으로부터 $$\pi$$ 대한 policy를 배운 과정이다.

On-policy의 경우에는 자연스럽지만 Off-policy의 경우는 좀 이상해보인다고 느낄 수 있다. 다른 policy의 경험을 통해 원하는 특정 policy를 학습한다?
Off-policy가 필요한 이유는 원하는 policy의 경험을 실제로 얻기가 너무 어려운 경우가 있기 때문인데 예를 들면, 로봇을 학습시키기 위해서 끝없이 움직여야 한다면
오랜 세월이 걸릴 것이다. 대신 다른 경험으로부터 로봇이 배울 수 있다면(simulation) 훨씬 편하게 학습할 수 있을 것이다. 이런 이유로 필요한게 Off-policy method이며
자세한 내용은 아래에서 다룬다.

***
## On-Policy Monte-Carlo Control

### 1) Policy Iteration Using Action-Value Function
이전 장에서 본 policy iteration을 보면 policy evaluation으로 state-value function을 구하고 greedy하게 policy improvement하는 과정으로 진행되었다.
(*계속 하다보면 언젠간 수렴*) 근데, state-value를 구한 뒤에 policy로 update하기 위해서는 아래 식에서 보듯이 모든 action에 대해서 알아야 한다. 이는 MDP를 알아야 된다는
뜻이므로 model-free와는 거리가 멀다. 그래서 state-value 대신 action-value function을 사용해서 policy improvement를 해서 이 문제를 해결한다.

![model free](https://whikwon.github.io/images/david_silver/model_free_Q.png)

policy iteration 과정을 그림으로 나타내면 아래와 같으며 state-value function과 마찬가지로 optimal policy, action-value function에 수렴한다.

![policy iteration with Q](https://whikwon.github.io/images/david_silver/policy_iteration_Q.png)

### 2) $$\epsilon$$-greedy Exploration
복잡한 상황 속에서 연속적으로 어떤 결정을 내린다고 했을 때 현재 상태에서만 고려해서 최선(*greedy*)의 방향으로 결정한다고 생각해보자. 이런 상황을 반복하면서
학습하면 금방 많은 action들에 대해 순서가 확실해질 것이고(*deterministic*) 이는 전체 action들 조합의 경우의 수 중에 아주 일부분만 사용하면서 학습하게 될 것이다.
그리고 학습이 완료되었을 때 못 본 조합이 너무도 많기 때문에 우리가 구한 optimal policy가 최적인지에 대한 답을 하기 어렵다.

그래서, 더 많은 조합을 탐험하기 위한 간단한 방법을 생각해냈는데 작은 확률로 greedy하지 않은 action을 취하게끔 하는 것이다.
이를 식으로 나타내면 아래와 같다.

![epsilon greedy](https://whikwon.github.io/images/david_silver/epsilon_greedy.png)

그럼, $$\epsilon$-greedy를 통해서도 policy improvement가 잘 되는지 확인 해야한다. 결론적으로 아래 theorem에 의해 policy improvement가
잘 되는 것을 확인할 수 있다.

![epsilon greedy theorem](https://whikwon.github.io/images/david_silver/epsilon_greedy_theorem.png)
ㅇ
### 3) Monte-Carlo Policy Iteration
위 내용들을(*action-value function, $$\epsilon$$-greedy*) 반영하면 아래와 같이 나타낼 수 있다. 여기서 policy evaluation에 대해
생각해보면 value function을 얻기 위해서 너무 많은 episode들을 봐야 한다. 그냥 매 episode마다 policy evaluation을 하면 안 될까?

![MC policy iteration](https://whikwon.github.io/images/david_silver/MC_policy_iteration.png)

가능하다. 아래 그림처럼 나타낼 수 있고 이런 식으로 update 하더라도 optimal action-value function, policy에 수렴한다.
MC에 어떤 식으로 적용하는지 상세하게 알아보도록 하자.

![MC policy iteration](https://whikwon.github.io/images/david_silver/MC_policy_iteration2.png)

MC에는 GLIE를 적용해서 사용하는데 무한히 exploration 했을 때 $$\epsilon$$값이 점차 줄어들어 최종적으로 greedy policy를 갖도록 하는 알고리즘이다.
이는 최종적으로 optimal action-value function에 수렴한다.(*아래 Theorem*)

![GLIE](https://whikwon.github.io/images/david_silver/GLIE.png)
![GLIE](https://whikwon.github.io/images/david_silver/GLIE_MC.png)

### 4) 예시: Blackjack
아래는 MC control로 blackjack 문제를 푼 결과이다. optimal value function 뿐 아니라 policy도 구해서 주어진 rule에서 어떻게 행동했을 때
가장 돈을 많이 벌 수 있을지 보여준다. 복원추출로 진행했을 때의 결과이기 때문에 실제 카지노에서 비복원추출로 진행되면 더 복잡해질 것이다.

![MC blackjack](https://whikwon.github.io/images/david_silver/MC_blackjack.png)

***
## On-Policy Temporal-Difference Learning

### 1) MC vs TD Control
앞 장에서 policy evaluation할 때 TD learning은 MC보다 장점이 많이 있다고 배웠다. variance가 낮고 online 학습이 가능하며 episode가 반드시 끝나지 않아도 된다는 점들이 있다.
그럼 단순하게 생각해보면 policy evaluation 단계에서 MC를 TD로 바꿔서 하면 어떨까? 라고 생각할 수 있다. improvement는 그대로 $$\epsilon-greedy$$로 놔두고
1-episode 당 update 되었던 MC를 1-time step으로 바꿔서 학습하면 되겠다. 이 생각대로 내용을 전개해보자.

### 2) Sarsa
MC의 update 식 $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \dfrac 1 {N(S_t, A_t)} (G_t - Q(S_t, A_t))$$에서 아래 update 식으로 바뀌게 된다.
식을 살펴보면 현재 state($$S$$), action($$A$$)와 다음 step으로 갈 때의 reward($$R$$), 다음 step에서의 state($$S'$$), action($$A'$$)에 대한 항들로 식이 구성된다.
이 알고리즘은 **Sarsa** 라고 부르며 이름은 각 항들의 첫 글자를 순서대로 이었을 때의 단어이다.

![sarsa](https://whikwon.github.io/images/david_silver/sarsa.png)

policy iteration의 과정을 보면 MC와 비교했을 때 매 time-step마다 update 한다는 점과 $$Q$$ 관한 식만 다르고 나머지 과정은 같다.

![sarsa](https://whikwon.github.io/images/david_silver/sarsa2.png)

아래는 Sarsa의 알고리즘이다.

![sarsa](https://whikwon.github.io/images/david_silver/sarsa3.png)

MC에서도 봤듯이 Sarsa로 했을 때 optimal action-value function으로 수렴하는지 확인이 필요하다. 아래 theorem과 같이 여러 조건 하에서
optimal action-value function으로 수렴한다.

![sarsa](https://whikwon.github.io/images/david_silver/sarsa_theorem.png)

### 3) 예시: Windy Gridworld
바둑판에 말이 있고 도착지점까지 이동하는 상황이다. state는 위치, action은 상하좌우 움직임, reward는 이동 시 -1로 주어진다.
바둑판 내 특정 위치에 바람이 계속 불고 있어서 움직임에 영향을 받게 된다. sarsa로 학습하면 아래처럼 처음에는 도착지점까지 도달하는데
아주 많은 step이 소요되는 것을 볼 수 있다. 하지만 계속 반복되면서 곧바로 도착지점으로 이동하는 것을 볼 수 있다.

![windy gridworld](https://whikwon.github.io/images/david_silver/windy_grid.png)

### 4) n-Step Sarsa
sarsa에 단순히 TD를 적용한 것처럼 앞 장에서 살펴 본 n-step TD도 동일하게 적용할 수 있다. 이 때도 마찬가지로 n이 무한대로 갔을 때 MC로 표현할 수 있다.

![n-step sarsa](https://whikwon.github.io/images/david_silver/n_sarsa.png)

이 외에도 n-step TD를 다 합쳤던 TD($$\lambda$$)의 개념도 Sarsa($$\lambda$$)로 확장할 수 있다. 바뀐 점은 state-value function이 아닌 action-value
function에 대해서 update한다는 점만 다르다. 다루었던 내용이므로 넘어가도록 하겠다.

forward에 대한 설명이다.

![forward sarsa](https://whikwon.github.io/images/david_silver/forward_sarsa.png)

backward에 대한 설명이다. ***eligibility trace*** 를 사용해서 과거 정보 중에 현재 상태에 대한 영향을 반영해준다.

![backward sarsa](https://whikwon.github.io/images/david_silver/backward_sarsa.png)

학습 알고리즘을 나타내면 아래 pseudo code와 같다.

![sarsa algorithm](https://whikwon.github.io/images/david_silver/sarsa_algorithm.png)

1-step Sarsa로 학습시키는 경우에 가운데 그림처럼 학습 초기 단계에 value function은 오직 도착지점과 아주 가까운 부분만 update 되어 비효율적이다.
하지만 Sarsa($$\lambda$$)로 학습시키는 경우에는 오른쪽 그림과 같이 가중치를 받아서 영향을 주는 과거의 정보들을 고려해서 함께 update하기 때문에
더 빠르게 학습시켜 문제를 풀 수 있게 된다.

![windy gridworld](https://whikwon.github.io/images/david_silver/windy_grid_sarsa.png)

***
## Off-Policy Learning
학습 시에 target policy와 behavior policy가 다른 과정을 **Off-Policy Learning** 이라고 한다. 중요한 이유는 아래와 같다.
- 다른 agent의 policy를 통해서 배울 수 있다.
- old policy를 학습에 다시 사용할 수 있다.
- Policy를 *exploratory* 하게 따르면서 *optimal* 한 값을 배울 수 있다. (*on-policy에서는 exploratory하면서 optimal할 수 없는 딜레마가 있다.*)
- 하나의 policy를 따르면서 *multiple* policy를 배울 수 있다.

### 1) Importance Sampling
importance sampling은 원하는 분포에 대한 expectation을 다른 분포에 대한 expectation을 통해서 추정하는 내용이다. Off-policy를 통해서
하려는 목적과 동일하므로 importance sampling을 적용시켜볼 수 있다.

![importance sampling](https://whikwon.github.io/images/david_silver/importance_sampling.png)

target policy($$\pi$$)와 behavior policy($$\mu$$)가 다르다고 가정했으므로 둘 사이의 관계에 importance sampling을 적용시켜보자. 우리는 $$\pi$$에 대한
return의 expectation 값을 구하는게 목적이다. (*value function*) 하지만 $$\mu$$를 통해서 구할 것이기 때문에 위에서처럼 $$\dfrac {P(X)} {Q(X)}$$
항을 추가해준다. off-policy learning에서  $$P(X)$$는 $$\pi$$, $$Q(X)$$는 $$\mu$$가 될 것이다. 이렇게 새로운 항을 고려해주고 이 때 구하는 return 값을
$$G_t^{\pi/\mu}$$라고 놓고 학습하면 된다.

MC에 적용시켰을 때는 아래와 같은데 MC와 동일하게 전체 trajectory를 고려하므로 variance가 너무 커지는 문제가 생긴다.

![importance sampling MC](https://whikwon.github.io/images/david_silver/importance_sampling_MC.png)

variance를 낮추기 위해서 TD에 적용도 가능하며 동일하게 bootstrapping 항에 weight로 $$\pi, \mu$$를 고려한 값을 곱해주면 된다.

![importance sampling MC](https://whikwon.github.io/images/david_silver/importance_sampling_TD.png)

***
### 2) Q-Learning
Off-policy를 위해 사용되는 다른 방법인 Q-learning을 소개한다. (*importance sampling과 비교해서 어떤 점이 좋고 나쁜지는 확인이 필요하다.*)
기존 TD와 비교했을 때 되게 직관적으로 behavior, target policy를 고려해서 항을 변경해준다. 아래 식을 보면 실제 다음 action에 대해서는 $$\mu$$
를 통해서 얻지만 $$Q$$를 update할 때 고려하는 다음 action은 $$\pi$$를 통해 얻는다. $$\mu$$를 통해 행동하지만 이를 통해서 $$\pi$$를 최적화하게 되므로
위에서 말한 off-policy의 내용과 동일하다는 것을 알 수 있다.

![q-learning](https://whikwon.github.io/images/david_silver/q_learning.png)

policy improvement도 살펴보면 $$Q$$를 구한 뒤에 우리의 목적인 $$\pi$$를 update하게 된다. 여기서 on-policy와 같이 $$\epsilon$$을 고려할 필요가 있을까?
없다. 왜냐면 실제 행동은 $$\mu$$가 하기 때문에 $$\mu$$ ***exploratory*** 하게 만들어주기 위해 $$\epsilon$$-greedy를 적용시켜야 하고 $$\pi$$는 말 그대로 ***optimal*** 한
행동만 생각하면 되므로 greedy로 개선하면 된다. (*on-policy에서 많은 trajectory를 보기 위해서 적용했던 $$\epsilon$$-greedy이 target에 더이상 필요가 없어졌다.*)

![q-learning](https://whikwon.github.io/images/david_silver/q_learning2.png)

Q-learning도 수렴하는지 확인해야하고 아래 theorem에 따라서 optimal action-value function으로 수렴한다.
![q-learning](https://whikwon.github.io/images/david_silver/q_learning3.png)
![q-learning](https://whikwon.github.io/images/david_silver/q_learning4.png)

### 3) 예시: Cliff walking
아래 예시는 cliff walking으로 state는 위치, action은 상하좌우 이동, reward는 절벽으로 떨어졌을 때 -100, 나머지는 -1로 주어진다.
이 조건에서 Sarsa와 Q-learning으로 초반 학습 과정을 비교하면 아래와 같다. 시작 단계에서 Sarsa가 Q-learning보다 reward가 훨씬 높게 나타나는 것을
볼 수 있다. 왜냐면 이전에 MC와 Sarsa와 비교에서도 그랬듯 다음 step만을 고려하기 때문에 cliff에 *떨어질 뻔* 하면 이 정보를 update하기 때문에
좀 안전한 길로 최적화한다. 하지만 Q-learning의 경우에는 다음 step만을 고려하기는 하지만 target policy가 greedy하게 학습되기 때문에 cliff에 가깝게 이동하는
방향(*최선*)으로 학습한다. 하지만 behavior policy는 $\epsilon$-greedy하게 행동하므로 절벽으로 가끔 떨어지게 되서 reward가 낮아지게 된다. 하지만 둘 다 어쩃든
수렴이 보장되어 있으므로 시간이 지나고 $\epsilon$이 작아지면 동일한 값으로 학습되게 되어 있다.

![cliff walking](https://whikwon.github.io/images/david_silver/cliff_walking.png)

***
## 정리
지금까지 배운 DP, TD에 관련된 내용을 Sampling 방식, Backup 방식에 따라서 정리해놓았다.

![DP vs TD](https://whikwon.github.io/images/DP_vs_TD.png)
![DP vs TD2](https://whikwon.github.io/images/DP_vs_TD2.png)


Reference: <br>
[David Silver RL Lecture5 slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf)
