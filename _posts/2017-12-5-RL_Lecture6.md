---
title:  "RL_Lecture6: Value Function Approximation"
date: 2017-12-15 00:00:00
comments: true
---

- David Silver의 Reinforcement Learning 6강을 정리한 내용이다.

이번 장에서는 Value function을 근사하는 방법에 대해서 배운다. 지금까지는 tabular case에 대해서 다루었고 이는 state, action에 대한 table을 만들어놓고 학습을 위한 iteration(Value iteration, Policy iteration)이 진행 됨에 따라 table 내 값이 변하는 방식이다. 하지만 tabular한 방식은 MDP가 커질 수록 한계에 부딪히는데 state, action table을 메모리에 계속 할당해야 한다는 점 때문에 그렇다.
그래서 이런 메모리에 대한 문제를 해결하기 위해서 value function을 근사해서 식으로 나타내면 어떨까? 라는 접근을 할 수 있다. 이 것이 이번 장의 주요 내용이다.

***
## Value Function Approximation
Backgammon($$10^{20}$$), Go($$10^{170}$$), Helicopter(continuous)와 같은 복잡한 문제를 풀기 위해서 기존에 배웠던 model-free method을 어떻게 *scale up* 할 수 있을까? &#8594; value function을 근사하자!

아래처럼 특정 parameter에 대한 state-value 혹은 action-value function을
근사한 식을 완성하면 되겠다. action-value의 경우에는 각 $$s, a$$에 대한 식이나 $$s$$에 대한 식으로 근사할 수 있겠다. <br>
$$\hat v(s, \mathbf{w}) \approx v_{\pi}(s) \text{ or } \hat q(s, a, \mathbf{w}) \approx q_{\pi}(s, a)$$

![value function approximation](https://whikwon.github.io/images/david_silver/value_approximation.png)

이렇게 근사하는 경우 장점은 첫째로 state에 관한 table을 사용하지 않기 때문에 메모리 문제를 해결해준다. 둘째로는 아주 정확한 true value function 값을 구하는 것이 아니라 수식으로 근사하기 때문에 일반화되는 특징을 갖게 된다. 이 경우 직접 관찰하지 않은 데이터에 대해서도 예측을 가능하게 해준다.

### Approximator 종류?
Approximator는 머신러닝 모델들로 아주 다양하다. 이 중에서 강화 학습에는 주로 ***differentiable*** 한 ***Linear combinations of features, Neural Network*** 를 사용한다. 그리고 학습할 때 각 trajectory 내 데이터들은 서로 correlated 되어 있으므로 non-stationary, non-iid 데이터이므로 supervised의 방식을 그대로 가져오면 안 되고 몇 가지 technique를 추가해주어야 한다.

***
## Incremental Methods(Policy Prediction)
### 1) Gradient Descent
딥러닝에서 가장 많이 사용하는 최적화 방법으로 강화 학습에서도 사용한다. 학습을 위한 목적 함수은 Mean-Squared Error(MSE)로 나타내고 구하려는 parameter인 $$\mathbf{w}$$에 대한 gradient를 update해주면 된다.

![SGD](https://whikwon.github.io/images/david_silver/SGD.png)

### 2) Linear Value Function Approximation
그럼 linear function으로 value function을 근사하는 예를 들어보자. $$\mathbf{x}(S)$$를 state를 나타내는 *feature vector* 라고 하면 $$\hat v(S, \mathbf{w}) = \mathbf{x}(S)^T \mathbf{w} = \displaystyle \sum_{j=1}^n \mathbf{x}_j(S)\mathbf{w}_j$$로 나타낼 수 있다.

근사할 식의 형태를 정했으니 목적 함수를 나타낼 수 있고 update parameter에 대한 gradient를 아래와 같이 구할 수 있다.

![linear value approximation](https://whikwon.github.io/images/david_silver/linear_value_approx.png)

value function을 linear function에 근사했을 때 우리가 기존에 사용하던 *table lookup* 을 근사한 linear function의 특별한 형태라고 생각할 수 있다.

![table lookup features](https://whikwon.github.io/images/david_silver/table_lookup_features.png)

### 3) Supervisor? target value?
기존에 supervised learning의 경우 supervisor가 존재해서 label을 제시해서 학습이 진행되었다. 강화 학습의 경우에는 supervisor가 존재하지 않지만 대신 reward
를 활용하는 방법을 생각해 볼 수 있다. 이전 장의 MC, TD 등에서의 ***target value*** 는 value function가 어떤 방향으로 학습되어야 하는지(**TD-Error**)를 제시해주었다. 이 값을 우리가 근사하려고 하는 value function의 target으로 놓고 학습할 수 있겠다.

MC는 $$G_t$$, TD는 $$R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w})$$, TD($$\lambda$$)는 $$G_t^{\lambda}$$가 target이 되겠다.

![target values](https://whikwon.github.io/images/david_silver/supervisor_target.png)

### 4) Monte-Carlo with Value Function Approximation
state($$S_t$$), return($$G_t$$) 쌍에 대해서 supervised learning을 하면 value function을 근사할 수 있다. MC는 function의 종류와 관계없이 항상 local optimum으로 수렴한다. MC의 return은 *unbiased* 하지만 noisy한 특징이 있다.

![MC value approximation](https://whikwon.github.io/images/david_silver/MC_value_approx.png)

### 5) TD with Value Function Approximation
state($$S_t$$), TD-target($$R_{t+1} + \gamma \hat v(S_{t+1} \mathbf{w})$$) 쌍에 대해서 supervised learning을 하면 value function을 근사할 수 있다. linear TD(0)의 경우에는 global optimum에 가깝게 수렴하나 neural net과 같은 non-linear의 경우에는 수렴이 보장되지 않는다.
TD-target은 *biased* 하지만 MC에 비해 variance가 작은 특징이 있다.

![TD value approximation](https://whikwon.github.io/images/david_silver/TD_value_approx.png)

### 6) TD($$\lambda$$) with Value Function Approximation
TD($$\lambda$$)의 forward, backward view를 살펴보자. forward의 경우에는 전체를 다 본 뒤에 step 별로 가중치를 줘서 합친 값인 $$G_t^{\lambda}$$을 target으로
supervised learning으로 value function을 학습시킨다.

backward의 경우에는 현재의 상태에 영향을 미친 과거 상태의 빈도($$\textbf{x}(S_t)$$)와 얼마나 최신 일인지에 대해 고려($$\gamma \lambda E_{t-1}$$)한 값을 target으로
supervised learning으로 value function을 학습시킨다.

![TD lambda value approximation](https://whikwon.github.io/images/david_silver/TD_lamb_value_approx.png)

***
## Policy control with Value Function Approximation
위에서 지금까지 살펴본 state-value function을 근사하는 일을 action-value function에 똑같이 적용시키면 Policy control 문제를 해결할 수 있다.
evaluation 단계에서 action-value function을 구한 뒤 $$\epsilon$$ policy improvement를 하면 된다.

![policy control](https://whikwon.github.io/images/david_silver/policy_control.png)

state-value에서 state에 관한 식이었지만 action에 대한 항도 추가해서 feature vector로 나타내주면 되고 $$\textbf{w}$$에 대해 학습하면 된다.
학습 방법도 MC, TD, TD($$\lambda$$) 모두 동일한 내용이므로 넘어가도록 하자.

![q approximation](https://whikwon.github.io/images/david_silver/q_function_approx.png)
![q approximation](https://whikwon.github.io/images/david_silver/q_function_approx2.png)
![q approximation](https://whikwon.github.io/images/david_silver/q_function_approx3.png)

***
## Bootstrap helps? Prediction Convergence?
여러 가지 모델로 학습시켰을 때 실제로 bootstrap이 도움이 되는지에 대한 비교 결과이다. $$\lambda$$가 1일 때 MC(no bootstrap)인데
실제 1에서 성능이 대체로 안 좋은 것을 확인할 수 있다. bootstrap은 대부분 도움이 되며 모델이나 환경에 따라 적절한 값을 찾는게 중요하다고 한다.

![bootstrap](https://whikwon.github.io/images/david_silver/bootstrap_helps.png)

근데 이 때, bootstrap이 수렴 여부에 영향을 미치는 것을 생각해봐야 하는데 실제로 **Baird's Counterexample** 의 경우에는 수렴하지 않게 된다.
각 경우들에 따라서 Policy evaluation에서의 수렴 여부를 정리하면 아래 표와 같다.

![convergence](https://whikwon.github.io/images/david_silver/convergence.png)

***
## Gradient TD
위에서 TD의 경우 bootstrap으로 인해서 많은 경우에 수렴이 보장되지 않는 것을 보았다. 이러한 수렴에 관한 문제를 해결하기 위해서 TD에서 항을 약간 변경한 gradient TD를 사용할 수 있다.
자세한 내용은 설명하지 않고 넘어가므로 교재를 참고해서 정리하자.

![gradient TD](https://whikwon.github.io/images/david_silver/gradient_TD.png)

***
## Control Convergencedavid_silver/
prediction보다 control은 더 수렴에 대한 보장이 되지 않는다고 한다. 계속 value function이 개선되는 것이 아니라 좋아졌다 나빠졌다를 반복하는(chatter) 경향을 보인다고 한다.

![control convergence](https://whikwon.github.io/images/david_silver/control_convergence.png)

***
## Batch Methods
지금까지 gradient descent 방법을 통한 value function을 근사하는 예를 살펴보았다. 간단하게 experience를 sampling해서 gradient descent에 비례해서 update 시켜주는 방식인데 이렇게
진행할 경우에 experience를 1회성으로 쓰고 다시 불러오기 때문에 비효율적이다. 그래서 조금 더 효율적으로 하기 위해서 experience를 왕창 뽑아서 batch(training data)를 만드는 것이 필요하고
마찬가지로 gradient descent 방법으로 반복적으로 학습할 수 있게 된다.

### 1) Least Square prediction
데이터만 $$\mathcal{D}$$로 많아졌지 위에서 본 MSE와 동일한 방법(Least squares)으로 value function을 학습시킨다. 이 때, 아래 식에서도 볼 수 있듯이 매 학습마다 **batch 전체** 를 사용한다.

![least square prediction](https://whikwon.github.io/images/david_silver/least_square_prediction.png)

### 2) Experience Replay
agent로부터 sampling한 batch 데이터 전체를 사용하는 것이 아니라 **batch에서 sampling 한 뒤** stochastic gradient descent로 학습하는 것이 **Experience replay** 이다.
딥러닝의 supervised learning에서 사용하는 stochastic gradient descent와 동일한 방법이며 이렇게 할 경우에 supervised learning에서 봤을 때처럼 학습 속도가 빠르다는 장점이 있을 수 있다.
또, 강화 학습의 경우에는 agent로부터 sampling한 experience가 서로 correlated 되어 있기 때문에 이를 decorrelated 하게 만들어주는 효과도 있다.  

![experience replay](https://whikwon.github.io/images/david_silver/experience_replay.png)

### 3) Experience Replay in DQN
DQN에 experience replay가 fixed Q-targets와 함께 사용된다. 방법은 agent로부터 experience를 sampling해서 batch를 만들고 *여기에서 일부 sampling*(**experience replay**)해서
 Q-learning target과 Q-network의 값을 구한 뒤 stochastic gradient descent로 학습하는 방법이다. Q-network 값은 update된 parameter를 통해서 구하면 되고 Q-learning target은
 *이전 parameter 중에서 fixed된 값으로 구한다.* (**fixed Q-targets**)

두 가지 방법을 통해 TD나 Sarsa를 neural network로 근사할 때의 불안정한 문제를 해결한다. 먼저, experience replay는 experience간의 correlation을 sampling을 통해서 해결하고 학습을 안정화한다. 그리고 fixed Q-targets은 parameter가 계속 학습되는 경우에 현재의 Q-network와 바로 이전의 parameter의 Q-network가 correlated 되어 학습시 불안정해지는데, 이 [문제](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df)를 이전 특정 지점의 parameter를
고정시켜놓고 이를 사용해서 Q-learning target을 구하는 network로 사용해서 해결한다. 일정한 step이 지날 때마다 Q-learning target network의 parameter를 update 시켜준다.

![DQN](https://whikwon.github.io/images/david_silver/DQN.png)

아래 표는 experience replay와 fixed Q-targets이 얼마나 효과적인지 보여주는 예시이다. 값들은 atari 게임을 학습시켰을 때의 점수이다.

![DQN stabilize](https://whikwon.github.io/images/david_silver/DQN_stablize.png)

### 4) DQN in Atari
아래는 DQN을 사용해서 비디오 게임인 Atari를 학습시킬 때 사용한 모델이다. CNN과 FC layer를 활용해서 화면의 이미지를 읽어서 state를 나타내고
action을 할 수 있도록 만들었다. 모든 조건들을 동등하게 학습시켰을 때 많은 게임을 사람보다 잘할 수 있게 되었다는 결론이다.

![DQN atari](https://whikwon.github.io/images/david_silver/DQN_atari.png)
![DQN_result](https://whikwon.github.io/images/david_silver/DQN_result.png)

### 5) Linear Least Squares Prediction
gradient descent 방법은 많은 iteration을 반복해야 원하는 optimal parameter를 찾을 수 있다. 그래서 regression에서의 normal equation과 같이
직접 해를 찾는 방법을 고려해본다. 하지만 이 때 구해지는 running-time은 $$O(N^3)$$으로 incremental solution의 $$O(N^2)$$보다 커서 복잡한 문제에 대해서
사용하기 어려우며 장점은 몇몇 경우에 대해서 incremental solution보다 수렴이 보장된다는 점이다.

MC, TD를 아래와 같이 식을 풀어서 바로 해를 구할 수 있다.

![linear least square](https://whikwon.github.io/images/david_silver/linear_LS.png)

incremental과 linear least square의 수렴 보장 여부를 나타낸 표이다.

![linear least square convergence](https://whikwon.github.io/images/david_silver/linear_LS_convergence.png)

Reference: <br>
[David Silver RL Lecture6 slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf)
