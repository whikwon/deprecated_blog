---
title:  "A Diversity-Promoting Objective Function for Neural Conversation Models"
date: 2017-10-25 00:00:00
comments: true
---

- response(응답)을 decode할 때 commonplace(일반적인)한 응답이 나오는 문제를 objective function을 수정함으로써 개선한다.

## 모델 소개
Seq2Seq와 같은 Dialog system에서 일반적으로 학습에 사용하는 Maximum likelihood는 특성상 *"I don't know"* 와 같은 일반적인 응답을 많이 생성하는 경향이 있다.
이 문제를 해결하기 위해서 새로운 objective function(목적 함수)인 Maximum Mutual Information(MMI)를 제안하고 좀 더 다양한 응답을 얻는데 목적이 있다.

Maximum likelihood가 일반적인, 안전한 응답을 생성하는 이유는 학습 데이터 내 큰 확률을 차지하기 때문인데 그럴 경우 다양한 답변이 나오지 않아
좋은 dialog system을 얻을 수 없다.

그래서 메세지와 응답을 모두를 고려할 수 있는 목적 함수인 MMI를 소개하고 사용했을 때 좀 더 다양한 응답이 나오는 것을 보이고자 한다.

## 모델 구조
기본적인 Seq2seq 모델을 사용하고 MMI에 대해서만 덧붙여서 설명을 하도록 하겠다.
기존에 다른 모델들에서 사용하던 목적 함수 대한 maximum log likelihood는 $$\hat T =  \underset T {{arg}{max}} \{log\ p(T|S) \}$$ 로 나타낼 수 있다.
$$S(message)$$가 주어졌을 때 $$T(response)$$의 가장 높은 확률의 값을 택하는 것이다. 그러다보니 기존 데이터로 학습하면 많은 메세지에 대한 확률이 높은
응답들이 나오게 모델이 학습되고 이게 decode 때에도 반영되어, *I don't know, I'm OK* 와 같은 일반적이고 의미없는 응답이 생성되게 된다.  

위와 같은 문제를 완화하기 위해서 MMI를 목적 함수로 제시한다. MMI의 maximum log likelihood는 아래와 같다.
아래 식은 $S$$, $$T$$ 대한 mutual information 모두를 고려해서 모델을 학습한다는 내용을 나타낸다.
$$\begin{align}
\hat T &= \underset T {{arg}{max}} \{log \frac {p(S, T)} {p(S)p(T)}\} \\
&= \underset T {{arg}{max}} \{log\ p(T|S) - log\ p(T)\}
\end{align}$$

식에서 $$log\ p(T|S)$$는 위에서 설명한 ML에서의 항과 동일하고, 중요한게 뒤에 추가된 $$log\ p(T)$$이다. 이 항은 $$T(response)$$ 자체적인 확률 분포를
나타내는데 자주 나타나는 응답일 경우 확률이 높을 것으로 생각할 수 있다. 앞서, 일반적인 응답이 높게 나오는 이유가 학습을 했을 때, 자주 발생하는 응답들이 maximum likelihood에 의해
선택되어 그런 것을 보았을 때 $$log\ p(T)$$를 $$log\ p(T|S)$$에서 적절하게 빼주면 일반적인 응답을 줄일 수 있을 것이라고 생각해볼 수 있다.

그래서 위 식에 $$\lambda$$ 항을 $$\log\ p(T)$$ 붙여서 조절을 해주고 식을 정리하면 아래와 같다. 정리된 식을 보면
$$S$$와 $$T$$ 상호 간에 영향을 미치고 그 결과로 모델이 학습이 될 것이다.
$$\begin{align}
\hat T &= \underset T {{arg}{max}} \{(1 - \lambda)log\ p(T|S) + \lambda log\ p(S|T) - \lambda log\ p(S)\} \\
&= \underset T {{arg}{max}} \{(1 - \lambda)log\ p(T|S) + \lambda log\ p(S|T)\} \\
\end{align}$$

위에서 소개한 MMI 방법이 음성 인식에서 사용하고 있는데, seq2seq 모델을 학습할 때에는 *empirically nontrivial* 하여 학습 시에는 maximum likelihood로,
테스트(decode) 시에만 MMI를 사용해준다.

## 상세 설명

practical하게 사용할 때 위에서 본 두 가지 식을 모두 사용할 수 있으며 각각의 특징이 달라 살펴보도록 하자. <br>
1) MMI-antiLM: $$log\ p(T|S) - \lambda log\ p(T)$$, 두번째 항이 빈도 높고 일반적인 응답에만 penalty를 주는게 아니여서
일반적인 문장이 문법적으로 맞지 않게 변형할 수 있다. $$\lambda$$의 값에 의해서 결정된다고 한다. 이 문제를 해결하기 위해서 $$p(T)$$ 대신 $$U(T)$$를 사용한다.
$$\begin{align}
U(T) &= \displaystyle \prod_{i=1}^{N_t} p(t_k | t_1, t_2, ..., t_{k-1}) \cdot g(k) \\
p(T) &= \displaystyle \prod_{k=1}^{N_t} p(t_k|t_1, t_2, ..., t_{k-1})
\end{align}$$
위 식에서 보듯이 생성되는 단어 길이에 따라 가중치를 두는 형식인데 처음 생성되는 단어에는 penalty를 주고 어느 정도 진행되면 penalty를 없애는 방식이다.
첫 단어가 이 후에 생성될 단어를 크게 결정하기 때문에 중요하기 때문이고, 이 후에 penalty를 없애 모델이 결정하게 둠으로써 문법적인 에러를 막는다는 생각이다.
(*실제 응답의 뒷 부분에서 문법적인 에러가 주로 발생한다고 한다.*)

$$g(k)$$는 아래와 같이 threshold($$\gamma$$)를 기준으로 나타내고 <br>
$$\begin{equation}
  g(k)=
    \begin{cases}
      1, & \text{if}\ k \leq \gamma \\
      0, & \text{if}\ k > \gamma
  \end{cases}
\end{equation}$$
최종적으로 MMI-antiLM의 목적 함수는 $$log\ p(T|S) - \lambda log\ U(T)$$로 나타낼 수 있다.

2) MMI-bidi: $$(1 - \lambda)log\ p(T|S) + \lambda log\ p(S|T)$$, decoding을 intractable하게 만들 수 있다.



## 이전 모델과의 비교

## 데이터셋


## 학습 조건

## 평가 방법


## 성능
Reference: <br>
Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, Bill Dolan. [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055). 2015.
