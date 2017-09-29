---
title:  "Dynamic Memory Network for Natural Language Processing"
date: 2017-09-29 00:00:00
comments: true
---

- Stanford 16강에 해당하는 내용인 QA(Question Answering) 관련 최신 논문들을 순서대로 정리하도록 하겠다.
- 먼저, 여러가지 상황에 대한 QA를 가능하게 만든 Dynamic Memory Network 모델을 보도록 하자.

### Introduction

QA는 되게 재밌는 주제라고 강의에서 소개하고 있다. 왜냐하면 지금까지 CS224n을 통해 살펴본 많은 모델들의 각각의
기능들이 가능해야 **일반적인 QA 기능** 을 수행한다고 말할 수 있기 때문이다. (*MT, Coreference resolution, NER, POS...*)
일반적인 QA기능을 가능하게 한 DMN을 논문에서 소개하며 내부 구조는 그렇게 복잡하지 않다. 다음 장에서 살펴보도록 하자.

### Dynamic Memory Networks

4가지 구성 요소 **Input Module, Question Module, Episodic Memory Module, Answer Module** 로 이루어진다.
아래 그림과 같이 구성되어 있으며 module 하나하나 살펴보도록 하자. <br>
![architecture](https://whikwon.github.io/images/NLP_DMN_architecture.png)<br>
<center> <i> &lt;Dynamic Memory Network 구조&gt;</i> </center> <br>

**1) Input Module** <br>
input은 질문에 답하기 위한 내용이 되는 본문쯤으로 보면 되겠다. 다른 모델들에서 했듯이 text를
distributed vector representation로 encode를 해줘야하고 여기에선 RNN 계열인 GRU를 사용한다.
본문에서 사용할 표기법은 문장들 $$T_I$$의 구성 단어를 $$w_1, ..., w_T$$라고 하고
GRU를 통과한 hidden state를 $$h_t = GRU(L[w_t], h_{t-1})$$라고 한다. 여기서 $$L$$은 embedding matrix이며
Glove로 구한 값을 사용한다.

**2) Question Module** <br>
input module과 마찬가지로 question의 text를 encode해주면 된다. 이 때도 GRU를 사용한다.
question 문장 $$T_Q$$의 단어들을 $$w_t^Q$$라 하고 GRU를 통과한 hidden state를 $$q_t = GRU(L[w_t^Q], q_{t-1})$$
라고 한다. $$L$$은 input module에서 사용된 embedding matrix이다. input module과 약간 다른 점이 있다면
input에서는 hidden state를 attention을 위해서 전부다 사용하지만 question에서는 마지막 hidden state값만
활용한다는 차이점이 있다.

**3) Episodic Memory Network** <br>
episodic memory module은 여기서 처음 등장하는 개념인데 여러 층으로 이루어지며 각 층에는 input값에 매칭되는 episode값이 존재한다.
episode $$e^i$$는 attention mechanism을 이용해서 생성되고 이 때, input값인 fact representation $$c$$와 question representation $$q$$와 이전 memory $$m^{i-1}를 참조한다.
이렇게 만든 각 층의 episode는 해당 층의 memory를 만들 때 쓰이며 GRU를 사용한다. 최종적으로 마지막 episodic memory 층의 마지막 memory는
answer module로 전달된다.

*방법은 위와 같고 직관적으로 보면 사실 중에 어느 값이 정답과 관련이 있을까? 라는 물음을 해결하기 위해서 질문과 사실을 보면서
가중치를 부여하고 기억에 남기고, 다시 질문과 사실, 기억을 고려해서 더 나은 가중치를 부여하고 기억에 남기는 과정을 반복하는 것으로 볼 수 있겠다.
이 때, 어떤 정답으로 수렴을 하면 명확하게 설명이 되겠지만 그런지는 모르겠다.*

층이 여러 개 있을 때 장점은 어떤 ***transitive inference*** 를 가능하게 해준다는 점이다.
질문이 여러 번의 논리적인 사고를(문장들을) 거쳐서 답을 얻어야 하는 경우가 해당된다. <br>
<예시> <br>
Input : *John put down the football*, *John is in the living room* <br>
Question : *Where is the football?* <br>
Answer : *living room* <br>

**Attention Mechanism** <br>
Episodic Memory Network에서 사용되는 attention mechanism를 살펴보자. 이 후에 발전되는 모델들에서 바뀌는 부분이기도 하다.  
논문에서 gating function이라고 하는 $$g_t^i$$를 구하고 이를 이용해서 우리의 목적인 episode를 생성해주면 된다.
앞에서 설명했듯이 input, question, 이전 memory가 필요하므로 $$g_t^i = G(c_t, m^{i-1}, q)$$로 나타낼 수 있다.
scoring function인 $$G$$는 feature set인 $$z(c, m, q)$$를 받아서 scalar score를 만든다고 볼 수 있다.
$$z(c, m, q)$$는 input, memory, question vector들 간의 유사성으로 나타낼 수 있고 식으로 표현했을 때
$$z(c, m, q) = \big [ c,m,q,c \circ q, c \circ m, \vert c-q \vert, \vert c-m \vert, c^T W^{(b)}q, c^TW^{(b)}m \big ]$$
로 나타낼 수 있다. 그리고 $$G$$는 two-layer feed forward neural network를 사용해서 최종적으로
$$G(c,m,q) = \sigma \bigg (W^{(2)} tanh \big ( W^{(1)} z(c,m,q) + b^{(1)} \big ) + b^{(2)} \bigg )$$ <br>
위에서 구한 식을 최종적으로 episode 구하는 GRU 식에 활용하면 아래와 같은 식을 얻을 수 있다.
\begin{align} h_t^i &= g_t^i GRU(c_t, h_{t-1}^i) + (1-g_t^i)h_{t-1}^i \\
e^i &= h_{T_c}^i \end{align}

**4) Answer Module** <br>
Answer module은 우리가 하려는 task에 따라서(*task에 따라서 개별적으로 학습시켜줘야 한다. 일반적인 모델이긴 하지만 각각을 따로 할 수 있다는 거지 한 번
학습시킨 모델이 모든 걸 할 수 있다는 뜻은 아니다.*) answer로 마지막 episodic memory값만 받거나 아니면 time step별로 받을 수도 있다.
Answer module도 다른 module들과 마찬가지로 GRU를 사용하며 마지막 memory의 값 $$m^{T_M}$$을 initial state로 받아서 아래와 같이
output $$y_t$$와 hidden state $$a_t$$를 생성한다. <br>
\begin{align} y_t &= softmax(W^{(a)}a_t) \\
a_t &= GRU([y_{t-1}, q], a_{t-1}) \end{align}

### Training
supervised로 다른 모델들과 유사하게 cross-entropy error 감소하도록 학습이 진행된다.
한 가지 특이한 점은 여기서 사용한 bAbI와 같은 gate supervision이 되는 dataset의 경우에는
각각의 gate마다 cross-entropy를 계산해서 합친 값을 cost로 두고 학습을 진행시킨다.
*gate supervision 이라는게 중간 단계의 값들에 대해서도 확인을 한다는 건가?*

### Experiments
각각의 task 목적에 맞게 학습을 시켜서 기존 SOTA 모델들과 비교했을 때 많은 분야에서 성능이 우세한 것으로 나타났다.
Question Answering, Sentiment Analysis, Sequence Tagging에서 SOTA 성능을 내었고 다른 모델들에 의해
또 갱신되므로 자세한 내용은 생략하도록 하겠다.

### Quantitative Analysis of Episodic Memory Module
DMN 구조의 핵심을 Episodic Memory Module로 보아도 무방하며 얼마나 중요한 역할을 하는지에 대한 정성적,
정량적으로 평가를 진행한다.
아래 표를 보면 memory의 수에 따라서 성능이 어떻게 달라지는 지 평가를 한 결과이다. 첫번째 task인
**three-facts** 를 보면 memory의 수에 따라서 성능이 크게 증가하며 0-1 pass인 경우는
아예 작동하지 않는다는 것을 볼 수 있다. 이는 앞서 설명한 것처럼 얼마나 전체 input(fact)를 보고
중요한 지점을 파악하는 데에 memory 수가 큰 역할을 한다는 것을 보여준다. 뒤에 **count** 와 **list/sets**
를 보더라도 memory 개수에 따른 성능의 증가가 두드러진다. 이렇게, 좀 복잡한 논리를 요구하는 과정의 경우에 memory가 큰 역할을
한다고 생각할 수 있겠다. **sentiment** 의 경우에는 2 pass에서 성능이 가장 좋고 이 후에는 감소하는 경향을
보이는데 왜 그런지 잘 모르겠다.
![quantative analysis](https://whikwon.github.io/images/NLP_DMN_episodic_memory1.png)<br>
<center> <i> &lt;Episodic Memory 정량적 평가 결과&gt;</i> </center> <br>

### Qualitative Analysis of Episodic Memory Module
이제 정성적인 분석이다. 위에서 본 결과 중에 Sentiment에 대해서 1 pass일 때와 2 pass일 때를 비교해서
모델이 pass의 개수에 따라서 어떤 식으로 작동을 하는지 확인할 것이다. 확인 대상은 input 문장의 각각 단어에 대한
attention 정보이다.
아래 그림에서 보듯이 1 pass와 2 pass 모델에 대해서 각각 비교했고 대체적으로 2 pass에서 더 특정 몇 개 단어에
집중하는 것으로 보인다. 이는 pass를 적게 통과하면 전체 내용에 대해서 뭐가 중요한 지 파악이 안 되므로
최대한 많은 정보를 전달하는 것이라고 생각해볼 수 있다. 

![qualitative analysis](https://whikwon.github.io/images/NLP_DMN_episodic_memory2.png)<br>
<center> <i> &lt;Episodic Memory 정성적 평가 결과&gt;</i> </center> <br>



Reference: <br>
