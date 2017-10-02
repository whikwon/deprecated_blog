---
title:  "Dynamic Memory Network for Visual and textual Question Answering"
date: 2017-09-30 00:00:00
comments: true
---

- Stanford 16강에 해당하는 내용인 QA(Question Answering) 관련 최신 논문들을 순서대로 정리하도록 하겠다.
- 앞서 살펴본 DMN 모델의 몇몇 부분을 개선하고 앞서 text에 대한 QA를 이미지에도 적용시켰을 때 작동하는 것을 보여준다.

## Introduction
GRU를 이용한 DMN을 앞선 논문에서 살펴보았고 기존에 input module을 대신해서 이미지를 받을 수 있는
새로운 module을 고안하고 나머지 module들을 일부 수정한 뒤 DMN 기존 방식처럼 연결해서 하나의 모델로 구성한다.
새롭게 고안한 이 구조를 **DMN+** 라고 칭하고 있다. 아래 그림에 잘 나타나 있는데 input module이 변한 것을 알 수 있고
나머지 module들은 같은 방식으로 활용되고 있다.
![architecture](https://whikwon.github.io/images/NLP_DMN+_architecture.png) <br>
<center> <i> &lt;DMN+ 구조&gt;</i> </center> <br>

## Improved Dynamic Memory Networks: DMN+
기존 DMN에서 DMN+로 넘어오면서 크게 두가지가 변경되었다. 첫째는 input module의 변화(*Text, Vision으로 나뉘어진다.*)이고
둘째는 attention mechanism, 셋째는 memory update 방식이다. 순서대로 살펴보도록 하자.

### 1) Input Module for Text QA
기존 Text QA의 경우를 먼저보자. (*뒤에서 Vision QA에 대해서도 따로 볼 예정이다.*)
input module의 문제점이 두 가지가 존재한다. 첫째는 GRU를 정보 전달하는 모델로 사용해서
단방향으로만 정보가 전달된다는 단점이 있다. 또한, supporting sentence가 먼 경우에 서로 상호작용이 어려워진다는
단점이 있다. 이런 single GRU에서 오는 단점을 해결하기 위해서 해당 부분을 2개의 구성 요소로 쪼갠다.

**Sentence Reader** 와 **Input Fusion Layer** 가 그것들이다. 전자의 경우 embedding(*단어들 -> 문장*)
의 역할만 수행하며 후자의 경우는 문장들 사이에 상호작용이 일어날 수 있도록 해준다. <br>
아래 그림을 보면서 좀 더 상세하게 알아보도록 하자.
![fusion layer](https://whikwon.github.io/images/NLP_DMN+_fusion_layer.png) <br>
<center> <i> &lt;DMN+ Text input module 구조&gt;</i> </center> <br>
***sentence reader*** 의 경우 positional encoder로 이루어져있고 각 문장의 단어들을 embedding 한 뒤에 encoding 해서 하나의 sentence vector로 만들어준다.
식과 표기법을 함께 살펴보자. sentence encoding을 $$f_i$$, word token을 $$[w_1^i, ...,w_{M_i}^i]$$로 나타내고
$$M_i$$는 문장의 길이이다. 우리가 구하려는 sentence encoding을 $$f_i = \sum_M^{j=1} l_j \circ w_j^i$$로 정의하고
내부의 값들은 각각 $$l_{jd} = (1 - j/M) - (d/D)(1-2j/M)$$, $$d$$는 embedding index, $$D$$는 embedding 차원을 의미한다.

다음은 ***input fusion layer*** 이다. 위에서 보다시피 특별한 건 없고 bi-directional GRU를 사용한다.
기존에 단방향으로 전달되던 정보를 양방향으로 전달 가능하게 해서 누락되는 부분 없이 모두 공유되도록 만들어준다.
식으로 나타내면 아래와 같고 각 항들은 방향에 따른 input fact의 hidden state 값을 나타내준다.
$$\begin{align}
\overrightarrow{f_i} &= GRU_{fwd}(f_i, \overrightarrow{f_{i-1}}) \\
\overleftarrow{f_i} &= GRU_{bwd}(f_i, \overleftarrow{f_{i+1}}) \\
\overleftrightarrow{f_i} &= \overleftarrow{f_i} + \overrightarrow{f_i}
\end{align}$$

### 2) Input Module for VQA
이미지에 대한 QA를 위해서 새롭게 도입한 input module이다. 기존에 computer vision 쪽에서 살펴봤던 ***captioning,
object segmentation*** 에서 이미지를 다룰 때의 내용과 유사하여 쉽게 이해할 수 있었다.
Visual input module은 각기 다른 역할을 하는 3가지 구성요소로 이루어진다.
***local region feature extraction, visual feature, input fusion layer*** 로 구성되며 앞서 설명한 Text QA에서의
구성요소와 거의 유사하다. 그림을 보면서 차례대로 살펴보자.
![Visual input module](https://whikwon.github.io/imagesNLP_DMN+_Visual_input.png) <br>
<center> <i> &lt;DMN+ Vision input module 구조&gt;</i> </center> <br>
***local region feature extraction*** 단계에서는 문자 그대로 이미지 내에 특정 위치에 해당하는 이미지 feature를 추출하는 작업을 수행한다.
해당 논문에선 CNN 구조의 VGG 모델을 지나서 최종적으로 $$d\ =\ 512 \times 14 \times 14$$의 차원을 갖게 되고 여기에서
width, height를 의미하는 $$14 \times 14$$의 한 grid가 위치 정보를 포함하게 되고 각각은 512 차원의 vector로 구성된다. <br>
***visual feature embedding*** 에서는 앞서 위치 정보를 포함하는 이미지 vector를 받아서 tanh를 activation function으로 갖는 linear layer를 통과시켜
textual feature space로 투영시킨다. <br>
***input fusion layer*** 에서는 위에서 embedding까지 끝난 정보를 bi-directional GRU에 흘려준다. Bi-GRU를 사용하는 이유는
앞서 설명한 내용과 동일하다.

### 3) The Episodic Memory Module
DMN+에서의 episodic memory module은 DMN에서 식, attention 부분만 조금 바뀌고 구조 자체는 거의 그대로이다.
먼저 변하지 않은 전체적인 구성을 본 뒤에 부분적으로 변한 부분을 보도록 하자.
표기법을 나타내고 아래에 식을 정의하도록 하자. ($$q$$는 question, $$m^{t-1}$$은 이전 memory, \overleftrightarrow{f_i}는 $$i^{th}$$ fact,
$$\circ$$는 element-wise 곱, $$\vert \cdot \vert$$는 element-wise absolute value, $$;$$는 벡터 concatenation 를 나타낸다.) <br>
식의 목적은 DMN과 동일하다. 최종적으로 **episode** 를 구하는게 목적이고 이를 위해서 **attention gate** $$g_i^t$$를 구해야한다.
구해야 할 $$g_i^t$$는 $$\overleftrightarrow{f_i}, q, m^{t-1}$$에 의해 결정된다고 한 뒤 식을 정의한다.
![Episodic memory module](https://whikwon.github.io/images/NLP_DMN+_episodic_module.png) <br>
<center> <i> &lt;DMN+ Episodic memory module 구조&gt;</i> </center> <br>
위 내용을 식으로 정의하면 아래와 같다. <br>
$$\begin{align}
z_i^t &= \big[ \overleftrightarrow{f_i} \circ q; \overleftrightarrow{f_i} \circ m^{t-1}; \vert \overleftrightarrow{f_i} - q \vert; \vert \overleftrightarrow{f_i} - m^{t-1} \vert \big] \\
Z_i^t &= W^{(2)} tanh \big(W^{(1)} z_i^t + b^{(1)} \big) + b^{(2)} \\
g_i^t &= \frac {exp(Z_i^t)} {\sum_{k=1}^{M_i} exp(Z_k^t)}
\end{align}$$ <br>
DMN과 비교해봤을 때 변경된 부분은 $$z_i^t$$를 구하는 식에서 앞 부분 $$f_i, m^{t-1}, q$$에 대한 내용이 빠졌는데
이는 성능에 큰 영향을 주지 않으면서 연산량만 잡아먹어서 빼버렸다고 한다.

이제 성능을 증가시키기 위해 변경한 attention 부분을 보도록 하자. <br>
기존에 **soft attention** 을 사용했는데 이 방법은 연산이 간단하다는 장점이 있으나 summation process로 인해
위치, 순서 정보를 잃어버린다고 한다.
그래서 **attention based GRU** 를 도입하게 되는데 앞서 soft attention이 잃어버리는 $$\overleftrightarrow{F}$$의 위치, 순서 정보를 유지시켜주는 것이
목적이다. 결론적으로는 GRU내 식에 attention gate $$g_i^t$$ 값을 넣어주는 것이며 기존의 식과 비교하면 아래처럼 수정된다.
$$\begin{align}
h_i &= g_i^t GRU(c_i, h_{i-1}) + (1-g_i^t)h_{i-1} \\
&= u_i \circ \tilde{h_i} + (1 - u_i) \circ h_{i-1} \\
\Rightarrow h_i &= g_i^t \circ \tilde {h_i} + (1 - g_i^t) \circ h_{i-1}
\end{align}$$ <br>
$$g_i^t$$가 scalar라서 vector인 $$u_i$$보다 시각화가 쉬워서 그림에 대해서 위치 별로 attention 정도를 나타낼 수 있다.
그리고 memory update를 위한 $$c^t$$는 기존과 마찬가지로 마지막 hidden state 값을 넘겨준다.

### 4) Episodic Memory Updates
마지막으로 살펴볼 것은 episodic memory update 방식이다. 기존에 GRU를 통해서 memory update를 해줬는데
이 논문에선 Sukhbaatar et al. 의 예시를 들면서 episodic pass 내에서 전부다 같은 것보다 update의 weight가
다른게 성능 면에서 유리하다고 얘기하고 있다. 그래서 update 시에 ***GRU 대신 ReLU*** 를 사용한다. (0.5% 성능 향상) <br>
$$\begin{align}
m^t &= GRU(c^t, m^{t-1}) \\
\Rightarrow m^t &= ReLU(W^t[m^{t-1}; c^t; q] + b)
\end{align}$$

$$\begin{align}
h_i &= g_i^t GRU(c_i, h_{i-1}) + (1-g_i^t)h_{i-1} \\
&= u_i \circ \tilde{h_i} + (1 - u_i) \circ h_{i-1} \\
\Rightarrow h_i &= g_i^t \circ \tilde {h_i} + (1 - g_i^t) \circ h_{i-1}
\end{align}$$

## Experiments
### 1) Text QA Results
DMN(이전 모델), DMN2(input layer -> input fusion layer), DMN3(soft attention -> attention based GRU), DMN+(untied model)
을 살펴보면서 구성요소 각각이 변화했을 때 성능이 달라짐을 확인한다. 아래 결과 비교를 통해 DMN -> DMN2에서 fact들 간 상호작용이 개선되었고
DMN2 -> DMN3에서 위치, 순서 정보가 보존되었고 DMN3 -> DMN+ 에서 DMN+가 DMN3 대비 overfitting 경향을 보이면서 성능이 향상되었음을 알 수 있다.
![DMN to DMN+](https://whikwon.github.io/images/NLP_DMN_to_DMN+.png)
<center> <i> &lt;DMN, DMN2, DMN3, DMN+ 각 구성요소 변화 별 성능 비교&gt;</i> </center> <br>
이 외에 End-to-End Memory Network와 Neural Reasoner 모델과 비교한 결과도 논문엔 나타나있으나 성능이 좋은 편이라는 결과를 제외하고 별 다른
비교해석이 없어서 생략하도록 하겠다.

### 2) Comparison to SOTA using VQA
아래 표에서 보듯이 VQA 관련 많은 항목들에서 SOTA 성능을 보인다. 정성적으로 attention gate $$g_i^t$$를 그림에 나타내보면
질문에 대한 대답을 적절하게 인식하고 있는 것을 확인할 수 있다.
![DMN+ VQA result](https://whikwon.github.io/images/NLP_DMN+_VQA_result.png)
<center> <i> &lt;VQA 와 기타 모델 성능 비교&gt;</i> </center> <br>
![DMN+ VQA result2](https://whikwon.github.io/images/NLP_DMN+_VQA_result2.png)
<center> <i> &lt;VQA attention gate 확인 결과&gt;</i> </center> <br>


Reference: <br>
Caiming Xiong, Stephen Merity, Richard Socher. [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/pdf/1603.01417). 2016.
