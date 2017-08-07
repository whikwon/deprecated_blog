---
title:  "Distributed Representations of Words and phrases and their Compositionality"
date: 2017-08-03 00:00:00
comments: true
---

- CS224n Lecture2 내용에 포함된 논문으로 Skip-gram에 대한 내용을 주로 다루고 있다.

#### 1. Introduction <br>
Distributed representations of words는 아주 예전부터 연구되던 주제이다. 과거에 연구가 진행되어 오다가
NLP에서 Skip-gram 모델이 발표가 되면서 낮은 연산량으로 training이 가능해졌다.
그리고 이렇게 word를 vector로 표현하게 되면서 언어학적인 규칙이나 패턴을 나타낼 수 있게 되었다.
이 논문에서는 기존 Skip-gram 모델에 몇 가지 연산, 성능 향상을 위한 기법을 소개하고 있다.
*Hierarchical softmax, Negative sampling, Subsampling of frequent word, Phase skip-gram* 이 그것들이다.
순서대로 상세하게 소개하도록 하겠다.

#### 2. The Skip-gram Model <br>
Skip-gram model을 training하는 방법은 특정 word를 중심으로 주변에 가장 있을만한 word를 찾는 것이다.
maximize할 objective function은 아래와 같다. <br>
$${1 \above 1pt T} \sum_{t=1}^T \sum_{-c\leqq j \leqq c, j\neq 0} log\ p(w_{t+j}|w_t)$$ <br>
$$(c : window\ size,\ w_t : center\ word)$$ <br>
그리고 구하고자 하는 center word일 때 output word일 확률은 아래와 같이 정의한다.
$$p(w_O|w_I) = {exp({v'_{w_O}}^T v_{w_I}) \above 1pt \sum_{w=1}^W exp({v'_w}^T v_{w_I})}$$ <br>
$$(v_w,v'_w :\ "input"\ and\ "output"\ vector\ representations\ of\ w,\ W: number\ of\ words\ in\ vocabulary)$$

#### 2.1 Hiearchical Softmax <br>
$$\nabla log\ p(w_O|w_I)$$를 구할 때의 연산량이 $$W$$에 비례하는데 $$W$$는 보통 $$10^5-10^7$$ 정도로 큰 편이라
줄이고자 하여 도입한 개념이다. 줄인 뒤의 연산량은 $$log_2(W)$$에 비례하게 된다.
자세한 내용이 CS224n에서 다루지 않아서 일단 pass하도록 하겠다.

#### 2.2 Negative sampling <br>
위에서 소개한 hierarchical softmax의 대체재로 사용되는 방법이다.
Noise Contrastive Estimation(NCE)에서 출발한 개념으로 NCE는 좋은 모델은 *Should be able to differentiate
data from noise by means of logistic regression* 해야 한다고 가정하고 있다.
그렇게 나온 Negative Sampling의 objective function은 아래와 같다.
$$log\ \sigma({v'_{w_O}}^T v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i} \sim P_n(w)[log\ \sigma(-{v'_{w_i}}^Tv_{w_I})]$$ <br>
$$(k : \#\ of\ negative\ samples)$$ <br>
기존 Skip-gram 모델의 Objective function을 대체하는 이 식을 보자.
먼저 input data(center word)에 대해서 target word를 포함 추가 k개의 오답(negative sample)을 $$P_n(w)$$의 확률로 sampling한다.
maximize하는게 목적이므로 target word가 될 확률($$\sigma({v'_{w_O}}^T v_{w_I})$$)을 최대로 하면서
negative word들의 확률 값에 샘플링 확률을 곱한 값을 최소로 갖는 $$v',v$$를 찾게 된다.
결국 앞선 식이랑 비교했을 때 굳이 negative 값들을 전부 확인 할 필요 없이 몇 개만 확인하자는 게
주요 내용이다.
negative word의 샘플링 확률은 $$P(w_i) = {f(w_i)^{3/4} \above 1pt {\sum_{j=0}^n f(w_j)^{3/4}}}$$ 로,
전체 단어 중에 얼마나 있는지 고려해서 뽑히게 된다. 위 식에서 $$3/4$$승은 empirical한 값으로 성능이 가장 좋다고 한다.

#### 2.3 Subsampling of Frequent Words <br>
큰 corpora(말뭉치)에는 거의 정보를 가지고 있지 않은 단어들이 많이 반복적으로 포함된다. (예: "in", "the", "a")
그래서 이런 쓸모없는 많은 정보들을 거르기 위해 Subsampling을 진행한다.
Subsampling 후 preserve probability를 아래와 같이 정의한다. <br>
$$P(w_i) = 1 - \sqrt {t \above 1pt f({w_i})}$$<br>
$$(f(w_i):the\ frequency\ of\ word\ of\ w_i,\ t: chosen\ threshold \sim 10^{-5})$$
단순하게 빈도가 높은 단어는 조금 남기고 낮은 단어는 많이 남긴다고 이해하면 되겠다.

#### 4. Learning Phrases <br>
corpora내에 많은 phrases들이 각각의 word들의 합쳐진 뜻이라기 보다는 완전히 새로운 뜻을 갖는 경우가
많았다. (예 : "New York Times", "Toronto Maple Leaf") 그래서 이런 phrases에 차라리 unique한 token을 줘서
training 시켜보자는 시도에서 출발하였고 결론적으로 성능이 증가하였다.
물론 phrases을 찾아내는 방법에도 여러가지가 있겠지만 여기에서 다루지는 않겠다.

기타 내용들은 skip-gram 모델의 성능 비교, 기존에 traditional한 모델들에 비해 성능이 좋다는 내용이고
Conclusion에 Phrases with skip-gram의 성능 향상, 더 많은 데이터의 중요성, Subsampling의 성능 향상, Negative sampling의
효율성에 대해서 다시 한 번 짚고 마무리 한다.

Reference: <br>
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).2013.
