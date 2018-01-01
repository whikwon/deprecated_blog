---
title: NMT by jointly learning to align and translate
date: 2017-09-05 00:00:00
layout: post
excerpt: "NMT에 Seq2seq + Attention mechanism을 활용한 모델에 대한 논문이다."
categories: [NLP, Paper]
comments: true
---
## Introduction

*Neural machine translation* 는 주로 *encoder-decoder* 구조로 이루어져 있으며 encoder는
translation하려는 source sentence를 fixed-length vector로 encode하고 decoder는 vector를
받아 translation을 진행해 output을 내놓는 방식이다. <br>
source sentence를 주어졌을 때 정확하게 translation을 할 수 있도록 확률을 최대화하게 학습이 진행된다. <br>
이러한 방식에서 문제점이 있는데 fixed-length vector에 모든 정보를 다 담아야 한다는 점이다.
문장이 길어질 수록 정보를 담기가 어려워질 것이고 실제로 동일 모델로 평가했을 시에도 input sentence
가 길어질수록 성능이 안 좋아지는 것을 확인했다.

이를 해결하기 위해서 **attention mechanism** 을 도입해서 모델이 target word를 generate할 때
source sentence의 어느 위치가 관련이 깊은지 확인하고 학습한다. 그래서 실제 predict할 때
previous generated target words에 더해서 학습한 source sentence의 위치와 관련된 context vector
도 고려해준다.

이렇게 했을 때 가장 중요한 점은 fixed-length vector로 전체 input sentence를 나타내지 않아도
되고 source sentence의 정보를 다 활용할 수 있게 되었다는 점이다.

그래서 결과적으로 더 긴 sentence를 다루었을 때 기존 대비 좋은 성능을 내었다.

## Learning to align and translate

논문에서 사용한 모델의 기본적인 구조를 설명한다. encoder는 **bidirectional RNN** 으로 이루어져있고
decoder는 **attention mechanism이 추가된 RNN** 을 사용하였다. 기본적인 구조는 아래와 같다.
![structure](https://whikwon.github.io/images/NMT_structure.png)

  - **Decoder** <br>
  Decoder의 target word를 생성하는 조건부 확률은 다음 식과 같이 나타낼 수 있다.
  $$p(y_i \vert y_1, ..., y_{i-1}, x) = g(y_{i-1}, s_i, c_i)$$
  $$s_i$$는 $$t = i$$에서의 hidden state이고 $$s_i = f(s_{i-1}, y_{i-1}, c_i)$$로 구해진다.
  기존 *encoder-decoder* 구조에서 한 가지 항이 추가되었는데 바로 **$$c_i$$(context vector)** 이다.
  $$c_i$$는 encoder의 *annotation* 들의 가중합으로 결정되며 결국 전체 input sequence 내에서
  어디에 초점을 맞춰야 할 지 알려주는 값으로 볼 수 있다. 수식으론 다음과 같이 나타낸다.
  $$c_i = \Sigma_{j=1}^{T_x} \alpha_{ij} h_j$$ <br> 그리고 가중치 $$\alpha_{ij}$$는 아래 식으로
  계산된다. $$\alpha_{ij} = \frac {exp(e_{ij})} {\Sigma_{k=1}^{T_x} exp(e_{ik})}$$,
  $$e_{ij} = a(s_{i-1}, h_j)$$
  여기서 $$e_{ij}$$는 *alignment model* 로 input의 $$j$$ 위치와 output의 $$i$$ 위치가
  얼마나 매칭되는지 나타내주는 값이며 $$a$$를 feedforward neural network로 놓고 모델 내 다른
  구성요소들과 같이 학습시킨다. <br>
  또한, alignment는 latent variable이 아니고 soft alignment로 놓아서 backpropagation
  을 통해서 학습이 가능하다. (*hard도 나오는데 차이 비교 필요함.*) <br>
  결론적으로 위 내용은 $$a$$를 학습시키면서 source sentence 위치의 가중치에 해당하는 $$\alpha_{ij}$$를 알 수 있고
  전체 정보 중 (*기존의 fixed length vector와 대비 된다.*) 특정 위치에 초점을 맞추어 선택적으로
  전달해준다.

  - **Encoder** <br>
  bidirectional RNN을 사용한다. BiRNN은 forward, backward RNN의 두가지로 구성되어 있으며
  forward RNN인 $$\overrightarrow f$$는 input sequence 순서대로 읽고 ($$x_1$$ to $$x_{T_x}$$)
  forward hidden state($$\overrightarrow {h_1},...,\overrightarrow {h_{T_x}}$$)를 게산한다.
  backward RNN $$\overleftarrow f$$은 반대로 거꾸로 읽고 ($$x_{T_x}$$ to $$x_1$$) backward
  hidden state($$\overleftarrow {h_1},...,\overleftarrow {h_{T_x}}$$)를 계산한다. <br>
  최종적으로 각각의 word $$x_j$$에 대해서 forward, backward hidden state를 합친
  $$h_j = {\big {[} \overrightarrow {h_j}^T; \overleftarrow {h_j}^T \big {]}}^T$$
  의 annotation을 얻을 수 있다. 이 annotation은 decoder에서 context vector를 구할 때 사용된다.

## Experiment setting

논문에서는 English-to-French translation에 대한 평가로 성능을 확인한다.
조경현 교수님의 14년도 [논문](https://arxiv.org/pdf/1406.1078)의 RNN Encoder-Decoder의 구조를  
비교 대상으로 삼아 진행하였다. 학습 방법이나 데이터셋 모두 동일하게 진행했다.
기존 모델인 RNNencdec와 attention이 적용된 RNNsearch 각각에 대해 ~30, ~50 길이(word)의 sentence로
학습을 시켰으며 최종적으로 학습이 끝난 후 *beam search* 를 사용해서 가장 조건부 확률이 높은 값을 결과로 얻었다.<br>

## Results

  - **Quantitative Results** <br>
  아래 결과를 보면 BLEU score를 보면 전반적으로 RNNsearch가 기존 모델인 RNNenc를 성능 면에서 압도하는 것을 볼 수 있다.
  또한, RNNsearch는 기존 phrased-based translation system과도 비슷한 성능을 내었다.<br>
  ![table1](https://whikwon.github.io/images/NMT_table1.png) <br>
  아래 그래프에서 RNNencdec를 봤을 때 sentence 길이가 증가할 경우 성능 저하가 심하게 일어나는 것을 볼 수 있다.
  하지만 RNNsearch를 봤을 때는 sentence의 길이에 Robust하여 attention이 긴 sentence를 다룰 때
  유리하다는 점을 확인했고 여기에 더해서 성능 저하 전에도 RNNencdec 대비 성능이 더 좋아져 학습에도 영향을
  미치는 것을 알 수 있다. (*정보가 전달이 잘 되니 성능도 더 좋곘지?*)
  ![BLEU_score](https://whikwon.github.io/images/NMT_BLEU_score.png)<br>

  - **Qualitative Analysis** <br>
  학습 시에 얻은 $$\alpha_{ij}$$를 통해서 source sentence를 넣어줬을 때 각각의 word가 generate된 word들에
  얼마나 영향을 미쳤는지 확인할 수 있다. <br>
  ![alignment](https://whikwon.github.io/images/NMT_alignment.png)<br>
  (a)를 살펴보면 영어와 프랑스어 사이의 다른 어순을 고려해서 translation이 잘 되었음을 알 수 있다.
  그리고 (d)를 통해서 soft alignment의 장점을 알 수 있다고 한다. (*이해가 잘 안됨..*)


## Conclusion

NMT 접근 방식 중 하나인 *encoder-decoder* 기존 방식의 문제점으로 input sentence를 fixed-length vector
로 다룬다는 점이 지적이 되었고 논문에서 이를 attention mechanism을 통해 해결하는 방식을 취한다. <br>
그럼으로써 long sentence에 대해서도 좀 더 잘 다룰 수 있게 되었다. 더 중요한 점은 traditional phrased-based
statistical machine translation의 성능에 비슷하게 되었다는 점인데 NMT가 생긴지 얼마되지 않아
성능을 따라잡았다는 것이다. (*실제로 이 때를 시작으로 기존의 모델이 거의다 NMT로 변경되기 시작한다.*) <br>
NMT 관련해서 계속 다뤄야 할 문제로 *unknown*, *rare words* 를 다루는 법을 언급하는데 다른 논문들을 통해서
보도록 하겠다.


Reference: <br>
Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473) 2014.
