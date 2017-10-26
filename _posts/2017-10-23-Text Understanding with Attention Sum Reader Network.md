---
title:  "Text Understanding with the Attention Sum Reader Network"
date: 2017-10-23 00:00:00
comments: true
---

- Document(문서) 내용에 대한 Question(질문)의 Answer(답)를 찾는 모델이다.
- 기존 모델들에서 사용되는 구조들을 차용해서 단순한 방법으로 좋은 성능을 내었다.

## 모델 목적
**문서 내 단어 중** 어떤 단어가 질문의 답에 가까운 지 찾는다. 문서에 없는 단어를 유추하거나 새롭게 생성할 수는 없다.

## 모델 구조
Attention Sum Reader Network(AS Reader)는 아래 그림과 같은 구조를 가진다. <br>
***Embedding - encoder(Document, Question) - dot product - softmax - probability of the answer*** 의 순서로 진행된다. <br>
새롭게 변형되어 사용되는 구조는 없고 전부다 기존 모델들에서 사용되는 구조들을 가져다가 사용했다. <br>
1) Embedding : 문서, 질문의 단어들을 embedding 해준다. (*문서, 질문에 대해 같은 look-up table을 사용하나?*) <br>
2) Encoder : bi-directional GRU를 사용하였다. 문서와 질문 각각에 대해서 진행된다. <br>
3) Dot product : 2에서 발생하는 두 개의 output을 dot product 해준다. <br>
4) Softmax : 3에서 구한 문서 내 위치가 다른 단어 각각에 해당되는 값들을 대상으로 softmax를 취해 확률로 바꿔준다. <br>
5) Probability of the answer : 4에서 위치가 다르지만 중복된 단어들이 사용되었을 수 있고, 이 확률을 전부다 합쳐서 해당 단어가 답일 확률이라고 정의해준다. <br>

![structure](https://whikwon.github.io/images/NLP_attentionsum_structure.png) <br>
<center> <i> &lt;Attention Sum Reader Network 구조&gt;</i> </center> <br>

## 이전 모델과의 비교
1) Attentive and Impatient Reader : 기본적인 모델의 구조를 차용했으며 답의 확률을 구하는 방식이 다르다. 문서 내 전체 단어에 대해서 가중치 합을 구하는 방식(*soft attention*, $$r = \sum_i s_i f_i(\textbf{d})$$)을 취하는데, 이럴 경우에 January와 March가 둘 다 답이 될 수 있는 경우 February라는 답이 나오곤 한다. 대신 AS Reader에서는 같은 단어들에 대해서만 확률을 합치므로 명확하게 January 혹은 March로
답이 나오게 된다. <br>
2) Chen et al. 2016 : attention weight 계산 방식이 다르다. 단순한 dot product를 사용하는 AS Reader와 달리 bilinear term을 사용한다. ($$s_i \propto exp(f_i(\textbf{d})^T W g(\textbf{q}))$$)
위의 Attentive and Impatient Reader에서 파생된 모델로 attention을 구하는 방식은 그대로지만 성능은 더 좋게 나온다고 한다. <br>
3) Pointer Networks : 답을 구하는 방식에 대해 가장 영향을 많이 받은 모델이다. 전체 단어에 대해서 가중치 합을 구하는 방식이 아닌 특정 단어에 대해서만 확률을 구하는 방식을 사용하고
차이점은 Ptr-Net은 MT에서 rare words 문제를 해결해주기 위해 나온 모델이므로 encoder-decoder 구조를 가지고 있다.

## 데이터셋
CNN, Daily Mail, Children's Book Test

## 학습 조건
- optimizer: Adam optimizer
- learning rate: 0.001/0.0005
- word embedding matrix init value: randomly uniformly from the interval [-0.1, 0.1]
- GRU network weight init value: random orthogonal matrices
- biases init value: 0
- gradient clipping threshold: 10
- batch size: 32
- regularization: no
- ensemble: yes, voting

## 평가 방법
평가는 ensemble voting을 통해 accuracy를 구하는 방법을 택했다.

## 성능
AS Reader의 성능은 아래와 같다.
![result2](https://whikwon.github.io/images/NLP_attentionsum_result2.png) <br>
<center> <i> &lt;AS Reader의 CNN, Daily Mail dataset에 대한 평가 결과&gt;</i> </center> <br>

![result](https://whikwon.github.io/images/NLP_attentionsum_result.png) <br>
<center> <i> &lt;AS Reader의 CBT dataset에 대한 평가 결과&gt;</i> </center> <br>


Reference: <br>
Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, Jan Kleindienst. [Text Understanding with the Attention Sum Reader Network](https://arxiv.org/pdf/1603.01547). 2016.
