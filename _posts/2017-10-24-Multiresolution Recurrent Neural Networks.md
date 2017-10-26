---
title:  "Multiresolution Recurrent Neural Networks"
date: 2017-10-24 00:00:00
comments: true
---

- 학습하기 어려운 coarse token을 따로 빼서 natural language token과 따로 학습을 시킨 뒤 합치는 과정을 통해 성능을 개선한다.

## 모델 목적
1. 여러 개의 input을 넣어서 output을 생성하는 모델을 구축하는 것.
2. 대화 내용 중 high-level abstraction에 대해 학습을 잘 하도록 만드는 것.
핵심은 coarse와 natural language token의 분리를 통해서 위 목적을 달성하는 것이며 중요하다고 생각되는 정보에 힘을 실어서
decoding할 때 반영이 되도록 해주는 것이라고 생각할 수 있다.   

## 모델 구조
Multiresolution RNN(MrRNN)는 아래 그림과 같은 구조를 가지며 크게 Coarse, Natural Language token을 처리하는 부분으로 나뉜다. <br>

1) Coarse representation <br>
- encoder: embedded된 coarse token을 받아서 encoding을 한 뒤 마지막 hidden state를 decoder로 넘겨준다.
- decoder: encoder의 마지막 hidden state 값을 받아 decoding을 한다.
- prediction encoder: decoder에서 generated된 token을 encoding해서 vector로 만든다. <br>


2) Natural language representation <br>
- encoder: embedded된 natural language token을 받아서 encoding을 한 뒤 마지막 hidden state를 decoder로 넘겨준다.
- embedding: encoder의 마지막 hidden state와 위 coarse 처리 시 prediction encoder에서 나오는 vector를 합쳐준다.
- decoder: coarse의 high-level 정보와 natural language 정보를 합쳐서 decoding을 한다. <br><br>
![structure](https://whikwon.github.io/images/NLP_multiresolution_structure.png) <br>
<center> <i> &lt;Multiresolution RNN 구조&gt;</i> </center> <br>

## 부가 설명
Coarse, Natural Language의 구분은 여러가지 방법이 있을 수 있는데 논문에서는 Noun, Activity-Entity representation 두 가지로
나누어서 진행했다. 두 방법 모두 POS tagger를 사용하였고 여기서 coarse로 선정한 기준은 해당 품사들이 대화의 내용에 많은 비중(*high-level abstraction*)을
차지하고 있다는 가정이 전제되어 있다.

## 이전 모델과의 비교
Hierarchical Recurrent Encoder-Decoder(HRED)의 구조를 기본으로 한다. HRED는 encoder RNN, context RNN, decoder RNN으로 구성되며
encoder RNN을 통과한 정보가 context RNN으로 입력이 되고 context RNN의 output이 decoder RNN으로 입력되어 문장을 생성한다.
자세한 모델의 구성은 잘 모르겠어서 논문을 보고 더 부가 설명하도록 하겠다.

## 데이터셋
Ubuntu Dialog Corpus, Twitter Dialogue Corpus

## 학습 조건
- optimizer: Adam optimizer
- early stopping
- learning rate: 0.0001/0.0002
- embedding dim: 300(Ubuntu), 400(Twitter)
- gradient clipping threshold: 1.0
- batch size: 40/80
- beam width: 5

## 평가 방법
precision, recall, F1, accuracy를 metrics로 활용했다.

## 성능
MrRNN의 성능은 아래와 같다. <br>
![result1](https://whikwon.github.io/images/NLP_multiresolution_result.png) <br>
<center> <i> &lt;MrRNN의 Ubuntu Dialog Corpus 대한 평가 결과&gt;</i> </center> <br>

![result2](https://whikwon.github.io/images/NLP_multiresolution_result2.png) <br>
<center> <i> &lt;MrRNN의 Twitter Dialogue Corpus에 대한 평가 결과&gt;</i> </center> <br>

![example](https://whikwon.github.io/images/NLP_multiresolution_example.png) <br>
<center> <i> &lt;모델 별 Ubuntu model 답변 예시&gt;</i> </center> <br>

Reference: <br>
Iulian Vlad Serban, Tim Klinger, Gerald Tesauro, Kartik Talamadupula, Bowen Zhou, Yoshua Bengio, Aaron Courville. [Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation](https://arxiv.org/pdf/1606.00776.pdf). 2016.
