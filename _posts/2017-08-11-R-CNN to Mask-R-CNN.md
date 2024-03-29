---
title:  "R-CNN to Mask R-CNN"
date: 2017-08-11 00:00:00
layout: post
excerpt: "Image Detection/Segmentation의 대표적인 모델인 R-CNN부터 Mask R-CNN까지의 흐름 정리 내용이다."
categories: [Computer Vision, Image Detection, Paper]
comments: true
---

## Image Detection/Segmentation

먼저, image classification을 제외한 다른 vision 분야에서 연구되는 주제에 대해서 소개하도록 하겠다. <br>
방법에 따라 아래와 같이 4가지로 구분되어 나누어진다. <br>
![Vision tasks](https://whikwon.github.io/images/vision_tasks.PNG) <br> <br>
1. Semantic Segmentation  <br>  
image내 pixel들이 어떤 class를 나타내는지 찾는 방법이다.
주로 학습에 사용되는 방법은 **Fully Convolutional로 transpose convolution** 을 사용한 up/down sampling으로
각 pixel에 대한 classification을 수행한다. label data는 당연히 pixel별로 classify된 data를 사용한다.

2. Classification + Localization  <br>
image내 **특정 object** 가 어느 위치에 있는지 확인하는 방법이다. <br>
기본적인 학습 방법은 bounding-box(bbox)에 대한 loss를 구하고 image에 대한 classify loss를 구해서
이 둘을 weighted sum한 total loss를 objective function으로 두고 학습을 진행하는 방식이다.
image내 기본적인 box가 존재하고, class할 object의 수도 주어지므로 이어지는 object detection의
쉬운 방식이라고 보면 되겠다. 이 방식을 통해 Human pose estimation이 가능하다.

3. Object detection  <br>
**몇 개의 object가 image내에 있는지 주어지지 않은 상태** 에서 각각의 object에 대한 위치와 class를 확인하는 방법이다.
classfication과 마찬가지로 computer vision 관련 가장 큰 task라고 보면 되겠다. 방법은 image를 ***많은 crop들로 나눈 다음에
이에 대해 개별적으로 classification을 수행한다.*** training시에 위 2번 방식과 동일하게 classification + Bounding-Box을
고려해서 많은 crop들에 대해 학습해야 하므로 매우 까다롭고 복잡하다. <br>
방식은 크게 2가지로 나뉘는데 첫번째는 **Region based method** 로 R-CNN, Fast-R-CNN, Faster-R-CNN, Mask-R-CNN 등이 포함된다.
두번째는 **proposal없이 진행되는 모델** 들로 대표적으로 YOLO와 SSD가 있다.

4. Instance Segmentation <br>
위의 내용들을 대부분 합친 내용이라고 보면 된다.
image내 많은 object들에 대해서 classification을 수행하고 각각의 object가 차지하는 pixel의 범위를 찾는 방법이다.
Mask-R-CNN이 최근에 나온 대표적인 모델이며 Region based method의 일종이다.
학습 방식은 Mask-R-CNN을 살펴보면서 보도록 하겠다.

***
Object detection 분야가 어떻게 흘러왔는 지에 대표적인 모델을 중심으로 내용을 소개하도록 하겠다.

## 1. R-CNN <br>

CNN을 활용해서 image detection을 성공적으로 해낸 모델이다. 다음에 계속 소개할 모델들은 R-CNN의 구조를
차용하고 발전시켰다.  <br>
크게 구성은 아래 그림과 같이 region proposals을 추출하고, CNN에서 feature를 구한 뒤에 나온 결과를
classify하는 순서로 구성된다. (*training 하는 방법은 조금 아래에*) <br>
![R-CNN](https://whikwon.github.io/images/R-CNN.PNG) <br>

위 내용을 좀 더 세부적으로 설명하자면 **selective search** 를 이용해서 Input image를 수 많은 **region proposals(bounding boxes)** 로 세분화한다.
이들 proposals은 제각각 다른 size를 갖고 있기 때문에 warp해서 같은 크기의 image로 만들어준다.

이렇게 만든 image를 CNN 구조에 넣어주어 feature를 계산하는 작업을 거치게 된다. 해당 논문에서는 AlexNet을
사용하였다.

이렇게 image의 proposals에 대한 feature를 구했으니 classify하는 일만 남았다. 여기에선 SVM을 사용해서
classify하였다.

세부적인 논의 사항이 필요한데, 먼저 ***bbox는 정확할까?*** 에 대한 답변은 ***No*** 이다.
selective search할 때 얻은 proposals들이 부정확하기 때문에 이를 학습시켜줘야 한다.
그래서 bbox regression을 CNN에서 나온 feature에 학습시켜준다.

두번째로는, AlexNet에서 사용했던 데이터와 detection을 위해 사용하려는 데이터가 다른 것 같은데 어떻게 해야할까?
라는 물음에는 pre-training을 해주면 된다. AlexNet의 마지막 FC layer의 class개수를 사용하려는 데이터의 총 class개수로
변경해준 후에 해당 데이터로 pre-training을 해주면 문제없다.

R-CNN은 CNN의 classification의 월등한 성능을 이용해서, 하나의 image를 더 잘게 쪼개서 보는 방법으로 다양한 object들을
제각각 detect할 수 있을까에 대한 자연스러운 접근 방식으로 볼 수 있겠다. <br>
하지만 초기 모델이다보니 단점이 극명하게 드러나는데 연산량에 대한 문제와 end-to-end 학습이 되지 않는다는 점이다.
classification을 학습시켜주고 bbox도 학습시켜줘야하고 CNN도 pre-training 시켜줘야하고.. 손이 많이 간다.
이를 단계적으로 해결하는 방향으로 추가 모델들은 움직이게 될 것이다.

- bounding box regression <br>
자세하게 설명되어 있는 블로그가 찾기 어렵다. [논문](https://arxiv.org/abs/1311.2524)을 보자.

- selective search <br>
Region proposal에 사용하는 selective search에 대한 설명은 다음 링크를 참조하도록 하자. [링크](http://www.cs.cornell.edu/courses/cs7670/2014sp/slides/VisionSeminar14.pdf) <br>
아래는 예시 그림이다.
![SS](https://cdn-images-1.medium.com/max/1600/1*ZQ03Ib84bYioFKoho5HnKg.png)

## 2. Fast R-CNN

R-CNN에서 가장 큰 문제는 training뿐 아니라 test시에 시간이 너무 오래걸린다는 점이다.(49초/VGG16)
Fast R-CNN에서 위 문제에 대한 큰 부분을 해결하였으며 핵심 내용은 RoIpooing이다.(2.3초/VGG16)

- **RoIpooling** <br>

  위에서 설명한 R-CNN내 Region proposals들이(1개 image당 약 2000개) 생성된 후에 CNN을 통과할 때
  실제로 엄청나게 많은 부분들이 겹친다는 사실을 깨닫게 된다. <br>
  그래서 이런 반복적인 Region proposals을 어떻게 효율적으로 할 수 있을까에 대한 해답으로 아래
  그림의 RoIpooling을 제시한다.
  ![RoIpool](https://whikwon.github.io/images/RoIPooling.PNG) <br>
  내용은 매우 간단하다. Region proposals들이 형태가 있다고 가정하고 이미지를 CNN을 통과시킨 후에
  나오는 feature들에 각각의 proposal들을 매칭시켜서 feature를 구한다. 그리고 이들을 RoIpooling(*max-pooling으로 구성 됨*)
  을 해서 모두 같은 크기의 RoI feature vector로 만들어주고 이를 FC layer를 통과시켜 classification
  과 bbox regression을 수행한다. <br>
  RoIpooling을 통해서 기존에 SS 이후에 47초 정도 걸리던 runtime을 0.32초로 단축시켰다. <br>
  (*이제 2초 남았다 -> Faster-R-CNN*)

전체 구조를 보면서 추가적으로 R-CNN에서 개선한 내용을 살펴보도록 하자.
![Fast R-CNN](https://whikwon.github.io/images/Fast-R-CNN.PNG)

[개선 사항] <br>
1. training을 single-stage로 할 수 있도록 구조를 변경. <br>
RoIpooling을 통과한 vector가 classfier, bbox regression 각각에 대한 loss를 한번에
구할 수 있도록하여 training을 쉽게할 수 있도록 하였다. <br>
2. **"background class** 를 추가하여 object class에 대해서만 학습을 진행할 수 있도록 변경. <br>
**IoU**(intersection over union)을 정해서 어떤 기준으로 background를 결정할 것인지 정한다. (0.5)
3. bbox regression 함수를 $$L_2$$ loss에서 robust $$L_1$$ loss(smooth $$L_1$$)으로 변경. <br>

추가적으로 scale invariance에 대한 내용이 잠깐 언급된다. **brute force(노가다)** 나 **multi-scale**
training 방법을 통해서 scale invariance를 얻을 수 있지만 메모리를 크게 잡아먹어 어려움이 있다고 한다.
Faster R-CNN에서 이를 다른 방법으로 접근해서 해결하는데 뒤에서 보도록 하겠다.

## 3. Faster R-CNN

앞선 모델들의 많은 개선을 통해 detection 속도가 빨라지긴 했으나 실시간으로 사용하기에 2.3초는 충분히 빠르진 않다. <br>
Fast R-CNN에서 SS이후에 가장 큰 bottleneck을 RoIpooling으로 해결했다고 하면 이제 실시간 detection을 위해 SS에
걸리는 2초에 대해 개선을 할 차례이다. 이에 대한 내용이 Faster R-CNN이며 핵심 내용은 **RPN(Region Proposal Networks)** 이다.

- **Region Proposal Networks(RPN)** <br>

  아래 그림을 보면서 RPN의 개념에 대해 설명하도록 하겠다. <br>
  ![RPN](http://mithril-ntu.github.io/L10/Screen%20Shot%202016-05-18%20at%204.22.52%20PM.png) <br>

  RPN의 핵심적인 아이디어는 network 내에 Region proposal할 수 있게 해서 전체를 Fully convolutional network로
  만들자는 데에서 출발한다. region proposal을 위해서 conv layer들을 지난 feature map을 $$n$$ x $$n$$의 sliding window를
  통과시켜 기존보다 낮은 차원의 vector로 mapping시킨다. (ZF : 256-d, VGG : 512-d) <br>
  그리고 각각의 vector를 1X1 conv를 통해 *cls* layer와 *reg* layer로 전달하며 값을 얻는다. <br>
  여기서 *cls* 는 object/non-object를 구분하는 역할을 하게 되고 *reg* 는 bbox의 꼭지점을 찾는 역할을 하게 된다.
  그리고 여기서 **anchor** 라는 개념을 도입한게 된다. 위 그림에서 k개의 box가 의미하는게 anchor인데
  역할은 sliding window를 통과한 각각의 정보들이 참조할 수 있는 box를 만들어주는 것이다. <br>
  sliding window의 중심점을 기준으로 다양한 크기의 box를 만들고 이에 대한 정보와 ground truth 정보, 그리고 predict한
  정보까지 비교해서 학습 시에 가장 실제 값에 유사한 anchor로 예측할 수 있게끔 도와주는 역할이라고 보면 된다. <br>
  해당 논문에선 k = 9(*3 scales, aspects ratio each*)를 사용했고 anchor를 통해 다양한 scale을 고려하게 되므로 <br>
  모델이 Translation-Invariant한 특성을 갖게 된다고 한다. 학습 상세한 조건은 아래에서 자세히 보자.<br>

- **Loss function of RPN** <br>

  Training시 기준이 전 모델들에 비해 까다롭다. ***1) IoU가 가장 높은 anchor에 positive label, <br>
  2) IoU가 0.7보다 높은 anchor에 positive label, 3) IoU가 0.3보다 낮은 anchor에 negative label <br>
  을 준 후에 positive label로만 학습을 진행한다.*** <br>
  Objective function은 아래와 같다. <br>
  $$L(\{p_i\},\{t_i\}) = {1 \above 0.5pt N_{cls}} \sum_i L_{cls}(p_i,p_i^*) + \lambda {1 \above 0.5pt N_{reg}} \sum_i p_i^* L_{reg} (t_i, t_i^*)$$ <br>
  $$i$$는 anchor의 index이고, $$p_i$$는 i가 object일 확률, $$p_i^*$$는 anchor가 positive면 1 외에는 0을 나타낸다.
  $$t_i$$는 predicted bbox의 정보를 의미하며, $$t_i^*$$는 ground-truth의 정보를 의미한다. <br>
  (*위치 정보는 x,y,h,w,로 이루어져 있으며 x,y는 box의 center, h,w는 각각 높이와 너비를 의미한다.*) <br>
  $$L_{cls}$$ 는 2개 class에 대한 log loss를 의미하며 $$L_{reg}$$는 $$t_i$$와 $$t_i^*$$의 smooth $$L_1$$ loss를 구한다. <br.
  이는 anchor box와 가까운 ground-truth의 차를 나타낸다고 보면 되겠다.
  위 내용을 정리하자면 1) object인지 아닌지에 대한 학습을 하며 2) bbox는 anchor와 ground-truth 와의 비교를 통해
  어느 정도 anchor를 기준으로 ground-truth에 가까워지게 학습한다.

- **training details**  <br>
  논문에서 학습했을 때는 $$128^2, 256^2, 512^2$$의 scale과 $$1:1, 1:2, 2:1$$ aspect ratio를 사용하였다. <br>
  그리고 VGGNet이 classification시에 마지막 conv layer의 spatial size가 7x7이라서 sliding window 하기 어렵지 않겠
  냐는 걱정을 할 수 있는데, detection에 사용되는 이미지는 classification에 사용되는 224x224보다 훨씬 큰 ***600x1000***
  이미지를 사용해서 마지막 layer를 거치고 40x60 크기의 spatial size를 얻을 수 있어 3X3 filter로 약 40x60x9개의
  anchor를 얻을 수 있다. 이들 중 바깥 부분의 cross-boundary를 제외하고 NMS를 하면 이미지 당 약 2000개의 bbox를
  얻을 수 있고 이들 anchor 중 각각 256개의 positive, negative label을 무작위로 샘플링해서 training시킨다.

- **Sharing Convolution Features for Region Proposal and Object Detection** <br>

  이제 RPN을 학습시켰으니 Fast R-CNN과 합치는 작업을 해야겠다. <br>
  4가지 step을 통해서 한다고 설명하고 있다. <br>
  먼저, RPN을 pre-trained model을 이용해서 독립적으로 학습시킨다. 그 다음에 학습된 RPN에서 생성한 Region proposals와
  pre-trained model을 이용해서 Fast R-CNN을 학습시킨다. 이 때까지는 서로 conv layer를 공유하고 있지 않아서 하나의 network에서 각각을 고정하고
  RPN, Fast R-CNN을 fine-tuning해주어 conv layer를 공유해서 하나의 network로 만들어준다.
  (*논문에선 둘을 합친 다음에 한 번에 training시키는게 어렵다고 얘기하고 있다.*)

추가 참고할 내용으로는 anchor 크기가 정해져 있기 때문에 더 큰 object을 detect하는데 좀 어렵지 않을까라는 생각이 들지만
object의 가운데에 초점이 맞춰지면 이를 중심으로 bbox의 크기를 조절해서 어느 정도까진 예측이 가능하다고 한다.  <br>

상세한 다른 모델과의 평가 결과는 논문을 참조하도록 하자. <br>
아무튼 Faster R-CNN의 의의는 RPN을 도입해서 실시간 object detection을 가능한 정도의 속도로 향상시켰다는 점이다.

## 4. Mask R-CNN
정리 필요.

Reference:  <br>
1. selective searach image : https://www.koen.me/research/pub/uijlings-ijcv2013-draft.pdf
2. Girshick, Ross, et al. ["Rich feature hierarchies for accurate object detection and semantic segmentation."](https://arxiv.org/abs/1311.2524)
Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
3. [Stanford CS231n Lecture11 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)
4. Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. ["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"](https://arxiv.org/pdf/1506.01497.pdf). 2016.
5. Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. ["Mask R-CNN"](https://arxiv.org/pdf/1703.06870.pdf). 2017.
6. [How RPN Works](https://www.youtube.com/watch?v=X3IlbjQs190) - YouTube
