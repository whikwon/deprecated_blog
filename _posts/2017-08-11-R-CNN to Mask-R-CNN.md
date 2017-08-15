---
title:  "R-CNN to Mask R-CNN"
date: 2017-08-11 00:00:00
comments: true
---

- image detection의 대표적인 모델인 R-CNN부터 Mask R-CNN까지의 흐름에 대해서 정리하도록 하겠다.

먼저, image detection을 제외한 다른 vision 분야에서 연구되는 주제에 대해서 소개하도록 하겠다. <br>
방법에 따라 아래와 같이 4가지로 구분되어 나누어진다. <br>
![Vision tasks](https://whikwon.github.io/images/vision_tasks.PNG) <br> <br>
1. Semantic Segmentation  <br>
image내 pixel들이 어떤 class를 나타내는지 찾는 방법이다.
주로 학습에 사용되는 방법은 **Fully Convolutional로 transpose convolution** 을 사용한 up/down sampling으로
각 pixel에 대한 classification을 수행한다. label data는 당연히 pixel별로 classify된 data를 사용한다.

2. Classification + Localization  <br>
image내 **특정 object** 가 어느 위치에 있는지 확인하는 방법이다. <br>
기본적인 학습 방법은 Bounding-Box에 대한 loss를 구하고 image에 대한 classify loss를 구해서
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
어떻게 흘러왔는 지에 대한 내용 정리

1. R-CNN
  CNN을 활용해서 image detection을 성공적으로 해낸 모델이다. 다음에 계속 소개할 모델들은 R-CNN의 구조를
  차용하고 발전시켰다.  <br>
  크게 구성은 아래 그림과 같이 region proposals을 추출하고, CNN에서 feature를 구한 뒤에 나온 결과를
  classify하는 순서로 구성된다. (*training 하는 방법은 조금 아래에*)
  ![R-CNN](https://whikwon.github.io/images/R-CNN.PNG) <br>

  위 내용을 좀 더 세부적으로 설명하자면 **selective search** 를 이용해서 Input image를 수 많은 **region proposals(bounding boxes)** 로 세분화한다.
  이들 proposals은 제각각 다른 size를 갖고 있기 때문에 warp해서 같은 크기의 image로 만들어준다.

  이렇게 만든 image를 CNN 구조에 넣어주어 feature를 계산하는 작업을 거치게 된다. 해당 논문에서는 AlexNet을
  사용하였다.

  이렇게 image의 proposals에 대한 feature를 구했으니 classify하는 일만 남았다. 여기에선 SVM을 사용해서
  classify하였다.

  세부적인 논의 사항이 필요한데, 먼저 ***bounding box는 정확할까?*** 에 대한 답변은 ***No*** 이다.
  selective search할 때 얻은 proposals들이 부정확하기 때문에 이를 학습시켜줘야 한다.
  그래서 bounding box regression을 CNN에서 나온 feature에 학습시켜준다.

  두번째로는, AlexNet에서 사용했던 데이터와 detection을 위해 사용하려는 데이터가 다른 것 같은데 어떻게 해야할까?
  라는 물음에는 pre-training을 해주면 된다. AlexNet의 마지막 FC layer의 class개수를 사용하려는 데이터의 총 class개수로
  변경해준 후에 해당 데이터로 pre-training을 해주면 문제없다.

  R-CNN은 CNN의 classification의 월등한 성능을 이용해서, 하나의 image를 더 잘게 쪼개서 보는 방법으로 다양한 object들을
  제각각 detect할 수 있을까에 대한 자연스러운 접근 방식으로 볼 수 있겠다. <br>
  하지만 초기 모델이다보니 단점이 극명하게 드러나는데 연산량에 대한 문제와 end-to-end 학습이 되지 않는다는 점이다.
  SVM을 학습시켜주고 Bounding-Box도 학습시켜줘야하고 CNN도 pre-training 시켜줘야하고.. 손이 많이 간다.
  이를 단계적으로 해결하는 방향으로 추가 모델들은 움직이게 될 것이다.

  - bounding box regression
  자세하게 설명되어 있는 블로그가 찾기 어렵다. [논문](https://arxiv.org/abs/1311.2524)을 보자.

  - selective search
  [selective search](http://www.cs.cornell.edu/courses/cs7670/2014sp/slides/VisionSeminar14.pdf)
  ![SS](https://cdn-images-1.medium.com/max/1600/1*ZQ03Ib84bYioFKoho5HnKg.png)

2. Fast R-CNN
  위에서 설명한 R-CNN내 Region proposals들이(1개 image당 약 2000개) 생성된 후에 CNN을 통과할 때
  실제로 엄청나게 많은 부분들이 겹친다는 사실을 깨닫게 된다. <br>
  그래서 이런 반복적인 Region proposals을 어떻게 효율적으로 할 수 있을까에 대한 해답으로 아래
  그림의 RoIpooling을 제시한다.
  ![RoIpool](https://whikwon.github.io/images/RoIPooling.PNG) <br>

  내용은 매우 간단하다. Region proposals들이 형태가 있다고 가정하고 이미지를 CNN을 통과시킨 후에
  나오는 feature들에 각각의 proposal들을 매칭시켜서 feature를 구한다. 그리고 이들을 RoIpooling(*max-pooling으로 구성 됨*)
  을 해서 모두 같은 크기로 만들어주고 이를 FC layer 넣어 classification을 수행한다. <br>
  전체 구조를 정리한 내용은 아래와 같다. training 시에는

  ![Fast R-CNN](https://whikwon.github.io/images/Fast-R-CNN.PNG)

3. Faster R-CNN

4. Mask R-CNN


Reference:  <br>
1. selective searach image : https://www.koen.me/research/pub/uijlings-ijcv2013-draft.pdf
2. Girshick, Ross, et al. ["Rich feature hierarchies for accurate object detection and semantic segmentation."](https://arxiv.org/abs/1311.2524)
Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
3. [Stanford CS231n Lecture11 slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)
