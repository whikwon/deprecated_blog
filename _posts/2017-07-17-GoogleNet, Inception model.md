---
title:  "GoogleNet, Inception model"
date: 2017-07-17 00:00:00
comments: true
---

- GoogleNet은 Google에서 발표한 2014년도에 ILSVRC에서 1등을 차지한 모델이다.  
- 이 후 몇 가지 개선사항을 추가해서 발표한 inception 시리즈에 대해서는 업데이트 예정이다.

***
### 1. Introduction <br>
핵심적으로 이 논문에서 언급하는 내용은 **inception layer** 의 도입으로 쌓이는 layer
에 의한 computational budget 해결 및 mobile, embedded computing을 위한 모델 개발 필요성이다.

***
### 2. Related Work <br>
*Despite of concerns that max-pooling layers result in loss of accurate spatial information*
ResNet에 이어서 pooling의 한계점에 대해서 언급하고 있는데 이 부분은 현재 pooling에 대한
trend를 확인한 후에 기록할 예정이다.
GoogleNet은 크게 2가지 연구에 영향을 받았다. 첫째는 neuroscience model of primate visual cortex, Serre et al.
에서 다양한 크기의 fixed Gabor filter를 사용해서 multiple scale을 다룰 수 있게 만든 점이다.
여기에서 착안해서 다양한 크기의 filter를 내재하는 inception layer를 만들었고 기존과 다르게 학습 및 반복적으로 사용할
수 있도록 하였다.
두번째는 [Network in Network - Lin et al.](https://arxiv.org/pdf/1312.4400.pdf)에서 representation power를 늘리기 위한
1X1 convolution layer 사용하는 점을 참고하여 GoogleNet에서 depth를 늘리며 연산을 줄이기 위한 dimension reduction 모듈의 용도로
사용한다.
뒷 부분에 R-CNN에 관련한 내용이 나오는데 image segmentation, detection에 대한 부분은 관련
논문을 읽고 마저 정리할 예정이다.

***
### 3. Motivation and High Level Considerations <br>
당시 trend는 neural network를 layer width, depth를 늘려 깊게 학습시키면 좋은 성능을 발휘하였다.
근데 깊게 학습할 경우에 overfitting과 computation resource의 문제가 생기는데 이를 둘 다 해결하기
위해서 sparsity의 도입이 필요하다고 얘기하고 있다.(이를 위한 FC → Convolution 대체 필요.)
또한, 데이터의 확률 분포를 아주 큰 신경망으로 표현할 수 있다면 높은 상관성을 가지는 출력들과 이 때
활성화되는 neuron들의 cluster 관계를 분석하여 최적의 topology를 구성할 수 있다고 한다. Arora의 [논문](http://proceedings.mlr.press/v32/arora14.pdf "보기") 참조.
그런데, 컴퓨터 연산은 uniform distribution일 때 가장 효율적이어서 정반대의 문제에 봉착한다. (sparse는 비효율적)
이를 해결하기 위해 sparsity는 결국 망 내의 연결을 줄이면서 세부적인 행렬 연산에서는 최대한 dense한 연산을 하도록 해야 하는데
기존에는 convolution을 활용해서 filter-level sparsity를 얻을 수 있었고, 다음 단계에로 inception architectuure를 연구 중이다.

***
### 4. Architectural Details <br>
*There will be a smaller number of more spatially spread out clusters that can be covered by convolutions
over larger patches, and there will be a decreasing number of patches over larger and larger regions*
위의 문제가 어떤 건지 제대로 이해가 안 된다. 위와 같은 patch-alignment 문제를 해결하기 위해 inception architecture
내에선 1X1, 3X3, 5X5 filter size만 사용하고 pooling도 효과가 있을까 해서 넣었다.
inception module을 도입했을 때 문제가 3x3, 5x5 convolution 연산량이 크다는 점인데, 이를 해결하기 위해서 1x1 convolution이
사용된다. (3X3 → 1X1 + 3X3, 아래 그림 참조.) <br>
![figure2](https://hackathonprojects.files.wordpress.com/2016/09/inception_implement.png?w=649&h=337)<br><br>
1x1 convolution 개념이 매우 중요한데 연산을 쪼개서 진행할 경우에 parameter수를 줄여 dimension을 낮춰주고
각 단계 별로 ReLu가 적용되어 non-linearity도 높혀주기 때문이다.<br>
결국 inception 구조의 이점은 두가지로 정리할 수 있다. 첫째, 연산량의 큰 증가 없이 넓은 망을 구축할 수 있다는 점.
둘째, 하나의 이미지 정보에 대해 여러 scale로 처리가 되고 합쳐져서 다음 단계에서 다양한 scale에 대한 feature를
추출할 수 있다.

***
### 5. GoogleNet <br>
GoogleNet의 상세한 구조, parameter수는 아래 표에 잘 나타나있다.
![table1](https://learningcarrot.files.wordpress.com/2015/11/googlenet-parameters.png) <br><br>
 #1X1, #3X3, #5X5, pool proj가 합쳐져서 inception layer output을 형성하게 되고 #3X3 reduce는 #3X3전에 1X1 convolution을
 수행하여 얻어진 feature-map의 개수를 의미한다. <br>
 ***#3X3 reduce, #5X5 reduce*** 항에서 알 수 있듯이 #3X3, #5X5 연산을 위한 feature-map이 크게 감소해서 연산량이 줄어듬(각각 50%, 91.7%)을 확인할 수 있다.
추가적인 사항으로 GoogleNet에서 이전 버전의 모델에서 자주 사용되진 않는 average pooling이 사용하였다.
이는 NIN에서 영향을 받은 점이며 효과적으로 feature-vector들이 추출되어 pooling만으로 classification이 가능하다고 설명한다.
그리고 새로운 항목으로 auxiliary classifier가 등장하는데 regularization 효과를 주기 위한 용도로 사용되었다.
구조는 간단하게 training 중간에 가지를 내어 output을 내고 이 Loss(x0.3)를 최종 output의 Loss와 함께 계산하여 gradient를 update해준다.
이 regularizer는 물론 test할 때는 사용하지 않는다. 아래 그림에서 우측 부분이 auxiliary classifier이며 전반적인
구조를 참조하기 바란다. <br>
![Figure3](https://qph.ec.quoracdn.net/main-qimg-0ed62ffe5bea704d591887c768e2ca14) <br><br>

***
### 6. Training Methodology
이것 저것 설명을 하고 있는데 명확하게 guide를 주는 건 없고 sampling을 이렇게 해라 정도만 나와 있다.

***
### 7. Result <br>
  - 같은 모델 7개를 sampling만 다르게 해서 학습시키고 ensemble하였다.
  - test시에 AlexNet보다 더 cropping을 많이 진행했다. 이미지 1개 당 144crop 사용.
  - multiple crop에 대한 softmax probabilities average로 prediction 진행하였다.
  아래 표는 classification 결과, crop수에 따른 성능 비교 결과이다.
  ![table2,3](http://img.blog.csdn.net/20160920104209399)

***
### 8. Detection Challenge Results
나중에 추가로 학습한 뒤에 채울 예정.

***
### 9. Conclusion
*Approximating the expected optimal sparse structure by readily available dense building
block is a viable method for improving neural networks for computer vision*
다시 한번 spasity에 대한 강조를 하고 있다.



Reference: <br>
Szegedy, Christian, et al. ["Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf), 2015.
