---
title: cifar10_resnet
date: 2017-09-04 00:00:00
comments: true
---

- CIFAR10 데이터셋으로 ResNet을 구현해놓은 코드를 분석한 자료이다.

### 구조

1. sub
- generate_cifar10_tfrecords.py : 보유하고 있는 CIFAR10 자료를 binary tfrecords 형식의 파일로 변환해서 저장한다.
여기에서의 장점은 train, eval, val을 따로따로 빼내어서 저장한 후에 사용해서 효율적으로 메모리를 운영할 수 있게 된다.
- model_base.py : ResNet 모델을 구성하는 모듈을 생성한다. 아래 model script에서 가져다가 쓴다.
  1. 필요 정보
    - batch_norm 관련 정보 (is_training, epsilon, )
    - Conv나 기타 모델은 다 high-level로 변수 필요없다.    
2. main
- cifar10.py : tfrecords 형식의 데이터를 불러오고 batch 크기에 맞게 불러오고 전처리해준다.
  1. 필요 정보
    - data 파일 위치 및 이름
    - train, eval 종류
    - batch 크기
  2. 진행 순서
    - 파일 이름을 불러온다.
    - queue를 만든다.
    - 전처리를 한다.
    - 배치를 만든다.
- cifar10_util.py : 로그 남기기 위함인거 같은데 아직은 잘 모르겠다. (일단 패스)
- cifar10_model.py : 모델 구상 및 inference 위에서 정의된 모듈을 가져와서 사용한다.
- cifar10_main.py : train, eval이 함께 진행된다. 그리고 CPU를 사용할건지 GPU를 사용할건지 사용하면 몇 개
를 사용할건지 설정하고 그에 맞게 loss를 구하고 합치는 작업을 지정해준다.


# CS231n assignment3

1. Config 지정해주고

2. Model 생성
  - placeholder
  - feed_dict
  - prediction
  - loss
  - training
  - run_epoch
  - train_on_batch
  - fit

# Hands on ML

1. Inception 모델 불러오기

2. Checkpoint restore해서 변수 값 지정해주고 training 시키지 않기 위해 냅두고
다른 layer 새로 연결

3. txt 파일을 불러와서 image, label 지정하고 batch 불러오기 (이제 feed_dict 사용 금지)
-> Queue 사용의 효율성이 어떤지 확인 필요하다.
