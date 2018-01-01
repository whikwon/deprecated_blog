---
title:  "Faster-R-CNN code 분석"
date: 2017-09-18 00:00:00
comments: true
---

- Faster-R-CNN 코드 분석

학습 순서는 이미지를 넣어주고 RPN으로 자르고 RoI한 뒤에 RPN의 loss와 RoI의 loss를 합친 후에
optimzer로 업데이트 해주는 방식이다.

살펴 볼 요소
1. RPN 방법, 상세한 요소들
2. RoI 방법, 상세한 요소들

## 구성 요소

- train_net.py
가장 중요한 메인 실행 스크립트이다.
  1. argument 가져온다.
    - script : config.py
    - method : cfg_from_file
    - arg : filename
  2. image db 정보 가져온다.
    - script : factory.py
    - method : get_imdb
    - arg : imdb_name
    - return : imdb
      - pascal_voc.py : pascal 데이터에 대한 정보를 가져온다.
  3. training roi 데이터를 가져온다.
    - script : train.py  
    - method : get_training_roidb
    - arg : imdb
    - return : roidb
      - roidb.py : prepare_roidb(imdb)
  4. output을 저장할 dir를 정한다.
    - script : config.py
    - method : get_output_dir
    - arg : imdb, weights_filename
    - return : output_dir
  5. 학습 때 사용할 device 지정
    - 따로 없음. arg에서 선언해준 값.
  6. 학습 때 사용할 network 지정
    - script : factory.py
    - method : get_network
    - arg : network 이름. arg에서 선언해준 값.
    - return : VGGnet_train()
      - override하는 모델 코드
      - network.py : anchor_target_layer 만들어주는 코드인듯 함. 
  7. 학습 시작
    - script : train.py
    - method : train_net
    - arg : network, imdb, roidb, output_dir, pretrained_model, max_iter)
      - filter_roidb : roidb 중에 사용할 수 있는 값들을 고른다.
      - SolverWrapper : snapshot, modified_smooth_l1, train_model 이 있다. snapshot은 살펴볼 예정, 나머진 loss 구하고 training 시키기.
