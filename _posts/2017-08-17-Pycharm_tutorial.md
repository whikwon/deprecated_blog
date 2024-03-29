---
title:  "Pycharm tutorial"
date: 2017-08-16 00:00:00
layout: post
excerpt: "Python에서 가장 많이 사용하는 IDE인 Pycharm의 유용성, 단축키 정리이다."
categories: [Python, IDE]
comments: true
---

- Pycharm을 사용하면서 유익한 tip에 대해서 정리하였다.

## 왜 Pycharm?

먼저, Pycharm을 사용하게 된 몇 가지 이유에 대해서 설명하고 이 IDE를 다루는 방법에 대해서 소개하도록 하겠다.

원래 Spyder라는 IDE를 잘 사용하고 있었는데 작업을 하면서 몇 가지 단점이 눈에 보이기 시작했고
일을 처리함에 있어서 반복적으로 걸림돌이 되면서 결국 Pycharm으로 변경하게 되었다.

그럼, Spyder의 단점은 무엇일까? <br>
자유도가 매우 떨어지며 너무 가볍다는 게 큰 단점(*반대로는 장점*)이다. <br>
거의 대부분 설정의 경우 초기값을 그대로 사용해야 한다. 짧은 script 작성을 위해 사용하기에는 편리하나
기본 설정들이 변경이 어려워 불편한 점에 대해서는 계속 불편을 겪어야 한다. 또한, 가벼운 script 작성 외
기타 기능을 거의 지원하지 않고 있다.

**그럼, Pycharm에서 얻은 장점들은 무엇인가?** <br>
첫번째로 다양한 툴과 연동시켜서 사용할 수 있다. 예를 들면 Git, Docker, Conda 등과 연동이 되어
작성하는 코드를 repo에 쉽게 올릴 수 있고, 가끔 python2 버전이 필요한 경우에 변경해서 사용할 수도 있다. <br>
두번재는 모든 작업이 단축키로 가능할 정도로 자유도가 높다. 이는 계속 사용할 수록 다루는 스킬이 점점 나아질 수 있다는
의미기도 하다. <br>
마지막으로는 코드 분석을 매우 용이하게 하도록 해준다. 내 컴퓨터에 있는 모든 패키지를 indexing해서(*무겁다*) 참조하는 모듈, 함수 등을
바로바로 알려준다. 그리고 마우스로 클릭하면 바로 어떤 함수인지 확인할 수 있어 앞으로 코드 분석할 때 매우 유용하다.

이런 장점들이 필요하게 된 계기는 보고 다루는 코드의 양이 많아졌기 때문이다. 그래서 조금은 더 나은 도구를 찾게 되었고
만족하고 있다.

***
## 단축키 정리

1. 검색
  - Ctrl + Shift + A : 모르는 걸 다 물어볼 수 있는 창이다. 무엇을 배우든 가장 먼저 익혀야 할 것.
  - Shift + Shift : 아무거나 다 찾는다. 파일명/변수/함수/등
  - Ctrl + E : 최근 방문 목록
  - Ctrl + Q : Documentation을 본다.
  - Ctrl + P : Parameter 정보를 본다.
  - Ctrl + B : 함수 위치 따라 들어간다. (navigate)
  - Alt + 좌/우 : 창을 좌/우로 이동한다. 각 창에 커서는 원래 위치한 곳으로.

2. 실행
  - Alt + Shift + E : 특정 범위만 console에서 실행
  - Alt + Shift + F10 : Script 실행

3. 이동
  - Alt + 1 : project 파일 list로 이동
  - Alt + 3 : console로 이동
  - Alt + 6 : TODO list로 이동
  - Alt + F12 : terminal(cmd)로 이동
  - ESC : escape, 작성 중이던 script로 이동하거나 실행창을 끈다.
  - Ctrl + Shift + F7 : 해당 이름의 변수/함수를 highlight한 뒤에 F3을 누르면 이동할 수 있도록 한다.
  - Ctrl + B : 변수/함수가 어디에 사용되고 있는지, 알려준다. 있는 곳으로 이동한다.
  - Alt + up/down : methods 간에 이동한다.
  - Ctrl + Shift + 위/아래 : 현재 줄의 내용을 위/아래로 이동시킨다.
  - Ctrl + W + F : 단어 전체 선택하고 find 실행한다. 위/아래로 이동시킨다.

4. 수정
  - Shift + F6 : 이름 바꾸기
  - Ctrl + D : 윗 줄 복사
  - Ctrl + Shift + J : 아래 줄 끌어올려서 한 줄로 만들기
  - Ctrl + / : comment/uncomment

5. 기타
  - # TODO [할 일]: TODO-List 작성
