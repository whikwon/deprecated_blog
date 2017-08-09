---
title:  "Git tutorial"
date: 2017-08-09 00:00:00
comments: true
---

- 생활코딩에서 지옥에서 온 Git 강의 내용을 정리하였다.

**Git Bash** : Windows에서 Linux 기반의 명령어를 사용할 수 있도록 해주는 프로그램이다.
  - ls -al : 현재 작업 폴더 내 파일 정보 확인 명령어
  - clear : 지난 입력/출력 내용을 다 지운다.
  - vim 1.txt : 1.txt를 생성한다. i를 누르면 입력 가능.
  - cat 1.txt : 1.txt 내용 확인.
**git init** : 현재 작업 폴더를 git 작업 폴더로 만들어주면서 .git 폴더를 생성하며 해당 폴더에 버전
관련 정보가 누적되게 된다.
**git status** : 현재 상태 확인
**git add** *filename* : git에 관리하는 파일 추가/수정 사항 추가
**git config --global user.name** *username* : 작업 시에 누가 했는지 알려줌
**git config --global user.email** *useremail* : 작업 시에 어떤 이메일이 했는지 알려줌
**git commit** : 버전을 업데이트한다.
  - -a : add를 자동으로 한다.
  - -m : message를 editor를 안 키고 업데이트하겠다.
  - -am : 위 2개를 합친 내용
**git log** : 버전 업데이트 상황 확인한다.
  - -p : 이전 commit과의 변경점을 보여준다.
  - *loginfo* : loginfo에 해당되는 commit의 변경점을 보여준다.
**git diff** : 현재 작업에 대한 변경점을 보여준다. (commit 전, add시 사라짐)
  - *loginfo1*..*loginfo2* : log1과 log2의 commit을 비교해서 나타내준다.  
**git reset** :
  - *loginfo* --hard : loginfo이 후의 버전을 삭제하고 해당 loginfo단계로 돌아간다.
**git revert** :
**git 명령어 --help** : 명령어에 해당되는 도움말 확인


git commit 전에는 항상 git add가 와야한다. 파일 하나 하나에 대해서 선택적으로 변경사항에 대해
commit해야 할 필요성이 있기 때문이다. (기존 버전 관리 시스템과 차별화된 점이라고 한다.)

commit 대기 상태는 ***stage area*** 에 있는 것이다.
git에는 stage(*commit 대기*)와 repository(*commit 완료값*)라는 개념이 존재한다.

commit 마다 주소가 존재한다. 이를 통해서 log를 알 수도 있다.

reset과 revert의 차이점을 알아야 한다.
reset을 하면 log가 사라지긴 하는데 다시 복구할 수도 있다.
reset은 협업 시에 절대 사용하면 안 된다.

***
git의 원리
1. 폴더 내 변경 점은 add 전에는 git에 아무런 영향을 주지 않는다.
2. 변경점을 add할 시에 object폴더 내 파일을 바라보게 하는 index가 생성되고 값은 object에 저장된다.
index의 앞의 두 글자가 object내 폴더명, 나머지가 파일명이 된다.
3. 같은 값들은 하나의 object에 저장되게 되고 서로 다른 파일들에 같은 변경점이 add될 경우
이들은 모두 같은 index를 갖게 된다. (중복에 대한 비효율을 막기 위함.)
4. sha1라는 메커니즘을 통해서 hash값을 얻는다. 그럼 같은 값에 대해 동일한 index를 얻을 수 있다.  




Reference: <br>
[생활코딩 - 지옥에서 온 Git](https://opentutorials.org/course/2708)
