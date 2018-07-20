---
layout: post
title: 'GAN - 나를 맛가게 한'
author: kion.kim
date: 2018-0x-xx 17:00
tags: [deeplearning, self-attention, nlp, sentence representation]
---


## 시작하며

GAN이 워낙 hot하기도 하고 해서 여가시간을 활용해서 coding을 해보기로 했습니다. 처음에는 워낙 개념도 간단해 보여서 주위에서 아무리 어렵다 어렵다 해도 그러려니 하면서 구현은 차일피일 미루다가 이번에 GAN을 구현해 보기로 하고 하나하나 차근차근 구현해 보았습니다. 역시 녹록치는 않았습니다. 코드를 복붙하는 것과 실제로 바닥부터 다시 모형을 구축하는 건 또 다릅니다. 여기서 말하는 바닥은 하드코드 엔지니어가 말하는 Cuda programming과 같은 그러한 레벨은 아닙니다. Deep learning framework를 이용해서 데이터를 준비하고 network를 정의하는 과정도 그 어떤 reference를 보지 않고 구축하기란 조금 다른 얘기였다는 것이지요. 삽질하면서 얻은 내용들을 정리해 보고자 합니다.


## GAN이 뭔가?

GAN은 가장 대표적인 생성 모형으로 2013년 Ian Goodfellow의 논문에서 시작이 되었습니다. 두개의 서로 경쟁하는 network들이 하나의 network는 계속 이미지를 만들어 내고(generator), 다른 하나의 network는 만들어진 image(fake image)와 실제 이미지(real image)를 구분하는 역할을 합니다. 위조지폐를 만들어내는 위조범과 위조 지폐를 감별하는 경찰의 이야기는 이제 이미 지겨운 cliche가 되어버린지 오래입니다.

가장 간단한 형태의 network는 
