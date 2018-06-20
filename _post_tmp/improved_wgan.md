---
layout: post
title: 'Improved training of Wasserstein GAN'
author: kion.kim
date: 2018-**-** 17:00
tags: [deeplearning, GAN, WGAN]
---

## 시작하며

GAN이 JS divergence에 준하는 분포간의 거리측도를 사용한 반면, WGAN은 W-거리라는 새로운 분포의 거리 측도를 이용하는 모형입니다. GAN의 instability의 원인 중 하나로 discontinuous한 거리측도라고 본 셈입니다. 아주 수학적으로 fancy하고, GAN의 stability를 architecture가 아닌 theory로 풀려는 접근이었습니다. W-거리를 구하기 위해서는 Lipschtz 함수족에 대해서 기대값을 계산하여야 하는데, 논문의 저자들은 고민 끝에 gradient를 clipping함으로써 Lipschtz 함수를 만들어 낸 것입니다. 논문에서 다음과 같이 이야기 하고 있습니다.

>![wgan_lipschitz](/assets/wgan_lipschitz.png)

본인들 스스로 `terrible`이라는 단어를 쓰면서, 많은 연구자들의 참여를 기다린다고 하는군요. 그래도 여전히 어느정도 잘 작동한다고 하기는 하는군요. 얼마 지나지 않아 Ishaan Gulrajani 외 4인의 'Improved training of Wasserstein GAN'이라는 논문에서 여기에 대한 대안을 제시합니다.
Weight clipping 대신에 critic 함수가 Lipschitz condition을 만족하기 위해 이 논문에서 어떻게 해결했을까요? 그 해답은 regularization입니다.


## Improved WGAN

언제나 그렇듯 새로운 논문이 나오게 되면 그 논문이 왜 나오게 되었는지를 설명하기 위해서 기존의 방법들을 부정하게 됩니다. 부정까지는 아니고, 단점을 언급하는 것이죠. 그런 단점을 보완한 방법이 바로 저자들이 주장하는 방법일테니까요. 여기서도 WGAN 논문의 몇가지 단점에 대해서 언급합니다.

저자들이 가장 먼저 언급한 단점은 weight를 clipping함으로써 critic의 surface가 너무 단조로워진다는 것입니다. 여기서 critic의 surface가 무슨 의미일까요? 결국 critic도 하나의 neural network입니다. 해당 network를
![improved_training_wgan_fig](/assets/improved_training_wgan_fig.png)


![improved_training_wgan_prop](/assets/improved_training_wgan_prop.png)
