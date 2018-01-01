---
title:  "Decoupled Neural Interfaces using Synthetic Gradients"
date: 2017-07-18 00:00:00
layout: post
excerpt: "2016년에 Deepmind에서 발표된 논문으로 Synthetic gradient에 대해 다룬다."
categories: [Paper]
comments: true
---

## 1. introduction <br>
Neural network의 최적의 weight를 구하기 위해서는 input에 대한 output 값을 구하고 loss를 구한 뒤에 forward,
back propagation을 통해 각각의 weight의 gradient를 update해주어야 한다. <br>
이 과정은 순서에 따라서 진행되어 이전 과정이 진행되지 않았을 경우 이 후 과정이 진행되지 못하는 ***locking*** 이 일어난다.
이런 ***Forward Locking, Update Locking, Backwards Locking*** 은 network의 running 및 update를 sequential, synchronous하게
진행되게 하며 이는 시간이 오래 걸리고 때에 따라선 intractable하게 된다. <br>
그럼 asyncronous하게 update를 했을 때 이점은 뭐가 있을까? 바로 parallelise training이다. 동시적으로 network의 running, update를
해줄 수 있으므로 연산 시간을 단축시킬 수 있다. <br>
이 논문에서는 update locking을 없애서 학습 시간 및 성능을 확인할 에정이다. 방법은 아래와 같다.
핵심 내용은 $$h_i, x_i, y_i, \theta_i$$에 dependent하던 BProp식을 $$h_i$$에만 dependent하다고 가정하는 것이다.<br><br>
$$\frac {\partial L} {\partial \theta_i} = f_Bprop((h_i, x_i, y_i, \theta_i),...) \frac {\partial h_i} {\partial \theta_i} \simeq \hat f_Bprop(h_i) \frac {\partial h_i} {\partial \theta_i}$$ <br>
($$h:activation, \ x:input, \ y: supervision, \ L: loss$$) <br><br>
그럴 경우 위 식은 $$h_i$$에만 dependent한 식이 되어 각 layer을 지날 때 Bprop을 통해 해당 layer의 gradient를 구해 즉시 update를 할 수 있게 된다.
뒤에서 RNN에의 적용 예제를 보면서 어떤 방식으로 적용되는 지 살펴보겠다.

## 2. Decoupled Neural Interfaces
asynchronously learning 할 수 있는 high-level communication protocol을 먼저 소개하겠다. <br>
![Figure1](https://norman3.github.io/papers/images/synthetic_gradients/f03.png) <br>
$$(f_A,f_B:\ model,\ h_A: \ output\ of\ model A,\ M_B: feedback\ model, c: other\ information,\ S_B:\ state\ of\ B,\ \hat \delta_A: synthetic\ gradient,\ \|\delta_A - \hat \delta_A\|:\ Loss\ for\ M_B)$$ <br><br>
위 protocol의 진행 과정을 설명하자면 $$M_B$$라는 $$B$$에 딸린 utility에 현재 running되는 정보, $$h_A, S_B, c$$를 전달하여 synthetic error $$\hat \delta_A$$를 구해서 즉시 $$f_A$$를 update한다. 그리고 끝까지 forward propagation, $$f_A$$까지 back propagation이 되어 얻어진 true utility값 $$\delta_A$$ 과 비교해서 $$M_B$$를 update한다. <br>
이 protocol에서 중요한 내용은 $$f_A$$와 $$f_B$$가 ***decoupled하게 update된 다*** 는 점이며 이를 활용해서 Decoupled Neural Interfaces(DNI)를 구상할 수 있다. <br>
아래 그림은 위에서 설명한 과정이 순서대로 일어나는 과정이다. (참조)
![DNI flow](https://storage.googleapis.com/deepmind-live-cms-alt/documents/3-6.gif)

## 2.1 Synthetic Gradient for Recurrent Networks
RNN에 적용한 결과를 설명한다. <br>
RNN이 무한히 전개된다고 생각했을 때를 생각해보면 $$N \to \infty$$가 되면서 $$F_1^{\infty}$$인 RNN이 된다. (아래 그림과 참조.) <br>
![RNN infinite](https://storage.googleapis.com/deepmind-live-cms-alt/images/3-7.width-1500_jiACRLG.png) <br><br>
이를 계산하기는 어려워서 실제로는 아래와 같은 형태로 구간을 나누어 weight를 update하는 과정을 수행하며 해당 과정을 truncated BPTT라고 한다. <br>
![RNN truncated BPTT](https://storage.googleapis.com/deepmind-live-cms-alt/images/3-8.width-1500_3rdF9so.png) <br><br>
내용을 수식으로 나타내면 <br>
$$\theta - \alpha \sum_{\tau=t}^{\infty} \frac{\partial L_\tau}{\partial \theta} = \theta - \alpha\left( \sum_{\tau=t}^{t+T} \frac{\partial L_{\tau}}{\partial \theta} + \left( \sum_{\tau=T+1}^{\infty}\frac{\partial L_{\tau}}{\partial \theta}\right)\frac{\partial h_T}{\partial \theta}\right) = \theta -\alpha\left( \sum_{\tau=t}^{t+T} \frac{\partial L_{\tau}}{\partial \theta} + \delta_T \frac{\partial h_T}{\partial \theta}\right)$$
이렇게 했을 때 단점은 RNN의 weight가 학습할 수 있는 time horizon 구간, temporal dependency를 제한한다.
이런 단점을 DNI로 해결할 수 있는데 방법은 아래 그림과 같이 **No backprop** 구간에 **synthetic gradient** 를 도입하는 것이다.
![RNN + DNI](https://storage.googleapis.com/deepmind-live-cms-alt/images/3-9.width-1500_1ahGJNx.png) <br><br>
이렇게 했을 때 time step간에 gradient가 전달이되어 synthetic gradient가 정확한 값으로 적용되었을 경우에 infinitely unrolled RNN에서
backpropagation을 한 효과, 즉 longer temporal dependencyy를 얻을 수 있다.
위 DNI protocol을 설명 시에 언급했던 synthetic gradient에 대한 training도 중요한데(***Auxiliary task***) true gradient를 실제로 얻는게 intractable하므로
target gradient라는 개념을 사용해서 bootstrapping으로 구한다고 한다. (***나중에 배우면 update 예정***)

## 2.2 Synthetic Gradient for Feed-Forward Networks
정리 필요


Reference: <br>
Jaderberg, Max, et al. ["Decoupled neural interfaces using synthetic gradients."](https://arxiv.org/pdf/1608.05343.pdf) arXiv preprint arXiv:1608.05343 (2016).
DeepMind Blog : https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/
