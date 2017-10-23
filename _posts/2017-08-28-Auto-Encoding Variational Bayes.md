---
title: Auto-Encoding Variational Bayes
date: 2017-08-28 00:00:00
comments: true
---

- GAN, FVBN과 함께 가장 인기가 많은 generative model인 VAE을 소개하는 논문이다.

## Variational_Bayesian_methods란?

**Variational Baysian** 는 machine learning이나 bayesian inference시 intractable한 integral을 approximate할 때 사용하는 방법 중에 하나이다.
주로 observed variable, parameter, latent variable이 있는 복잡한 statistical한 모델에 사용된다. 두가지 목적으로 주로 사용된다. <br>
첫째는 posterior probability of unobserved variable의 analytical approximation을 위해서 쓰인다. 목적은 statistical inference이다. <br>
둘째는 marginal likelihood of observed data의 lower bound를 끌어내기 위해서 사용한다. 목적은 model selection하기 위함으로 marginal likelihood가
높으면 data에 더 잘 fit할 것이라는 아이디어에서 출발한다.

## mean-field approximation

variational inference에서 위에서 얘기한 것처럼 latent variable $$z$$의 data $$X$$가 주어졌을 때의 posterior distribution을
구하는게 목적으로 사용되는데 intractable하니 근사하기 위해서 variational distribution을 이용한다.
$$P(Z \vert X) \sim Q(z)$$
여기서 $$Q(z)$$는 단순한 분포를 사용해서 $$P(z \vert X)$$와 최대한 가깝게 하는 것이 목적이므로 KL divergence를 최소화하는 방향으로
가면 될 듯하다. KL divergence는 아래 식으로 나타난다.
$$D_{KL}(Q \Vert P) = \Sigma_z Q(Z) log {Q(z) \above 0.5pt P(z \vert X)}$$ Bayes 정리를 사용해서 식을 조금 더 풀면
$$logP(X) = D_{KL}(Q||P) - \Sigma_z Q(z) log {Q(z) \above 0.5pt P(z,X)} = D_{KL}(Q||P) + L(Q)$$ 라는 식을 얻을 수 있고
$$logP(X)$$는 $$Q$$에 대해 고정이니 $$L(Q)$$를 최대화하는게 $$Q$$로부터 $$P$$의 KL divergence를 줄일 수 있는 길이다.
이 내용을 VAE에 사용해서 Object function을 구하게 된다.

## Variational AutoEncoder란?

VAE를 소개하기에 앞서 **generative 모델** 은 무엇일까? generative모델은 data를 생성할 수 있는 모델이다. <br>
앞에서 본 GAN의 경우가 generative 모델에 속하며 VAE와는 조금은 다른 형태를 가지고 있다. <br>

GAN에서는 Generator와 Discriminator을 학습시켜 내쉬 균형에 최대한 도달하도록 한 후
random noise가 학습된 Generator를 통과하며 transformation되었을 때 우리가 원하는 data 샘플을
얻는 방식을 취한다.

그 반면에 VAE의 경우에는 z라는 latent variable을 도입해서 이로부터 우리가 원하는 data를
생성하는 원리를 갖는다. (*autoencoder의 encoder, decoder를 생각하면 되겠다.*)

latent variable은 우리의 상상력, 어떤 추상적인 feature라고 생각하면 되겠다.
이 블로그[https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/]에서 예시를 통해 잘 설명하고 있는데
번역을 통해 이해를 돕기로 하겠다.

먼저, Notation은 아래와 같다.

>
1. $$X$$ : 모델링하려는 data (*animal*)
2. $$z$$ : latent variable (*우리의 상상력*)
3. $$P(X)$$ : data의 확률 분포 (*animal kingdom*)
4. $$P(z)$$ : latent variable의 확률 분포 (*상상력의 원천, 뇌*)
5. $$P(X | z)$$ : latent variable이 주어졌을 때 생성 data의 분포 <br>
(*상상력이 실제 animal로 되는 것*)


우리는 $$P(X)$$를 알고 싶다. law of probability에 따라 $$P(X) = \int P(X \vert z)P(z)dz$$로 나타낼 수 있다.
$$P(X,z) = P(X \vert z)P(z)$$ 이므로 $$P(X,z)$$를 알거나 $$P(X \vert z), P(z)$$을 알면 생성 모델을 만들 수 있겠다.

VAE의 핵심 아이디어는 위 식의 $$P(z)$$ 대신 $$P(z|X)$$를 사용하는 것인데 이건 직관적으로 생각해봤을 때는
가지고 있는 data로 부터 얻은 latent variable이 우리가 생성하려고 하는 data의 latent variable에서
그렇게 벗어나지 않을 것이기 때문이다. (*동물들을 예시로 가진 사진들에서 보든 전체 동물 모두를 고려하든
더듬이가 있다던지 기계로 되어있다던지... 이럴리는 없고 특징은 비슷비슷할 것이다.*)

그럼 $$P(z|X)$$를 구해야 하는데 이 또한 아직 모르는 값이므로 Variational Inference(VI)를 해준다.
(*Bayesian inference 방법이 한 가지로 다른 대표적인 예로는 MCMC 방법이 있다.*)
VI는 복잡한 모델을 단순한 형태의 분포로 근사해서 optimization 문제로 접근하자는 방법이다.
VAE에서는 $$P(z|X)$$를 Gaussian 란분포인 $$Q(z|X)$$으로 근사하고 두 분포의 차이인
KL divergence를 최소화하는 방법을 택할 수 있다.

$$\begin{align}
D_{KL}[Q(z | X) \Vert P(z | X)] &= \sum_z Q(z | X) \, \log \frac{Q(z | X)}{P(z | X)} \\
                            &= E \left[ \log \frac{Q(z \vert X)}{P(z \vert X)} \right] \\
                            &= E[\log Q(z \vert X) - \log P(z \vert X)
\end{align}$$ <br>

위 식에서 Bayes rule을 사용해서 식을 일부 수정하면 <br>
$$\begin{align}
D_{KL}[Q(z \vert X) \Vert P(z \vert X)] &= E \left[ \log Q(z | X) - \log \frac{P(X | z) P(z)}{P(X)} \right] \\
                                        &= E[\log Q(z | X) - (\log P(X | z) + \log P(z) - \log P(X))] \\
                                        &= E[\log Q(z | X) - \log P(X | z) - \log P(z) + \log P(X)]
\end{align}$$ <br>

위와 같이 나타낼 수 있고 expectation항에 z에 depend하지 않는 항들을 밖으로 꺼내면 아래와 같이 된다. <br>

$$\begin{align}
D_{KL}[Q(z \vert X) \Vert P(z \vert X)] &= E[\log Q(z \vert X) - \log P(X \vert z) - \log P(z)] + \log P(X) \\
D_{KL}[Q(z \vert X) \Vert P(z \vert X)] - \log P(X) &= E[\log Q(z \vert X) - \log P(X \vert z) - \log P(z)]
\end{align}$$ <br>

항들을 조금 더 정리해보자.<br>

$$\begin{align}
\log P(X) - D_{KL}[Q(z \vert X) \Vert P(z \vert X)] &= E[\log P(X \vert z) - (\log Q(z \vert X) - \log P(z))] \\
                                       &= E[\log P(X \vert z)] - E[\log Q(z \vert X) - \log P(z)] \\[10pt]
                                       &= E[\log P(X \vert z)] - D_{KL}[Q(z \vert X) \Vert P(z)]
\end{align}$$ <br>

최종적으로 얻은 결과는 아래와 같다.
$$\log P(X) - D_{KL}[Q(z \vert X) \Vert P(z \vert X)] = E[\log P(X \vert z)] - D_{KL}[Q(z \vert X) \Vert P(z)]$$

위 식을 봤을 때 구조는 $$Q(z \vert X)$$가 encode net, $$z$$가 encoded representation, $$P(X \vert z)$$가
decoder net을 하는 **autoencoder** 의 구조와 유사함을 알 수 있다. 그래서 붙여진 이름이 **VAE** 이다.

**Object function** 의 항들을 설명하면 첫번째 항은 reconstruction loss이다. encoder를 통과한 z가 X를 생성했을 때 얼마나 유사한지
에 대한 expected negative log likelihood라고 보면 되겠다. 이 항은 decoder가 reconstruction을 잘하게 해준다.

두번째 항은 $$q_{\theta}(z \vert x)$$ 와 $$p(z)$$의  KL divergence로 $$q$$로 $$p$$를 나타냈을 때 얼마나 많은 정보 손실이
있는지, 얼마나 둘이 가까운지 나타내주는 값이다. <br>
주로 VAE에서 $$p(z) = N(0,1)$$로 두고 진행하기 때문에 $$z$$가 normal distribution과 다르면 패널티를 부여한다.
$$z$$가 한쪽으로 치우치지 않고 다양하게 분포할 수 있도록 일종의 **regularizer** 역할을 한다고 볼 수 있겠다.

그럼 encoder와 decoder의 각각 parameter들에 대해 loss function을 optimize해서 각각 값들을 update하게 될 것이다.

이제 그럼 어떻게 objective function을 optimize할 것이냐는 문제만 남았다. <br>
첫번째 항인 $$D_{KL}[Q(z \vert X) \Vert P(z)]$$은 $$Q(z \vert X)$$를 Gaussian으로, 여기서 샘플링한 z를 받은 $$P(z)$$도 Gaussian으로 설정한다.
그럼 수식을 풀어서 계산할 수 있다.(논문 Appendix 참조.) <br>

나머지 항인 $$E_{z \sim Q}[log P(X \vert z)]$$가 문제가 되는데 $$z$$의 샘플링이 많아질 수록 연산량이 커져서 문제가 된다.
그래서 SGD처럼 1개의 $$z$$를 뽑아서 위 항을 $$P(X \vert z)$$라고 근사하고 전체 dataset $$D$$에서 $$X$$ 샘플을 뽑아서 SGD를 하는 것으로 생각할 수 있다. <br>
최종적으로 아래 식이 구해지고 식을 들여다보면 $$P(X \vert z)$$가 $$Q$$에 더이상 dependen하지 않게 된 것을 볼 수 있다. <br>
$$E_{X \sim D} [log P(X) - D[Q(z \vert X) \Vert P(z \vert X)]] = E_{X \sim D[log P(X \vert z) - D[Q(z \vert X) \Vert P(z)]]}$$
이럴 경우에 Lower bound식에 $$Q$$에 관련된 항도 존재하기에 제대로 학습되기가 어렵다.

이렇게, $$z$$가 샘플링되는 과정 때문에 발생하는 문제를 살펴보았고 이 문제를 ***reparamerization trick*** 을 사용해서 문제를 해결한다.
![VAE-process](https://whikwon.github.io/images/VAE-proccess.png)
기존 모델(왼쪽)을 보면 $$Q(z \vert X)$$로부터 얻은 $$\mu(X), \Sigma(X)$$로부터 $$z$$를 샘플링해서 바로 넣어주는 방법을 취했다면
trick을 사용해서(오른쪽) noise variable $$\epsilon \sim N(0,I)$$로부터 샘플링을 해서 $$z = \mu(X) + \Sigma^{1/2}(X) * \epsilon$$의 식으로부터
$$z$$를 구해서 넣어주는 방식을 택한다. 아래 그림에서 설명이 아주 잘 나타나 있다. 파란색 원(*random node*)에서는 Gradient descent가 막혀버려서
이를 해결해준 것이라고 보면 되겠다. <br>
![reparametization](http://1.bp.blogspot.com/-V-m6dOVaUL8/WQ2JKJ4Jj4I/AAAAAAAABrA/BjxqKMDfR6ggYCCqUNlBFiS4cqlyisgKACK4B/s1600/vae_3.PNG)

그렇게 최종적으로 gradient descent를 할 식은 아래와 같다. <br>
$$E_{X \sim D} [E_{\epsilon \sim N(0, I)}[log P(X \vert {z = \mu (X) + \Sigma^{1/2}(X) * \epsilon})] - D[Q(z \vert X) \Vert P(z)]]$$

Test시에 샘플을 generate할 때는 encoder를 제외하고 $$z \sim N(0,I)$$에 놓고 진행하면 된다. <br>
$$Q(z \vert X)$$가 $$P(z)$$에 충분히 가까워졌으면 좋은 샘플을 얻을 수 있겠다. Test 시 진행과정은 아래 그림과 같다. <br>
![test_time](https://whikwon.github.io/images/test_time.png)

Tutorial 내용

단순히 X를 그냥 만드는 건 매우 어려우니 z라는 latent variable을 도입해서 $$f(z;\theta)$$가 $$X$$에 가깝게 가도록 하자.
law of total probability에 의해서 $$P(X) = \int P(X \vert z;\theta)P(z) dz$$로 낱나낼 수 있고 maximum likelihood를
하면 우리가 원하는 $$P(X)$$가 최대가 되는 $$\theta$$(?)를 구한다.
그리고 $$P(X \vert z;\theta) = N(X \vert f(z;\theta), \sigma^2 * I)$$로 output 분포를 Gaussian이라고 가정한다.
뭐 Gaussian이 아니어도 상관없는데 $$\theta$$에 대해서 continuous하고 연산 가능하면 된다. (그림으로 봤을 때 pixel의 output이 흑과 백 즉, 0,1이면 bernoulli 분포가 나오겠다.)

$$p(z)$$를 prior, $$p(x)$$를 observed로 놓고 posterior를 구하는게 목표라고 했을 때 Bayes 정리에 의해서 $$p(z \vert x) = {p(x \vert z)p(z) \above 0.5pt p(x)}$$
가 성립하는데 intractable해서 유사 분포로 근사시킨다. 그게 바로 $$q_{\lambda}(z \vert x)$$ 여기서 lambda는 어떤 분포인지 나타내준다. (e.g. Gaussian)

VAE는 단순히 autoencoder와 연산 방식이 비슷해서 붙여진 이름.


Reference: <br>
Diederik P Kingma, Max welling. [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114). 2013. <br>
Carl Doersch. [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908). 2016. <br>
[wikipedia - Variational-bayesian_methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) <br>
[stackoverflow - generative vs discriminative](https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm) <br>
[blog1](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) <br>
[blog2](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/) <br>
