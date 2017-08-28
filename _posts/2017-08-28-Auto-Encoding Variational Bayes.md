---
title: Auto-Encoding Variational Bayes
date: 2017-08-28 00:00:00
comments: true
---

- GAN, FVBN과 함께 가장 인기가 많은 generative model인 VAE을 소개하는 논문이다.

먼저, Generative 모델은 무엇인가? data를 생성할 수 있는 모델이다. <br>
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
VAE에서는 $$P(z|X)$$를 Gaussian 분포인 $$Q(z|X)$$으로 근사하고 두 분포의 차이인
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



Generative vs Discriminator : https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm


Reference: <br>
