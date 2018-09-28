---

layout: post
title: Machine Learning Tutorial 2018(주최:전자공학회)
feature-img: "assets/img/md_image/machine learning tutorial 1.png"

tags: [MLT2018, machine learning]
excerpt_separator: <!--more-->

---

### Day Session1: Maximum likely hood, Large Margin Prediction, Expection&Maximaztion 

- Kaist 유창동 교수

No free lunch: 좋은 결과를 내려면 Assumption을 해야 함. 존재하는 모든 function으로 모델링 할 수 없음

Averaging: 주어진 유한 샘플에서의 평균 <!--more-->

Expection: 전체 데이터에서의 기대 값, pdf를 알아야 하므로 variance가 있는 값을 출력함

Inference: From sampled observation, say something about the whole population

**VC theorm**: Test에러는 train 에러 + sqrt(L) 보다 작거나 같다.--> Test 에러 예측

$$ E_test = 
$$
E_{test} ≤ E_{train} + \sqrt{\frac{d(log(2n/d) + 1) + log(4/\sigma)}{n}}
$$
n: train data

d : VC dimention(Hypothesis 표현할 수 있는 fuction)

δ: 신뢰도, 어느정도의 classifier를 만들 것인가

**SVM using (Sub) Gradient Desecnt**

Convex는 미분 함수보다 본 함수가 위에 있으면 됨, 미분 불가능할 때는 해당 점에서 아무 함수나 convex 조건 만족하는 것을 하나 택하면 됨

learning rate에 1/step을 곱해주면 step에 따라 점차 learning rate이 줄어듬

i.i.d. assumption: independent & idetifically distruted random variables(서로 독립이고 확률분포가 같은 sub sample 분포)

**Maximum Likely Estimation(MLE)**: train 데이터에 대해 최대가 되는 것을 찾음. 데이터가 많으면  사용하면 됨.

**Maximum a posterior probability(MAP)**: 사전 확률을 가정, Bayesian Theory에서 Posterior는 Likely hood * Prior, 양쪽에 log를 취하면 곱이 덧셈이 되므로 log posterior는 log likely hood + log porior 텀이 되고, 결국, likely hood에 prior 분포가 regularization 역할을 한다.

---

### Day 1 Session2: Monte Carlo Simulations

- 연세대학교 박태영 교수

샘플링은 uniform(0~1)이 된다는 가정에서 출발

기본적인 샘플링은 uniform 확률로 값을 선택

**inverse CDF method**

uniform으로 선택된 값을 inverse CDF에 넣으면 x 값이 나온다. CDF의 정의가 입력값을 넣으면 확률이 나오므로

**Rejection method**

샘플링하기 쉬운 envelop function ge(x)를 가정하고, f(x)/ge(x)로 acceptance 확률을 정의한다. f(x)와 ge(x)의 차이가 크면 accept 안될 샘플이 많아짐

**MCMC(Markov chain Monte Carlo)**

이전까지는 independent이였는데, MCMC는 dependenc한 샘플링에 대한 고려이다.

Markov Chain은 state간의 관계(transition rule)를 알고 있을 때, stationary 분포를 찾는 것이 목적이라면,

MCMC는 stationary 분포를 알고 있을 때, efficient transition rule을 찾아서 샘플링하는 것이 목적

dependent해도 ergodic하면 T->unlimited stationary하고 unique하다. 

reversible Markov chain은 detailed balance conditions를 만족하여 stationary 분포가 존재한다는 것은 guarantee하지만 unique하지는 않다.

MCMC는 Variational Basyes와 목적이 비스한데 VB는 prior를 가정하여 posterior와 유사한 것을 추정한 후 거기서 샘플링하는 것이고 MCMC는 분포에 맞게 샘플링하는 관계를 찾는 것이다. VB는 분포를 가정하므로 biased 될 수 있으나 속도는 빠르다. partial collapsing 기법으로 MCMC의 속도를 개선하는 연구가 있으나 현재까지 practical한 field에서는 VB가 효율적이다.

---

### Day 2 Session 1: Nonparmetric Bayesian Modeling

- Kaist, 정연승 교수

Frequenctist는 일반적으로 주어진 데이터에서 확률 분포를 얻는 것은 Maximum Likely Hood를 사용한다. 데이터가 무수히 많으면 전체 데이터(posterior)와 비슷한 분포를 뽑을 수 있지만 그렇지 않으면 주어진 데이터에 over-fit된다.  (y: data, θ: parameters)
$$
y ∽ P(y|\theta)
$$
주어진 θ에서 y가 될 확률을 최대화(MLE)하도록 θ를 학습한다.

Bayesian은 posterior를 예측하기 위하여 모수의 분포를 가정한다. 
$$
y ∽ P(y|\theta) ,      y ∽ P(\theta)
$$
이것은 다시 joint probability로 다음과 같이 나타낼 수 있고,
$$
P(y, {\theta}) = p(y|{\theta})p({\theta})
$$
다시 Bayesian theory로 나타내면
$$
P({\theta}|y) = \frac{P(y|{\theta})P(\theta)}{P(y)} = \frac{P(y|\theta)P(\theta)}{\int{P(y|\theta)P(\theta)d\theta}}
$$
Posterior를 정의할 수 있다. 결국, posterior는 가정한 prior에서 likely hood가 최대화되도록 학습된다. 여러가지 posterior가 있을 수 있는데, 그 중에서 가정한 prior가 되는 것으로 학습하겠다는 것이다. 샘플이 굉장히 많지 않아도 posterior를 학습할 수 있음(MAP 방식)

이것은 여러 분포 중에서 가정한 prior에 대해 posterior를 구하겠다는 것으로 prior가 regularization term이 된다고 할 수 있다. 양쪽에 log를 취하면 loss의 regularization 처럼 더해진다.

우리가 prior를 Gaussian으로 가정하면 negative log likely hood는 MSE(Mean Square Error)가 되고, Bernoulli로 가정하면 Cross Entropy가 된다.

![1533633607617](C:\Users\USER\AppData\Local\Temp\1533633607617.png)

posterior는 prior와 likely hood 사이에 존재하게 되고, MAP에서도 데이터가 너무 없으면(likely hood의 분산이 커짐, 신뢰도가 낮아짐) Posterior가 prior에 많이 dependant하므로 posterior는 prior에 가까워진다. 반대로 데이터가 많으면 likely hood에 가까워져 MLE 모델이 된다.

![1533634102330](C:\Users\USER\Downloads\1533634102330.png)

Posterior가 복잡해지만 하나의 Gaussian으로 모델링이 불가능하다. 여러개의 Gaussian으로 이뤄진 Gaussian Mixture 모델을 사용하여 모델링한다. 
$$
f(y_i) = \sum_{k=1}^K \pi_k N(y_i; \mu_k, \sigma_k^2)
$$
k: mixture 숫자, πk: mixture의 weights

Bayesian Inference를 할 경우 Gaussian Mixture는 Dirichlet 분포로 가정한다.

Parametric은 multiple한 Gaussian으로 이뤄져있지만 분포의 갯수는 정해주어야 한다. 

non-parametric Bayesian은 Gaussian이 무수히 많다는 가정으로 모델링한다. 무한하다는 것이 아니라 무수히 많다는 가정. 나중에는 중요한 몇 개만 남게 됨. 자료가 많아서 분포가 복잡해 질 수록 늘어나게 됨

non parametric은 Dirichlet Precess Mxisture로 모델링한다. 

non parametric은 결국 확률모델을 가정하지 않고 푸는 것이라고 할 수 있다. 없는 것은 아니고 flexible한 것이다.

Prior는 Dirichlet Process Mixture(DPM)으로 되어 있음, 확률 분포 여러개로 prior가 모델링될 것인데, 그것들의 set을 만들어내는 function들을 만들어내는 함수 p(F), DPM

DP에서 k가 정해지면 gausian 분포를 5개를 가지고 있는 여러가지 형태의 process들이 샘플링됨

이것이 뽑히는 예를 들면 **Stick-breaking representation**이  있음. 처음에 하나가 뽑히면 1-a에서 뽑히고... 계속 됨. 이때의 확률은 여러가지 random하게 섞어서 subset을 형성함. 이때 선택은 Beta(1,a)분포에서 선택되는데 a가 매우 크면 잘게 쪼개져서 posterior를 더 잘 나타낼 수 있으나 시간이 오래걸린다.

Polya Urm Scheme은 먼저 하나가 선택되고 다음에 선택할 때는 이전에 속할 확률과 새로운 곳에 속할 확률의 합으로 표현. 이것은 China Restorant Process(CRP)와 유사하다. CRP는 파티션 분할이 목적이다. clustering, 어떤 table에 어떤 사람이 들어가 있는지 아는게 목적

DPM은 많은 process들로 구성되지만 결국, weight가 '0'에 가깝지 않은 important한 것의 몇 개가 안됨.

Inference(sampling)은 Gibbs samplig(basic MCMC)나 variational inference

Variational Inference는 가정을 하기때문에 biased되지만 속도가 빠르다. 결국, 연산량만 많다면 MCMC 샘플링 통계쪽에서는 Variational Inference를 고려하지 않는 경우가 많다고 함

**Hierarchical Dirichlet Process**은 여러 그룹이 있을 때 어느 그룹 안에 있는 cluster가 다른 그룹에도 있는 형태를 말함. 그룹간에 cluster를 share하고 있는 형태를 고려한 것

이것은 Chinese restaurant Franchise로 표현할 수 있고, 다른 table(그룹)인데 같은 메뉴를 먹을 수 있는 것을 의미하며, table이 달라도 같은 메뉴를 먹을 수 있음(다른 그룹에서 cluster를 공유하는 형태)

**Beta Process**

DPM은 unsupervised Clustering이라면 Beta process는 latent feature estimation이라고 할 수 있다. 어떤 원인이 무엇이었는지 확인이 가능하다. event에 관여한 latent feature를 찾는 것을 의미한다. 

여기서는 observation에서 feature를 공유한다. 여러 개의 feature 중에서 의미 있는 것만 선택되므로 weight는 binary matrix이다. 여기서는 Indian Buffet Process(IBP)로 representation 가능하다. Buffet에서 순서대로 지나가면서 음식이 담긴 점시(feature)를 선택하는 것이다. feature는 무한하다고 가정한다. feature의 순서는 중요하지 않음. DPM은 Gaussian으로 구성된다면 BP는 Bernoulli로 구성된다.  

---

### Day 2 Session 2: Online Learning/Optimization for Machine Learning(어려움^^;)

- Princeton, Elad Hazan 교수

Gradient Descent를 전체 데이터(epoch)에 대해서 하는 것은 시간이 너무 오래 걸려 비효율적이다. stochastic gradinet descent를 하면 수렴 시간이 너무 오래 걸린다. 결국, batch 단위로 regularization하면서 online learing한다. online learning은 데이터가 계속 추가되면서 학습이 이어지는 형태이므로 batch들의 optimum(minimum)이 전체의 minimum이 되어야 한다. 결국, regulariztion이 중요(iid가정), 무한히 많은 hypothesis를 갖는 loss 구조에서도 i.i.d.가정으로 잘 해결 가능.

 convex나 non-convex이냐에 따라 다른 optimization이 필요하고 adaGrad(adaptive Gradient Descent)는 convex나 non-convex 둘 다 가능하고 수렴속도도 빠르다. non convex는 2nd oder GD로 해결가능하다. 

regularization은 classifier에서 reduce complexity이고 optimize 측면에서 안정화 하는 것이다.

adaptive Grad는 adaptive하게 regularization하는 것이다. 일정한 update는 sparse한 곳을 계속 일정하게 업데이트하므로 비효율적인데 adaGrad는 sparse한 곳에서 learning rate이 커진다. 

GD++에서는 Variance를 줄이는 기법도 있고, 2nd order를 사용하는 Newton's method도 있다. 

새로운 approach로 GGT가 있다.   *여럽다. 그냥 adaGrad쓰자, GGT나*

---

### Day 3 Session 1: Memory Network, Neural Turing Machine(NTM)

- 서울대, 검건희 교수

**MemN2N 모델**

Memory Network는 모든 data를 memory에 가지고 있으며, 어디를 읽을지를 학습하는 것

이미지에서는 CNN이 거의 솔루션이 되었는데, Language Model에서는 RNN이 솔루션이 아님, LSTM의 hiddee이 memory 역할을 하지만 너무작아 long term 씬에서는 처음 input을 잃어버린다.

그래서, large external memory를 가지는 모델을 제안하게 되었다. 

memory addressing은 0, 1 형태(hard)로 되어 미분 불가능 구간이 존재, 

memory addressing을 soft attention으로 변경, 모든 memory에 대해 softmax를 하여 미분가능한 addressing

메모리를 한번만 보는 것이 아니라 여러번 보는(hops) 방식 고안됨

글 또는 질문에 따라 sentence의 순서가 중요할 수 도 있고 아닐 수도 있다. 

구조는 sentense에서 A, C를 embeding하고(Bag of word, word to vector) 질문을 B로 embeding한 후에 B에서 embeding한 u와 A에서 embeding한 m을 가지고 p(weight) 메모리 attention(addressing) 을 만들고, C에서 embeding한 c와 weight p를 곱해서 output o를 만든다. 그리고, o와 질문이 embeding된 u를 합쳐서 Fully connected network로 학습하여 정답을 a를 찾는다.

A, B, C는 embeding하는 weight이다. 

multiple memory lookups(hops)은 FC 이전에서만 u(k+1) = u(k) + o(k) 으로 interation하여 embeding weight를 다르게 하는데 메모리를 줄이기 위해 비슷한 것들은 sharing하여 사용한다. 여기서 sharing하는 방법은 Adjacent와 Layer-wise 방식이 있다. 

그리고, Sentense를 representation(embeding)할 때, 2가지 방식이 있는데 Bag of Word는 전체 문장에 대해 동일한 확률을 적용하는 것이고, Position encoding(PE)는 문장의 처음과 끝에 weight를 주는 것이다.(문장의 처음과 끝의 중요도가 높음, 주로 영어에서)

그밖에도 여러가지 방식이 있는데, Random noise를 추가하기 위하여 10% 확률로 empty를 추가하기도 한다.

**Neural Turing Machine(NTM)**

동기는 LSTM은 숫자조차 copy하지 못한다고 함. 긴 숫자를 불러주면 memory 부족해서 앞에서 알려준 것을 잊는다고 함.

NTM은 MenN2N과 비슷한데 Memory를 Write(update)할 수 있다는 것이 큰 차이라고 할 수 있다. 

*Gradient Dscent는 미분가능하지 않아도 convex이면 사용가능하다. 그런데 network가 깊어지면 backpropagation이 깊어져 미분안되면 앞단까지 전달하기 힘들다.*

NTM에서는 heads라는 개념이 등장하는데 이 weight를 통해 Read/Write의 addressing이 결졍된다. 이 weight는 softmax의 output으로 총합은 1이다. 역시 여기서도 head는 soft attention으로 메로리를 addressing하고,

Read는 메모리에 이 weight를 곱하면 되고, Write는 이전 값을 컨트롤러(Neural network)의 output 중 wt를 이용해 이전 메모리에서 남길것과 지울 것의 비율이 결정되고 at를 weight에 곱해서 update한다.

LSTM처럼 일부만 기억에 남게 되는 것이다.

Head의 weight를 구하는 방법은 Content addressing과 Location addressing이 있는데 두 가지를 모두 사용한다.

 content addressing은 메로리의 content만 중요하게 생각하는 것으로 단순 곱을 사용하고, interpolation(preprocessing)후에 Location addressing하는데 특정 싸이즈 window로 convolution한다. convolution하고 나면 smoothing되는데 이후에는 sharpening하여 준다. (교재 page 230부터)

controller는 feedforward network나 LSTM을 사용하는데 LSTM을 사용하는 것이 좀 더 잘된다. 아마도 CPU에 chache를 약간 가지고 있는 형태라 feedforward보다 유리할 수 있다. 

글을 요약하는 문제에서 public DB를 사용하면 그냥 앞문장만 선택하는 것보다 성능이 떨어질 수 있다. 그런데 이것의 원인은 public DB가 두괄식으로 bias되어 있어서 이기때문이고, 대부분의 기사들이 앞쪽에 요약을 하는 형식으로 글을 쓰기 때문, 그래서 서울대에서는 DB를 직접 만들어버림, reddit쪽에 글 작성시 요약을 하는 것을 원칙으로하는 community가 있는데 이것을 활용, 다른 public DB와 비교하면 uniform하게 퍼져있음

여기서 몇가지 scheme들을 추가하여 학습하니 잘되었다. output에서 dilated convolution 사용 등

---

### Day 3 Session 2: Using Neural Networks for Modeling and Representing Natural Languages

- Facebook Tomas Mikolov

슬로바키아 명문대 출신으로 박사 논문에서 Word2Vector를 제안하여 자연어 학습에 혁신을 불러옴, 구글에 스카웃되었다가 facebook으로 옮김, 이후 큰 연구성과는 없는 것으로 보임

word2vec은 모든 단어에 vector로 표현하는 것이다. 이렇게 하면, queen - woman + man -> king 이런 것이 가능하고 simmility도 가능하다. 현재 이런게 vector화 해놓은 global dictionary를 구글이나 MS등에서 제공하고 있다.

Regularization이유: 뉴럴넷이 메모라이즈를 통해 학습해버리면 weight가 매우 큰 경우, complexity가 높은 상태인데 이 경우 Regulariztion 텀을 loss에 붙여주면 weight을 decay하는 효과가 있다. 

또한, vanishing gradient도 막아줄 수 있다. 

이후는 NLP 내용 ... ^^;

---

### Day 3 Session 3: Bayesian Machine Learning

- 서울대학교 김용대 교수

이 교육을 듣게된 이유이다. 김용대 교수님은 non-parametric bayesian 분야에서 국내 최고라고 함.

원래 MLT 교육에서 non-parametric 교육을 했는데 현재는 kaist 정연승 교수한데 넘기고 baysian neural network쪽을 강의한다고 함

일반적으로 **Frequenctist와 Bayesian**을 나누는데 Frequentist는 주어진 데이터로만 분포를 구하고, Bayesian은 prior를 가정하고 구한다고 말함

그리고, Frequentist는 실제 분포를 결정하는 옵티멈한 파라미터가 unknown이지만 fixed 되어있다고 보고 주어진 데이터로 estimation하는 것이고,

Bayesian은 observation만 fixed이고 나머지는 모두 unknown이며, unknown은 모두 random variable로 봄

그런데, 김용대 교수님은 Frequentist는 maximize나 minimize(estimate)해서 분포를 구하고, Bayesian은 적분해서 구한다고 함.

통계에서는 point 추정과 interval(region) estimation이 있는데 Bayesian은 interval 분포가 나오게 되므로 uncertainty를 알 수 있다고 함

Frequentist도 interval을 주어서(신뢰 구간 95%를 부여함) uncertainty를 줄 수 있는데 이렇게 하면 posterior (real) 분포에 interval을 주는 것이 아니므로 실제보다 uncertainty가 작게 추정된다. 즉, 테스트 환경에서 예상 못한 에러가 발생할 수 있다. 

Bayesian이 좋다. 그런데 왜 안쓰나? 적분을 해야 한다. bayesian theory에서 
$$
P({\theta}|y) = \frac{P(y|{\theta})P(\theta)}{P(y)} = \frac{P(y|\theta)P(\theta)}{\int{P(y|\theta)P(\theta)d\theta}}
$$
posterior는 결국, normalize(분모)를 적분해야 함. 이 적분이 힘들다. 그래서 많이 사용안한다.

이 적분을 구하는 방법이 여러가지가 있음

**1) Analytical approaches**

손으로 직접 계산하는 경우 conjugate prior의 경우만 가능한데, likely hood와 posterior의 분포가 같은 경우를 말한다. 그러나 이 경우도 계산할 경우 전문가(?)의 도움을 받아야 함, 파라미터가 많아지면 '0'에 가까워지기 때문. 그리고, 다루는 문제가 조금만 복잡해져도 conjugate prior가 성립안된다.

**2) MCMC**

그 다음으로는 MCMC가 있다. Bayesian은 MCMC가 가능해지면서 부흥 시작함. 계산량이 많았으나 성능 좋아지면서 가능해짐.(prior를 가정하고 하는 것이지만 무한대로 놓고 풀면 오차 줄어들고 무한대로 놓는 것은 너무 많은 계산량이 필요했으나 최근에는 가능해짐)

MCMC의 main idea는 Markov Chain으로부터 stationary한 분포가 되는 θ들을 찾는 것이다. 

Gibbs sampler는 conditional distribution으로부터 샘플을 만든다. 1차원문제에서 여러번 샘플링하여 고차원으로 확장한다. 

Metropolis-Hastings algorithm은 normalize없이 acceptance와 rejection을 정해서.. 결국, density가 높은 곳은 자주 방문하고, 낮은 곳은 덜 방문, 단, Random work를 할 때, 다른 분포로 넘어갈때가 느리다. 이런 속도를 높이는 것이 Hamiltonian MCMC 최근 논문

**3) Variational Inference**

 Variational Inference는 파라미터 v에 의한 variational distribution q로 posterior를 approximate하는 것이다.  q는 추정할 수 있는 분포이다. 우리가 구할 수 있는 분포로 posterior를 근사하는 것이다. 이 두 분포를 가깝게 근사하기 위해서 Kullback-Leibler Divergence를 이용한다. q에 의한 적분(평균)

normalize 텀 적분을 위해 ELBO를 정의하고, 이것은 결국 conditional 적분을 joint 적분으로 바꿔서 적분한다는 concept 이다. 

통계쪽에서 보는 Variational Inference는 KL에서 q에 의한 Expection이 좀 의심스럽고, point 추정은 ok인데,  interval(uncertainty)는 힘들것 같다고 한다고 함

**4) Assumed Density filtering(ADF)**

 posterior를 얻는 방법은 online으로 p(x|θ)q(θ)를 estimation하여 q(θ)에 넣고 계속 iteration함

variational inference에서 KLD는 KL(q||p)를 minimize하는 q를 찾는 것인 반면, ADF는 q에 의해 p를 approximate하는 것이다. KL(p||q)

**5) Expectation Propagation**

### Bayesian Machine learning

**Bayeian sparse factor modeling**

Indian Buffet Process처럼 latent factor(feature)를 찾는 것이다. neural network에서 fully connected된 것의 connection을 없애는 것이 sparse factor로 설명 될 수 있다.

일반적으로 prior를 선정할 때, 대충 줘도 데이터가 많아지면 likely hood가 중요해지므로 porior는 별로 중요해지지 않음. 그런데 factor 모델에서는 factor 모델의 갯수가 포아송 분포를 따라 데이터가 많아져도 문제가 될 수 있음

**Probabilistic(Bayesian) Neural Network**

![bayesian neural networkì ëí ì´ë¯¸ì§ ê²ìê²°ê³¼](C:\Users\USER\Documents\홍규석\gyuseog.github.io\_posts\machine learning tutorial 1.png) 

데이터가 없어서 uncertainty가 높은 부분은 variance가 큰 것을 볼 수 있음. 저 부분에서는 판단을 안하는 것(unknown)이 바람직할 수 있다. Bayesian Neural Network에서는 이런 것이 가능하다. 

**Variational AutoEncoder(VAE)**

Generative 모델, GAN과의 차이는 GAN은 discriminater를 속이는 출력을 내려는 경향(mode colaps)있다. 2 class의 분포일 때 잘 맞추는 하나의 class 분포만을 generate하는 경향

그러나, VAE는 분포는 잘 찾기 때문에 다양하게 분포를 찾는다. 다만 GAN보다 blury한데 이것은 학습 방법, 분포를 추정하는 방법의 문제라기 보다 분포를 구할 때 적분을 해서 smoothing 되는 것 때문이라고 예측된다고 함(by 김용대) 이분을 median이나 mode로 바꾸거나 하는 식으로 GAN만큼 해상도 높게 만들 수 있을 것이라고 함.

김용대 교수 랩에서는 Variational Inference를 사용하지 않고, Kernel Density Estimation을 이용해서 Density를 추정하였고, Kernel 계산이 dimention 저주에 걸리므로 z로 projection하여 구한 후에 다시 가져오는 형태, 그리고 계산하면 log안에 sum이 있는 형태가 되는데 이때는 EM알고리즘으로 풀 수 있고 잘되었다고 함.

최근에는 EM알고리즘을 Bayesian Neural Network으로 풀어서 uncertaionty가 포함되면 GAN보다는 훨씬 좋아질 것이라고 함. 김용대 교수님은 Generative 모델은 GAN보다는 VAE쪽을 마사지하는 것(분포를 모델링한 것에서 Generate)이 더 성능이 좋을 것이라는 확신이 있다고 함. 물론 최근 VAE와 GAN을 합치는 연구들이 많이 있다고 함.

---

### Day 4 Session 1: Policy Gradient Methods for Reinforcement Learning

- Kaist 김기응 교수

일반적인 supervised learning은 loss를 정의하고 정해진 학습데이터에 대해서 loss를 최소화한다.

그런데, RL의 특징은 loss의 function을 모른다. 미분할 수 없고, 학습데이터가 action에 따라 다르게 형성된다. 

loss를 너무 최적화 해도 안된다. explore를 적당히 해야 다른 경로도 찾을 수 있다. 

RL문제는 sequential decision problem이다. 

구성은 state가 있는 environment states S, action set A, reward function r(s, a), probabilistic transition function T = Pr(s'|s, a), distcount rate γ(미래에 대한 reward도 고려, 더 먼 미래일 수로 작음)

![markov decision process reinforcement learningì ëí ì´ë¯¸ì§ ê²ìê²°ê³¼](https://www.researchgate.net/profile/Leliane_Barros/publication/220812955/figure/fig1/AS:305679391838208@1449891054186/A-Markov-decision-process-Square-nodes-represent-the-action-circles-represent-the.png) 

결국, reward를 최대화하기 위한 action을 구하는 것이 목적임

RL에서 모든 것이 주어지면 아주 쉽게 구할 수 있지만 대부분 Transition Function이 정해지지 않아 학습해야 함

 **Value Prediction**

policy만 모를 때 policy를 가정하고 주어진 action, state들을 대입시켜 n차 방정식을 구한다. stage가 n개면 n차 방정식, State-value function을 구하거나, action-value function을 구한다.

state-value: 해당 state에 대한 value function(V)을 말하고, action-value : 해당 action을 했을 때의 value를 의미하고 Q function이라고 한다. 

그러나 reward와 transition을 몰라 연립방정식 푸는게 불가능

 **Policy Optimization**

retrun(reward)를 최대화하기 위한 optimal policy를 찾는다.



policy를 최적화하는 방법은 3가지가 있다.

**Model-based algoritm**

데이터로부터 Trasion function과 value function을 학습하여 policy를 찾는다.

st, at, rt, st+1을 알고 이것을 통해 reward function, transition function, Q function을 찾는다.

**Model-free(or Value-based) algorithm**

바로 Q function을 찾아서 policy를 찾는다.

st, at, rt, st+1이 주어지면 이것을 통해 Q function을 업데이터 하며 Q를 찾는다. 수식(page 390)이 gradient descent로 weight를 업데이트하는 수식과 비슷함(미분형태 Q(t+1) - Q(t) ) 

DQN 계열이 이 방식이다.

action이 finite한 경우 Q만 잘 찾으면 각 action에 대한 Q를 가지고 있으면 되기 때문

DQN은 Q를 neural network로 표현, 입력으로 s, a를 넣지 않고 s만 넣어서 따로 학습, 시간절약

DQN의 replay memory는 explore를 위한 것

stage와 action이 finite한 경우 DQN 좋음, continuous하면 너무 많은 a에 대한 Q가 필요해 softmax하는 것도 애매한 상황이 됨

**Policy search algorithm**

바로 policy를 찾는다.

장점: neural network로 학습한다. high demensional하거나 continuous한 action의 경우 DQN으로 힘들다. policy를 직접 학습해야함

stochastic한 policy를 얻을 수 있다. 각 action에 대한 확률

그러나, high variance 문제(deep neural network의 문제와 동일)가 존재, 오래전부터 있던 방식인데 Deep Neural Network이 어느정도 가능해지면서 RL도 해결

End to End 학습이 가능함, s,a로부터 policy 파라미터 θ를 학습. policy를 미분하면 일반 DNN과 학습 동일

최근에는 두가지를 조합하는 방식이 좋은 성능을 보인다. Actor-Critic이 두가지 조합

Policy는 **softmax policy**가 있는데 discrete action의 경우에 해당하고 gradient 수식 존재(page 392)

continuous한 경우 policy를 **Gaussian Policy**로 가정하고 구한다. 역시 미분 수식이 이미 다 풀어져 있다.(**policy gradient**)

Deep RL은 policy를 찾기 위한 feature를 convolution net이나 deep neural network로 구성하는 것이다.

**Policy Gradient**(page 393)

enviroment와 dependent한 p를 구해야하는데 잘 풀면 이부분이 제거된다. 가장 basic한 버전 1은 N개마다 평균적으로 gradient를 구하는 것인데 variance가 너무 커서 학습이 거의 안된다. 대부분 이 variance를 줄이기 위한 바법들을 고안한다. 

버전 2는 기존 policy gradient 수식의 Reward function을 Q function으로 대체하는데 이것은 처음부터 현재까지 더하는 게 아니라 현재와 미래의 reward가 계산된 것만 고려하는 의미가 된다. 이렇게 하면 최종 기대값은 달라지지 않는다는 것이 증명되어있다. 단, local값은 달라져서 variace를 줄여주는 역할을 한다.

그래도 여전히 크다.

**Reducing variance via Baseline**

policy gradient의 reward에 상수값을 빼준다. 상수값을 빼준다. 이것은 gradient의 기대값에 아무런 영향을 주지 않는다. b 값을 얼마로 줄지는 계산이 가능한데 계산이 복잡하여 특정 상수값을 사용한다. 그런데 이것도 잘 사용하지 않고, b(st) 형태로 state마다 다르게 적용시킨다. 결국 b(st)는 value function이라고 치환하면 Advantage Function(기존 방식이라고 함)과 동치가 된다. 

이는 외부영향이 없으면 policy가 말하는 데로, 외부 영향이 있으면 다른 것을 쓸것인데 그것에 대한 loss를 구하는 것: advantage function (Regularization과 비슷)

**TRPO (Trust Region Policy Optimization)**

일반적인 supervised learning에서는 step size가 데이터가 많아지면 별로 중요해 지지 않는데,

그런데 RL에서는 잘 못 설정된 step size는 계속 영향을 받게 되고 중첩될 수 있다. sequential하게 데이터가 바뀌므로, RL에서는 학습데이터가 고정되어 있지 않기 때문에

surrogate objective. 다른 것을 대체하여 그것으로 업데이트한다. 이것은 기존 것의 lower bound (page 397)되고 Trust Region과 같아서 TRPO라고 부른다. TRPO는 loss에 새로운 텀을 추가하는 것이며 그것으로 기존 것의 lower bound가 된 것에서 옵티멈을 찾기 때문에 옵티멈 존재, 정해진 값보다 작을 때문 업데이트하는 구조이며 특정 상수는 정해주는데 별로 영향을 안받지만 0.05 정도를 정해준다. 

이것을 하려면 2차 미분하고 식이 복잡해진다. 이것을 1차미분 텀으로 만든 것이 PPO이다.

**PPO (Proximal Policy Optimization)**

PPO는 TRPO의 KL텀을 근사하여 1차 미분 텀으로 바꾸기 위해 if 조건을 추가하여 update한다(page398)

**Deterministic Policy Gradient (DPG)**

action이 discrete한 경우 softmax policy, continuous한경우 Gaussian policy를 사용 

그런데, stochastic한 output이 컨트롤에는 좋지 않다. 예) 왼쪽회전으로 52%  해라 ^^;

deterministic한 output을 주면 더 효율적인 컨트롤이되고, 식이 simple해진다. 

DPG에서는 critic은 value를 estimation하여 actor에 Q function을 넣어주고 actor는 이를 이용해 policy gradient를 계산한다. output은 deterministic한 action을 준다.

action에 대한 stochastic한 output을 deterministic하게 만드는 것은 variance를 줄여주면 어느 한점의 deterministic한 action값이 된다. 그런데 이렇게 줄이다 보면 (page 401) policy gradient가 +-로 급격히 증가한다. 그래서 잘 안될 것으로 생각했었는데,

이렇게 되면 policy gradient의 적분이 전체를 하지 않고도 Q function에서 +- e만큼만하면 적분이 쉬워짐

즉, 수식이 매우 간단해짐

그런데, 일반적으로 RL은 stochastic한 output으로 exploration할 수 있다. RL은 옵티멈에 수렴해도 항상 다른 길을 search해서 더 좋은 것을 찾을 수 있어야 함

여기서는 DQN의 replay memory처럼 update하는 policy와 stochasitc behavior policy를 따로 둔다. update되는 policy로 action을 하지 않고 critic으로부터 얻은 Q function으로 behavior policy로 사용한다. 결국, 나중에는 critic의 Q function이나 학습된 policy가 거의 같아지지만 exploration을 위해 이렇게 한다.

**Deep Deterministic Policy Gradient(DDPG)**

DPG를 deep neural network로 구성한 것

** 참고로 discret한 경우 feature만 잘 찾으면 CNN 같은 걸로 policy를 한번에 찾을 수 있다.

예제들과 A3C와 같이 async하게 여러 개의 agent를 학습하는 프레임워크들 제공하는 link 참고(page403)

---

### Day 3 Session 3: Deep Learning

- Kaist 김준모 교수

** 엄태웅, 파킨슨병 논문에서 imbalance한 문제와 subject간의 차이가 큰 것을를 해결하기 위해 subject 그룹별 모델을 만들고 ensemble하였더니 좋아졌다고 함

**Rectified Linear Unit(ReLu)**

ReLu를 쓰면 vanising gradient도 해결되고 그래서 더 깊은 network가 가능. 아예 사용안 하면 될 것 같지만 그렇게 하면 non-linear가 안됨. ReLu는 piece-wise-tile과 같은 역할을 한다. 까가운 것은 같은 평면에 속하고 다른 것들은 다른 평면에 속할 수 있다. deep 하게 layer를 쌓으면 이 평면이 더 작아져서 아주 detail하게 학습가능하다. 

ReLu를 사용하면 pre-training이 거의 필요없게 된다. 

**Dropout**

compexity가 높아지면 overfit우려가 있다. 일반적으로 overfit을 없애려면 ensemble을 시켜야 한다. dropout은 ensemble하는 것과 거의 같다. 

특정 ace 뉴런이 학습을 잘하면 다른 뉴런들은 학습하지 않아도 loss는 작다. 비효율적인 뉴런들이 있는데 이것들이 더해지면 inference할 때 방해가 될 수 있다. 즉, ace가 있으면 random하게 쉬게하고 쉬는 동안은 다른 뉴런들이 학습하게 한다. 

**Local minima of DNN**

예전에는 2D에서 dimension이 높아지면 local minima가 많아진다고 생각함. 그런데 차원이 높아지면 높은 곳에서 모든 차원에서 감소하는 local minima가 존재할 확률이 작아짐(2^n승), saddle point는 옵티마이져가 벗어날 수 있음. 결국, 낮은 곳에서만 local minima가 존재하고 어느 하나만 찾아도 글로벌 옴티멈과 비슷비스함. 고차원에서는 그렇게 됨

CNN 자세히 설명, CNN은 FC보다 weight의 메모리를 많이 줄일수 있다. CNN에서 첫번째 layer는 edge를 학습하는데 실제 포유류 수정체? 렌즈가 그렇다고 함

이미지넷 DB 때문에 이미지쪽 분야가 큰 성과를 거두었다.

필터는 3x3을 사용하면 5x5, 7x7을 쓰는 효과를 준다.

구글은 network속에 network를 넣은 **inception** 구조를 9번 반복시킴으로써 성능을 개선하였다. 3x3과 5x5 필터 결과를 concat하고 메모리줄이기 위해 1x1 conv를 사용한다.

**VGG**는 3x3, 19layer만으로도 좋은 결과를 냈다. conv를 두번연속하고 maxpooling한다.

 **ResNet**은 layer를 계속 늘리면 너무 늘어나면 성능이 떨어지는 것을 확인, 그런데 성능이 유지는 되도 떨어지면 안되야한다. 그래서 확인해 보니 weight가 diagonal하게 1인 고차원 matrix를 학습하는 것이 매우 잘 안되기 때문을 알아내서 미리 이전 것을 copy하고 더해주어 weight가 '0'을 만들어서 기존 것을 유지하게 한다. diagonal이 전부 1인 것보다는 weight 메트릭스가 전부 '0'인 것을 더 잘 찾는다. 이렇게 152 layer까지 쌓는게 가능하고  사람 수준을 넘어서는 성능을 냄

**Attention**: feature를 selection 할 수 있는 actention을 학습하면 더 좋아질 수 있다. 

**Continual Learning**

한번 배운 network에 또 다른 것을 학습시키면 기존에 배운 것을 잊게 된다. task를 잊지않고 복습없이(기존 데이터 사용 안하고) 다른 것을 학습하는 것이 목적

domain adaptation은 새로운 데이터가 label이 없을 때 데이터만으로 정보를 배우는 것이다. label을 알고 있는 데이터를 모르는 데이터로 converse하여 classification하는 것이다.

**Network Minmizing**

hintton의 distilation은 부모 network의 soft label로 simple한 네트워크를 학습 시킬 수 있고, dark knowledg를 학습할 수 있다고 함

김준모 교수팀은 output이 아닌 부모의 network를 모방해서 줄여서 사용하는 것을 하였고 성능이 hintton것 보다 좋았다고 함

**Adversarial Attack**

사람이 보기는 것과 완전 다른 판단을 하게 하는 이미지를 생성. 머신이 엉뚱한 답을 하는 입력들.

**(my idea)** *우리는 GAN에서 일반적으로 Generator를 주로 사용하여 augmentation으로 쓴다. GAN에서 discriminater를 사용하는 아이디어로써, 인증에서의 adversarial attack을 막아줄 GAN의 discriminater를 학습 시키는 것이다.*



# Machine Learning & Stochastic Study

- ensemble하면 좋아지는 이유는

  single 모델간에는 random에 의한 오차 e가 있고, 이는 가우시안 분포를 따른 다는 가정이 있으면, 이를 모아서 평균하면, 오차 e가 줄어든게 된다.

  오차의 분포가 가우시안이 아니라면 평균이 아닌 다른 연산을 통해 모델을 ensemble해야한다. 이 분포의 오차를 줄이는 방법을 Gradient Descent로 찾을 수도 있다. Gradient Boosting 등이 있을 수 있고, 이런 방법은 오차 e의 분포는 모르지만 ensemble하여 최소가되는 연산을 찾아낼 수 있다. 분포가 가우시안이면 평균이 바로 그 최적 연산이 된다.