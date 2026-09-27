# Lecture 1 — Flow Matching / Rectified Flow
## 장기 복습용 Remind Note

> 목표: 슬라이드 내용을 그대로 요약하는 것이 아니라, **강의 흐름 + 공식 정의 + 내가 실제로 헷갈렸던 지점 + 질문하면서 이해가 정리된 방식**을 한 번에 다시 떠올릴 수 있게 정리한다.

---

# 0. 이 강의는 대체 무엇을 하려는 강의인가?

이 강의의 큰 질문은 다음과 같다.

> **새로운 데이터를 생성하려면, 어떻게 하면 쉬운 분포에서 실제 데이터 분포로 샘플을 옮길 수 있는가?**

이미지 한 장을 고차원 벡터 \(x \in \mathbb{R}^d\)라고 생각하면, 이미지 한 장은 고차원 공간의 점 하나가 된다.

실제 이미지 데이터셋에는 이런 점들이 많이 모여 있고, 이 점들이 이루는 분포를 target/data distribution이라고 생각한다.

문제는:

- 실제 데이터 샘플들은 가지고 있음
- 하지만 실제 데이터 분포 \(p_{\text{data}}(x)\)의 수식은 모름
- 따라서 그 분포에서 "새로운 이미지"를 직접 샘플링하는 방법도 모름

반면 Gaussian noise처럼 단순한 분포는 샘플링하기 쉽다.

그래서 생성 문제를 다음처럼 바꿔 생각한다.

\[
\text{easy-to-sample source distribution}
\longrightarrow
\text{target data distribution}
\]

즉,

\[
X_0 \sim p
\]

에서 시작해서 어떤 변환 \(\Phi\)를 통해

\[
X_1=\Phi(X_0), \qquad X_1 \sim q
\]

가 되도록 만들고 싶다.

---

# 1. 6장 — The Generative Modeling Problem

## 핵심

- \(p\): source distribution
- \(q\): target/data distribution
- \(X_0 \sim p\)
- \(X_1 \sim q\)
- \(\Phi\): source를 target으로 보내는 변환

### 내가 처음 헷갈렸던 부분

> “왜 굳이 source에서 target으로 옮긴다고 생각하지? 그냥 이미지를 만들면 되는 거 아닌가?”

결국 둘은 같은 말이다.

생성 함수를

\[
z \mapsto x
\]

라고 생각하면, random input \(z\) 전체의 분포는 source distribution이고, 출력 \(x\) 전체의 분포는 target distribution이다.

즉

\[
\text{random input} \rightarrow \text{image}
\]

를 분포 전체 관점에서 보면

\[
\text{source distribution} \rightarrow \text{target distribution}
\]

이 된다.

---

# 2. 7–11장 — Model / Flow / Velocity

## 2.1 한 번에 옮기지 않고 시간 \(t\)를 둔다

전체 변환을 한 번에 하지 않고,

\[
X_0 \rightarrow X_{0.1} \rightarrow X_{0.2} \rightarrow \cdots \rightarrow X_1
\]

처럼 연속적인 이동으로 본다.

여기서

\[
t\in[0,1]
\]

은 실제 시간이 아니라 **생성 과정의 진행도**다.

- \(t=0\): source
- \(t=1\): target
- \(0<t<1\): 중간 상태

## 2.2 \(\psi_t(x_0)\)의 의미

\[
\psi_t(x_0)
\]

는:

> 초기값 \(x_0\)를 flow를 따라 \(t\)만큼 진행시켰을 때의 현재 위치

이다.

이미지 문제라면 이것도 이미지와 같은 차원의 벡터다.

\[
\psi_t(x_0)=X_t
\]

로 생각해도 된다.

중요:

- \(x_0\)는 하나의 trajectory를 볼 때 고정된 초기조건
- \(t\)만 변한다
- 따라서 \(x_0\)는 \(t\)로 미분할 때 상수 취급

예:

\[
t=0 \Rightarrow \psi_0(x_0)=x_0
\]

\[
t=0.5 \Rightarrow \psi_{0.5}(x_0)=x_{0.5}
\]

\[
t=1 \Rightarrow \psi_1(x_0)=x_1
\]

## 2.3 Velocity와 ODE

위치가 시간에 따라 변하면 그 변화율이 velocity다.

\[
\frac{d}{dt}\psi_t(x_0)
\]

가 바로 현재 trajectory의 순간 velocity.

강의의 ODE 표현:

\[
\frac{d}{dt}\psi_t(x_0)
=
u_t(\psi_t(x_0))
\]

의미:

> 현재 위치 \(\psi_t(x_0)\)에서의 velocity를 \(u_t\)가 알려준다.

ODE의 일반형:

\[
\frac{dx(t)}{dt}=f(x(t),t)
\]

즉,

- 현재 상태 \(x(t)\)
- 현재 진행도 \(t\)

를 넣으면 현재 변화율을 출력한다.

Flow Matching에서는:

\[
u_\theta(x,t)
\]

가 바로 그 velocity field를 근사하는 neural network다.

### 내가 이해한 표현

> **“위치 \(x\)와 진행도 \(t\)를 주면 속도를 내는 함수”**

---

# 3. 12–15장 — Velocity가 분포를 만든다 / 가장 단순한 Flow Matching

## 3.1 한 점이 아니라 분포 전체를 생각하기

\[
X_t=\psi_t(X_0)
\]

는 샘플 하나의 이동이고,

\[
X_t\sim p_t
\]

는 많은 샘플들을 이동시켰을 때 시간 \(t\)에서 이루는 전체 분포다.

즉 velocity field \(u_t\)를 따라 source의 모든 점을 움직이면 \(p_t\)라는 중간 분포가 만들어진다.

## 3.2 Simplest Flow Matching

source에서 \(x_0\), target에서 \(x_1\)을 하나씩 뽑아 pair를 만든다.

그리고 둘 사이를 직선으로 연결한다.

\[
x_t=(1-t)x_0+t x_1
\]

이 경로를 \(t\)로 미분하면:

\[
\frac{dx_t}{dt}=x_1-x_0
\]

따라서 이 pair에 대한 정답 velocity는:

\[
u(x_t,t)=x_1-x_0
\]

학습 데이터 한 건은 사실상:

\[
(x_t,t)
\rightarrow
(x_1-x_0)
\]

이다.

모델은:

\[
u_\theta(x_t,t)
\]

를 출력하고,

\[
u_\theta(x_t,t)\approx x_1-x_0
\]

가 되도록 MSE로 학습한다.

## 3.3 내가 여기서 크게 헷갈린 부분

### 질문 1
> \(x_t,t\)만 주면 \(x_0,x_1\)을 복원할 수 있나?

아니다.

\[
x_t=(1-t)x_0+t x_1
\]

식 하나만으로는 \(x_0,x_1\) 둘을 유일하게 알 수 없다.

모델은 \(x_0,x_1\)을 역산하는 게 아니라,

\[
(x_t,t)\rightarrow velocity
\]

라는 함수를 직접 학습한다.

### 질문 2
> 특정 pair에서는 velocity \(x_1-x_0\)가 \(t\)와 무관한데, 왜 \(t\)를 넣나?

개별 pair만 보면 그렇다.

하지만 모델은 pair ID를 입력받지 않고 수많은 pair를 하나의 공통 함수로 학습한다.

따라서 현재 \(x\)가 생성 과정의 어느 진행도에 있는지 알려주는 \(t\)가 필요하다.

---

# 4. 16–20장 — Conditional Flow / Marginalization Trick

여기서 강의 난이도가 급격히 올라간다.

## 4.1 왜 conditional flow가 필요한가?

서로 다른 pair들이 같은 \((x,t)\) 근처를 지나면서 서로 다른 velocity를 요구할 수 있다.

예:

- 어떤 pair는 오른쪽 위
- 다른 pair는 오른쪽 아래

그러나 하나의 neural network \(u_\theta(x,t)\)는 같은 입력에 하나의 velocity만 출력할 수 있다.

## 4.2 \(u_t(x\mid x_1)\)

특정 target sample \(x_1\)을 고정했을 때:

\[
u_t(x\mid x_1)
\]

는

> 현재 위치 \(x\), 시간 \(t\)에서 최종 target이 \(x_1\)이라고 가정했을 때의 velocity

이다.

## 4.3 \(p_t(x\mid x_1)\)

\[
p_t(x\mid x_1)
\]

는:

> 최종 target이 \(x_1\)이라고 조건을 걸었을 때, 시간 \(t\)에서 현재 데이터가 \(x\)일 확률밀도

이다.

반면:

\[
p_t(x)
\]

는 모든 target을 섞어 본 전체 중간 분포다.

## 4.4 Marginal velocity

전체 velocity는 conditional velocity들의 단순 동일평균이 아니라, 현재 \(x,t\)에서 각 target이 얼마나 가능한지를 반영한 확률 가중평균이다.

\[
u_t(x)
=
\mathbb E[
 u_t(X_t\mid X_1)
\mid X_t=x
]
\]

내가 이해한 표현:

> **현재 \(x,t\)에 도달했을 가능성이 높은 target-conditioned flow에 더 높은 가중치를 주어 평균낸 velocity**

## 4.5 왜 이게 중요한가?

Flow Matching이 개별 pair의 target을 정확히 보존하는 것은 아니다.

보존하려는 것은 **전체 target distribution**이다.

즉

\[
Z_0^{(i)}\to X_1^{(i)}
\]

를 반드시 만족할 필요는 없고,

최종적으로

\[
Z_1\sim p_{\text{target}}
\]

이면 된다.

## 4.6 Flow Matching Loss vs Conditional Flow Matching Loss

원래 Flow Matching loss:

\[
\mathcal L_{FM}(\theta)
=
\mathbb E
\left[
\|u_t^\theta(X_t)-u_t(X_t)\|^2
\right]
\]

하지만 marginal velocity \(u_t(X_t)\)는 직접 구하기 어렵다.

그래서 실제로는 conditional velocity를 정답으로 사용:

\[
\mathcal L_{CFM}(\theta)
=
\mathbb E
\left[
\|u_t^\theta(X_t)-u_t(X_t\mid X_1)\|^2
\right]
\]

직선 interpolation에서는:

\[
u_t(X_t\mid X_1)=X_1-X_0
\]

가 된다.

중요한 점:

모델 출력은

\[
u_t^\theta(X_t)
\]

이지

\[
u_t^\theta(X_t\mid X_1)
\]

가 아니다.

왜냐하면 \(X_1\)은 학습 때 정답 velocity를 만드는 supervision 정보이지, inference 때 알고 있는 값이 아니기 때문이다.

---

# 5. 23–27장 — Conditioning

## 5.1 Conditioning이란?

지금까지는 아무 target data나 생성했다.

실제 생성에서는:

- Husky
- Corgi
- 특정 text prompt
- 저해상도 이미지
- 특정 구조 정보

같은 조건을 반영하고 싶다.

그래서 label/condition \(Y\)를 추가한다.

## 5.2 25장 — Conditional probability / velocity

\[
p_t(x\mid y)
\]

는:

> 조건 \(y\)를 만족하는 target들만 봤을 때 시간 \(t\)에서의 중간 분포

\[
u_t(x\mid y)
\]

는:

> 조건 \(y\)를 만족하는 target distribution으로 가기 위한 marginal velocity

## 5.3 26장 — Conditional Flow Matching Loss

장표의 모델은:

\[
u_t^\theta(X_t\mid Y)
\]

이다.

즉 neural network는:

- \(t\)
- 현재 데이터 \(x\)
- condition \(y\)

를 입력받아 velocity를 출력한다.

장표의 함수 형식:

\[
u_t^\theta(x\mid y):
[0,1]\times\mathbb R^d\times\mathbb R^k
\rightarrow \mathbb R^d
\]

## 5.4 \(\pi_{0,1,Y}\)

\[
(X_0,X_1,Y)\sim \pi_{0,1,Y}
\]

에서 \(\pi_{0,1,Y}\)는 특별한 종류의 분포 이름이 아니다.

\[
(X_0,X_1,Y)
\]

라는 training tuple의 **joint / coupling distribution**을 나타내는 notation이다.

특히 \(X_1\)과 \(Y\)는 독립이 아니다.

예:

- \(X_1\): 실제 Husky 이미지
- \(Y\): Husky label

## 5.5 condition의 종류는 서로 달라도 됨

27장 예시:

- class condition
- inverse problem
- text-to-image

예를 들어 super-resolution에서는:

\[
Y=\text{저해상도 이미지}
\]

\[
X_1=\text{고해상도 이미지}
\]

즉 condition은 출력 자체가 아니라 **어떤 출력이 맞는지를 제한하는 정보**다.

주의:

class label과 저해상도 이미지가 반드시 동일한 encoder/동일한 구현으로 들어가는 것은 아니다.

수학적으로는 둘 다 \(Y\)라고 추상화할 수 있지만 실제 architecture는 다를 수 있다.

---

# 6. 28–30장 — Guidance

이 부분은 처음에 가장 크게 헷갈렸던 부분 중 하나.

## 6.1 Conditioning과 Guidance는 다름

### Conditioning
> 무엇을 만들 것인가?

예:

\[
y=\text{Husky}
\]

### Guidance
> 그 조건을 얼마나 강하게 반영할 것인가?

즉:

\[
\boxed{Y=\text{조건의 내용}}
\]

\[
\boxed{w=\text{조건의 영향력}}
\]

## 6.2 왜 Guidance가 필요한가?

conditional model이 완벽하게 학습되어도

\[
p(x\mid y)
\]

는 하나의 넓은 분포다.

예를 들어 Husky라는 조건 안에도:

- 허스키가 작게 있는 이미지
- 배경이 많은 이미지
- 애매한 허스키
- 전형적인 허스키

등 다양한 sample이 존재할 수 있다.

따라서 inference/sampling 때:

> 다양성을 조금 희생하더라도 prompt/condition을 더 강하게 따르게 하고 싶다

는 요구가 생긴다.

Guidance는 **모델을 재학습하지 않고 sampling 시점에서 condition strength를 조절**한다.

## 6.3 29장 — Classifier Guidance for Flow Matching

\[
\tilde u_t(x\mid y)
=
u_t(x)
+
w\,b_t\nabla_x \log p_t(y\mid x)
\]

각 항:

- \(u_t(x)\): 원래 unconditional velocity
- \(p_t(y\mid x)\): 현재 \(x\)가 조건 \(y\)일 확률
- \(\nabla_x \log p_t(y\mid x)\): \(x\)를 어느 방향으로 수정하면 \(y\)일 확률이 더 커지는지
- \(w\): guidance strength
- \(b_t\): 시간에 따른 scale
- \(\tilde u_t\): 실제 sampling에 사용할 최종 velocity

내가 이해한 표현:

> **원래 생성 방향 + 조건을 더 잘 만족시키는 보정 방향**

## 6.4 Classifier-Free Guidance

\[
\tilde u_t(x\mid y)
=
(1-w)u_t(x)+w\,u_t(x\mid y)
\]

별도 classifier gradient 없이:

- unconditional velocity
- conditional velocity

를 섞는다.

---

# 7. 31–39장 — Rectified Flow / Reflow

## 7.1 왜 straight path가 중요한가?

학습된 velocity model로 실제 생성할 때는 ODE를 따라가야 한다.

\[
\frac{dx}{dt}=u_\theta(x,t)
\]

컴퓨터는 이를 한 번에 정확히 풀 수 없으므로 수치적으로 step을 밟는다.

예:

\[
t: 0\rightarrow0.25\rightarrow0.5\rightarrow0.75\rightarrow1
\]

각 step에서:

\[
x_{t+\Delta t}
\approx
x_t+\Delta t\,u_\theta(x_t,t)
\]

이런 계산을 수행하는 것이 ODE solver다.

경로가 많이 휘어 있으면 작은 step이 많이 필요하다.

경로가 직선에 가까우면 큰 step, 심하면 1-step으로도 갈 수 있다.

따라서:

\[
\boxed{\text{straighter flow} \Rightarrow \text{faster sampling}}
\]

## 7.2 왜 직선 pair로 학습했는데 learned flow는 직선이 아닐까?

학습용 pair:

\[
X_t=(1-t)X_0+tX_1
\]

는 직선이다.

하지만 서로 다른 pair의 직선들이 crossing할 수 있다.

같은 \((x,t)\)에서 서로 다른 velocity를 요구하면 하나의 neural network는 둘을 동시에 만족시킬 수 없다.

그래서 실제 learned velocity field는 여러 요구를 종합한 방향을 내고, 실제 trajectory는 휘어진다.

## 7.3 두 종류의 경로를 구분

### \(X_t\)
학습용으로 사람이 만든 직선 interpolation

\[
X_t=(1-t)X_0+tX_1
\]

### \(Z_t\)
학습된 velocity field를 실제로 따라갔을 때의 trajectory

\[
\frac{dZ_t}{dt}=v(Z_t,t)
\]

따라서:

\[
\boxed{X_t=\text{training path}}
\]

\[
\boxed{Z_t=\text{learned model이 실제로 만드는 path}}
\]

## 7.4 Crossing에 대한 정리

처음 random pair의 직선 \(X_t\)는 crossing할 수 있다.

하지만 실제 learned vector field는 같은 \((x,t)\)에서 하나의 velocity만 정의하므로, trajectory들은 원래 pair의 crossing을 그대로 재현하지 못한다.

그 결과:

- 개별 원래 pairing은 깨질 수 있음
- learned path는 휘어질 수 있음
- 전체 target marginal distribution은 보존되는 것이 목표

## 7.5 Reflow

처음에는 random pair:

\[
(X_0,X_1)
\]

로 학습한다.

학습이 끝난 뒤 source sample \(Z_0\)에서 시작해 실제 flow를 따라 sampling:

\[
Z_0\rightarrow Z_1
\]

하면 새로운 source-target correspondence가 생긴다.

이 \((Z_0,Z_1)\)를 새 pair로 사용한다.

그리고 다시 직선:

\[
Z_t=(1-t)Z_0+tZ_1
\]

을 만들어 Flow Matching을 재학습한다.

즉:

\[
\boxed{
\text{Reflow}
=
\text{기존 learned flow로 새 pair 생성}
\rightarrow
\text{새 pair로 다시 Flow Matching}
}
\]

내가 이해한 표현:

> **처음의 랜덤 pairing을, 모델이 실제로 만들어낸 더 자연스러운 pairing으로 교체한다.**

---

# 8. 40–42장 — Reflow의 성질

## 8.1 Transport cost 감소

crossing된 pair를 rewiring하면 전체 source-target 이동거리가 줄어드는 경향이 있다.

즉 더 비효율적인 X자 연결보다 자연스러운 pairing을 사용한다.

## 8.2 Straightness 증가

Reflow를 반복하면 learned trajectory가 점점 더 straight해진다.

목적:

\[
\boxed{\text{sampling step 감소}}
\]

## 8.3 Marginal preserving

pair는 바뀌어도 target distribution은 유지되어야 한다.

\[
p(Z_1)=p(X_1)
\]

즉:

> 개별 source-target 대응관계는 바뀌어도 괜찮지만 전체 target marginal은 같아야 한다.

---

# 9. 43–45장 — FLUX

Flow Matching 계열이 실제 image generation/editing 모델에 사용되는 예시.

예:

- instruction-driven image editing
- scene modification
- style/detail preservation
- object removal / replacement
- product detail extraction

핵심:

\[
\boxed{\text{Flow Matching은 생성뿐 아니라 image editing에도 활용 가능}}
\]

---

# 10. 46–53장 — Reflow의 문제: Distribution Drift

## 10.1 문제

Reflow에서는 이전 모델이 만든 pair:

\[
(Z_0,Z_1)
\]

를 다시 학습에 사용한다.

그런데 \(Z_1\)은 실제 target \(X_1\)이 아니라 **generated/fake sample**이다.

기존 모델이 완벽하지 않다면:

\[
p(Z_1)\neq p(X_1)
\]

일 수 있다.

이 fake sample을 다시 정답처럼 학습하면 다음 모델은 실제 target보다 이전 모델의 generated distribution에 더 맞춰질 수 있다.

이 현상이:

\[
\boxed{\text{distribution drift}}
\]

## 10.2 Reflow step이 늘어나면

실험상 Reflow 횟수가 증가하면서 generated distribution이 real target에서 멀어지고 KL divergence가 증가할 수 있다.

즉:

\[
\text{Reflow 반복}
\Rightarrow
\text{straightness 증가}
\]

와 동시에

\[
\text{target drift 증가 가능}
\]

이라는 trade-off가 생긴다.

---

# 11. 51–53장 — Fake Pair / Real Pair / Conic Reflow

## 11.1 Fake Pair

기존 Reflow:

\[
Z_{0,F}\sim N(0,I)
\]

\[
Z_{1,F}=V(Z_{0,F})
\]

pair:

\[
(Z_{0,F},Z_{1,F})
\]

여기서 \(Z_{1,F}\)는 generated sample.

## 11.2 Real Pair

실제 target image:

\[
X_1
\]

에서 flow를 역방향으로 따라 source-side point를 얻는다.

\[
Z_{0,R}=V^{-1}(X_1)
\]

따라서 real pair:

\[
(Z_{0,R},X_1)
\]

여기서 \(R\)은 real-pair side를 뜻한다고 이해하면 된다.

즉:

> \(Z_{0,R}\)는 실제 target \(X_1\)을 역추적해서 찾은 source-side 대응점.

## 11.3 Conic Reflow

단순히

\[
Z_{0,R}\leftrightarrow X_1
\]

한 줄만 사용하지 않고, \(Z_{0,R}\) 주변을 random perturbation해서 여러 source-side point를 만든다.

\[
Z_{0,R}'
=
\operatorname{slerp}(Z_{0,R},\epsilon,\zeta)
\]

그리고:

\[
Z_{0,R}'
\rightarrow X_1
\]

경로들을 학습한다.

여러 perturbed source points가 하나의 real target으로 향하는 모양이 cone과 비슷해서 Conic Reflow라고 부른다.

핵심 목적:

\[
\boxed{\text{fake target만 사용하지 말고 real target }X_1\text{로 다시 anchor}}
\]

해서 distribution drift를 줄이는 것.

---

# 12. 54–56장 — Straightness 측정

## 12.1 Curvature

전체 trajectory가 얼마나 직선에서 벗어났는지 측정.

\[
S(Z)
=
\int_0^1
\mathbb E
\left[
\|(Z_1-Z_0)-\dot Z_t\|^2
\right]dt
\]

작을수록 straight.

## 12.2 IVD — Initial Velocity Delta

시작점에서의 velocity가 최종 endpoint 방향과 얼마나 다른지 측정.

\[
IVD(Z,t_0)
=
\mathbb E
\left[
\|(Z_1-Z_0)-\dot Z_{t_0}\|^2
\right]
\]

왜 중요?

1-step generation에서는 거의 초기 velocity가 전체 결과를 결정하기 때문.

즉:

- Curvature: 전체 경로의 휘어짐
- IVD: 시작 방향의 오차

---

# 13. 내가 계속 헷갈렸던 핵심 포인트 모음

## 13.1 \(x_t\)와 \(t\)

- \(t\): 실제 시간이 아니라 생성 진행도
- \(x_t\): 그 진행도에서의 현재 데이터 상태

## 13.2 \(\psi_t(x_0)\)

초기값 \(x_0\)를 고정했을 때 \(t\)에서의 위치.

\[
\psi_t(x_0)=X_t
\]

## 13.3 Velocity

\[
u_t(x)
\]

는:

> 현재 위치 \(x\), 현재 진행도 \(t\)에서 어디로 얼마나 움직일지

## 13.4 모델이 배우는 것

모델은 target \(x_1\)을 직접 입력받아 찾는 게 아니다.

\[
(x_t,t)\rightarrow velocity
\]

를 배운다.

conditional model이면:

\[
(x_t,t,y)\rightarrow velocity
\]

## 13.5 Sampling

sampling은:

> 학습이 끝난 모델을 사용해 실제 output sample을 생성하는 과정

Flow Matching에서는:

\[
X_0\sim p_{\text{source}}
\]

에서 시작해 ODE solver로 velocity를 계속 따라가 최종 \(X_1\)을 얻는다.

## 13.6 Step

step은 \(t\) 자체가 아니라:

> \(t=0\rightarrow1\) 구간을 ODE solver가 몇 번의 numerical update로 나누어 이동하는가

1-step:

\[
0\rightarrow1
\]

4-step:

\[
0\rightarrow0.25\rightarrow0.5\rightarrow0.75\rightarrow1
\]

## 13.7 Conditioning vs Guidance

Conditioning:

> 무엇을 만들 것인가

Guidance:

> 그 condition을 sampling 때 얼마나 강하게 반영할 것인가

## 13.8 Flow Matching vs Rectified Flow

Flow Matching:

> source→target velocity field를 학습

Rectified Flow / Reflow:

> learned flow가 너무 휘는 문제를 줄이기 위해 pairing을 재구성하고 다시 학습

---

# 14. 전체 강의 흐름을 한 번에 보기

```text
[Generative Modeling Problem]
easy source distribution
        ↓
target data distribution

        ↓

[Flow]
X0 → Xt → X1
continuous path

        ↓

[Velocity]
(x, t) → velocity

        ↓

[Simplest Flow Matching]
Xt = (1-t)X0 + tX1
target velocity = X1 - X0

        ↓

[Conditional / Marginalization]
개별 target별 conditional velocity
        ↓ probability-weighted average
marginal velocity

        ↓

[Conditional Generation]
(x, t, y) → velocity

        ↓

[Guidance]
sampling 시 condition strength 조절

        ↓

[Problem]
random pair들의 straight path가 crossing
        ↓
learned actual trajectory는 curved

        ↓

[Rectified Flow / Reflow]
learned flow가 만든 새 pairing으로 재학습
        ↓
straighter flow
        ↓
fewer ODE steps / faster sampling

        ↓

[Problem]
generated fake pair만 재사용
        ↓
distribution drift

        ↓

[Conic Reflow]
real target을 다시 supervision에 사용
        ↓
drift 완화

        ↓

[Evaluation]
Curvature
IVD
FID / Precision / Recall
```

---

# 15. 최종 한 문장

> **Flow Matching은 source distribution에서 target distribution으로 가는 velocity field를 학습하는 생성 방법이고, Rectified Flow/Reflow는 그 learned trajectory를 더 straight하게 만들어 sampling을 빠르게 하려는 방법이다. 다만 Reflow에서 generated sample만 다시 학습하면 target distribution drift가 생길 수 있어 real pair를 활용하는 보정이 필요하다.**

---

# 16. 복습할 때 최소한 기억해야 할 식

### Flow / ODE

\[
\frac{dX_t}{dt}=u_t(X_t)
\]

### Linear interpolation

\[
X_t=(1-t)X_0+tX_1
\]

### Conditional target velocity

\[
u_t(X_t\mid X_1)=X_1-X_0
\]

### Marginal velocity

\[
u_t(x)
=
\mathbb E[
 u_t(X_t\mid X_1)
\mid X_t=x
]
\]

### Conditional model

\[
u_t^\theta(x\mid y)
\]

### Classifier-Free Guidance

\[
\tilde u_t(x\mid y)
=
(1-w)u_t(x)+w\,u_t(x\mid y)
\]

### Real pair in Reflow

\[
(Z_{0,R},X_1)
=
(V^{-1}(X_1),X_1)
\]

### IVD

\[
IVD(Z,t_0)
=
\mathbb E
\left[
\|(Z_1-Z_0)-\dot Z_{t_0}\|^2
\right]
\]

---

# 17. 이 강의를 다시 볼 때의 권장 복습 순서

1. **6장**: source / target distribution
2. **10–11장**: \(\psi_t\), velocity, ODE
3. **15장**: \(X_t=(1-t)X_0+tX_1\), velocity target
4. **17–20장**: conditional flow / marginalization / CFM loss
5. **25–29장**: conditioning / guidance
6. **33–39장**: crossing / learned trajectory / Reflow
7. **47–53장**: drift / fake pair / real pair / Conic Reflow
8. **54–56장**: straightness / IVD / experiments

이 순서로 보면 강의 흐름이 다시 이어질 가능성이 가장 높다.
