# Lecture 2 — Markov Decision Processes 정리

David Silver의 Lecture 2는 강화학습에서 앞으로 계속 사용하게 될 수학적 대상들을 정의하는 강의다.
중요한 점은, 여기서 정의하는 \(G_t\), \(v(s)\), \(q(s,a)\), \(v^*(s)\) 등을 처음부터 실제 환경에서 정확하게 계산하려는 것이 아니라는 것이다.

이 강의의 역할은 먼저 다음을 수학적으로 명확히 만드는 것이다.

> 강화학습에서 우리가 궁극적으로 알고 싶고, 추정하고, 최적화하려는 값이 정확히 무엇인가?

이후 Lecture 3~7에서는 이 값들을 실제로 어떻게 계산하거나 근사할 것인지를 배우게 된다.

---

## 1. Markov Property

Markov property는 다음과 같다.

$$
P(S_{t+1}\mid S_t)
=
P(S_{t+1}\mid S_1,\ldots,S_t)
$$

즉,

> 현재 state를 알고 있다면 미래를 예측하기 위해 과거 history를 추가로 알 필요가 없다.

과거가 영향을 주지 않았다는 뜻이 아니다.
과거의 정보 중 미래 예측에 필요한 정보가 이미 현재 state \(S_t\) 안에 들어 있다는 뜻이다.

따라서 state는 미래에 대한 **sufficient statistic**으로 볼 수 있다.

---

# 2. Markov Process

Markov Process 또는 Markov Chain은

$$
\langle S,P\rangle
$$

로 정의한다.

현재 state가 \(s\)일 때 다음 state가 \(s'\)일 확률은

$$
P_{ss'}
=
P(S_{t+1}=s'\mid S_t=s)
$$

이다.

즉 가장 기본적인 구조는

$$
s \xrightarrow{P_{ss'}} s'
$$

이다.

여기에는 아직 action도 reward도 없다.

Transition matrix \(P\)는 모든 \(s\rightarrow s'\) 확률을 담고 있고, 각 state에서 결국 어떤 다음 state로는 이동해야 하므로 각 row의 합은 1이다.

---

# 3. Markov Reward Process

Markov Process에 reward와 discount factor를 추가하면 MRP가 된다.

$$
\boxed{
MRP=\langle S,P,R,\gamma\rangle
}
$$

여기서 reward와 value를 구분하는 것이 중요하다.

---

## 3.1 \(R_{t+1}\)과 \(R_s\)

$$
R_{t+1}
$$

은 실제 trajectory에서 transition을 한 번 수행했을 때 관측되는 **reward 확률변수**다.

현재

$$
S_t=s
$$

라고 하더라도 transition 결과가 stochastic하면 \(R_{t+1}\)은 여러 값 중 하나가 될 수 있다.

반면

$$
R_s
=
E[R_{t+1}\mid S_t=s]
$$

이다.

따라서:

$$
\boxed{
R_{t+1}
=
실제로\ 발생하는\ 1-step\ reward
}
$$

$$
\boxed{
R_s
=
state\ s의\ 1-step\ 기대\ reward
}
$$

라고 이해하면 된다.

예를 들어 \(s\)에서 transition했을 때 reward가

$$
10\quad(20\%)
$$

$$
0\quad(80\%)
$$

라면,

$$
R_s
=
0.2(10)+0.8(0)
=
2
$$

이다.

실제 한 trajectory에서는 \(R_{t+1}=10\) 또는 \(0\)이지만,

$$
R_s=2
$$

는 그 확률변수의 기대값이다.

### 시간 index를 이해하는 방법

현재가 \(t\) 시점이면

$$
S_t
$$

에서 transition이 발생하고 그 결과로

$$
R_{t+1}, S_{t+1}
$$

을 받는다.

따라서

$$
R_s=E[R_{t+1}\mid S_t=s]
$$

는

> t 시점의 state가 \(s\)일 때, t+1 시점에 받게 될 reward의 평균

이라고 읽으면 된다.

---

# 4. Return \(G_t\)

Reward는 한 step만 본다.

하지만 강화학습에서는 현재 action의 영향이 먼 미래까지 이어질 수 있기 때문에 미래의 reward 전체를 봐야 한다.

그래서 Return을 정의한다.

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+\cdots
$$

여기서 중요한 점:

$$
\boxed{
G_t는\ 기대값이\ 아니다.
}
$$

\(G_t\)는 **한 trajectory에서 실제로 발생한 미래 reward들을 discounted sum한 확률변수**다.

즉 하나의 episode가 실제로

$$
S_t\rightarrow S_{t+1}\rightarrow S_{t+2}\rightarrow\cdots
$$

진행되었다면, 그 trajectory에서 발생한

$$
R_{t+1},R_{t+2},R_{t+3},\ldots
$$

를 이용해 \(G_t\) 하나를 계산할 수 있다.

---

## 4.1 \(R_{t+1}\)과 \(G_t\)의 대응

같은 기준 시점 \(S_t\)에서 보면:

| 범위       | 확률변수        |
| -------- | ----------- |
| 1-step   | \(R_{t+1}\) |
| all-step | \(G_t\)     |

즉

$$
R_{t+1}
\leftrightarrow
G_t
$$

라고 대응시켜 생각할 수 있다.

둘의 index가 하나 어긋나 보이는 이유는:

* Reward는 transition이 끝난 시점으로 indexing
* Return은 출발 state의 시점으로 indexing

하기 때문이다.

실제로

$$
G_t
=
R_{t+1}
+\gamma R_{t+2}
+\cdots
$$

이므로 둘 모두 \(S_t\)에서 출발한 결과를 설명한다.

---

# 5. Value Function

State value는

$$
v(s)=E[G_t\mid S_t=s]
$$

이다.

즉

> state \(s\)에서 시작했을 때 발생 가능한 return \(G_t\)의 평균

이다.

우리가 사용한 표현으로 정리하면:

$$
\boxed{
R_s
=
현재\ state의\ 1-step\ 기대보상
}
$$

$$
\boxed{
v(s)
=
현재\ state의\ all-step\ 기대보상
}
$$

그리고 구조적으로:

| 범위       | trajectory에서의 확률변수 | state 기준 기대값 |
| -------- | ------------------ | ------------ |
| 1-step   | \(R_{t+1}\)        | \(R_s\)      |
| all-step | \(G_t\)            | \(v(s)\)     |

즉

$$
R_{t+1}\leftrightarrow G_t
$$

와 동시에

$$
R_s\leftrightarrow v(s)
$$

라는 공통 구조로 볼 수 있다.

---

# 6. 미래 branch와 \(G_t\)

현재 state \(s\)에서

$$
s\rightarrow a
$$

또는

$$
s\rightarrow b
$$

가 가능하다고 하자.

그리고 이후에는

$$
a\rightarrow a_1,a_2
$$

$$
b\rightarrow b_1,b_2
$$

만 가능하며, \(b\) branch에 들어가면 앞으로 영원히 \(a_1,a_2\) 쪽으로 갈 수 없다고 하자.

그러면 처음 어떤 transition을 했느냐에 따라

$$
R_{t+2},R_{t+3},\ldots
$$

의 가능한 분포 자체가 완전히 달라진다.

따라서 \(G_t\)는 단순히 서로 독립적인 reward들을 더하는 값이 아니다.

$$
\boxed{
G_t는\ trajectory에\ 의존하는\ 확률변수다.
}
$$

그리고

$$
v(s)=E[G_t\mid S_t=s]
$$

를 정확하게 계산하려면 원칙적으로 \(s\)에서 출발하는 모든 가능한 trajectory와 각각의 확률을 고려해야 한다.

실제 RL에서는 이를 모두 enumerate하지 않고 trajectory들을 sample하여 \(v(s)\)를 추정하게 된다.

---

# 7. Discount Factor \(\gamma\)

Return은

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+\cdots
$$

이므로 \(\gamma\)는 미래 reward를 얼마나 중요하게 볼지를 결정한다.

$$
\gamma\approx0
$$

이면 immediate reward 중심이고,

$$
\gamma\approx1
$$

이면 먼 미래 reward까지 중요하게 본다.

종료가 보장되는 episodic problem에서는

$$
\gamma=1
$$

도 사용할 수 있다.

---

# 8. Bellman Equation

Return은 다음처럼 재귀적으로 쓸 수 있다.

$$
G_t
=
R_{t+1}
+
\gamma G_{t+1}
$$

따라서 value 역시

$$
v(s)
=
R_s
+
\gamma
\sum_{s'}P_{ss'}v(s')
$$

로 쓸 수 있다.

의미는:

$$
\boxed{
현재\ state의\ value
=
현재\ state의\ 1-step\ 기대 reward
+
discount된\ 다음\ state\ value의\ 기대값
}
$$

여기서

$$
\sum_{s'}P_{ss'}v(s')
$$

는 단순 산술평균이 아니라 **transition probability를 이용한 가중평균**이다.

---

## 8.1 등비수열과 비슷하게 이해하기

Return은

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+\cdots
$$

처럼 무한한 미래를 포함한다.

이를 직접 전부 펼쳐서 계산하려는 대신

$$
G_t
=
R_{t+1}
+\gamma G_{t+1}
$$

라는 현재와 다음 시점의 재귀 관계를 이용한다.

이 아이디어는 등비수열에서

$$
S=1+r+r^2+\cdots
$$

를

$$
S=1+rS
$$

로 변형해 무한합을 계산하는 것과 비슷하다.

MDP는 등비수열 자체는 아니지만,

> 무한하게 이어지는 미래 전체를 현재와 다음 단계 사이의 관계로 압축한다

는 수학적 전략이 유사하다.

이것이 Bellman equation의 핵심 아이디어다.

---

# 9. Bellman Equation의 Matrix Form

모든 state의 value를 vector로 쓰면

$$
v=R+\gamma Pv
$$

이고,

$$
(I-\gamma P)v=R
$$

이므로

$$
v=(I-\gamma P)^{-1}R
$$

로 직접 계산할 수도 있다.

따라서 MRP의 \(P,R,\gamma\)를 완전히 알고 있고 state space가 작다면 \(v\)를 직접 계산할 수 있다.

하지만 큰 환경에서는 이런 방식이 비싸기 때문에 이후 Dynamic Programming, Monte-Carlo, TD 등이 등장한다.

---

# 10. MDP — Action의 등장

MRP에서는

$$
s\rightarrow s'
$$

라는 transition만 있었다.

MDP에서는 중간에 action이 들어간다.

$$
\boxed{
MDP=\langle S,A,P,R,\gamma\rangle
}
$$

Transition probability는

$$
P_{ss'}^a
=
P(S_{t+1}=s'\mid S_t=s,A_t=a)
$$

가 되고,

Reward 역시

$$
R_s^a
=
E[R_{t+1}\mid S_t=s,A_t=a]
$$

가 된다.

즉 이제는

$$
s\rightarrow a\rightarrow s'
$$

라는 한 단계가 추가된다.

이것을 확률 구조로 보면:

$$
s
\xrightarrow{\pi(a|s)}
a
\xrightarrow{P_{ss'}^a}
s'
$$

이다.

---

# 11. Policy

Policy는

$$
\pi(a|s)
=
P(A_t=a\mid S_t=s)
$$

이다.

즉

> state \(s\)에서 action \(a\)를 선택할 확률

이다.

MDP에서는 state만 주어졌다고 미래 trajectory의 확률분포가 결정되지 않는다.

어떤 action을 선택할지 정해주는 policy가 있어야 한다.

따라서 MRP에서는

$$
E[G_t\mid S_t=s]
$$

라고만 써도 됐지만,

MDP에서는

$$
E_\pi[G_t\mid S_t=s]
$$

처럼 policy를 명시한다.

여기서 \(E_\pi\)는 새로운 종류의 expectation 연산이 아니다.

> policy \(\pi\)가 만들어내는 action-selection distribution과, 환경의 transition distribution이 결합되어 만들어내는 trajectory distribution에 대한 expectation

이라는 뜻을 축약해서 쓴 notation이다.

---

# 12. MDP + Policy = MRP

Policy가 고정되면 MDP의 action randomness를 평균내어 다시 하나의 MRP처럼 표현할 수 있다.

$$
P^\pi_{ss'}
=
\sum_a
\pi(a|s)P^a_{ss'}
$$

$$
R^\pi_s
=
\sum_a
\pi(a|s)R_s^a
$$

따라서

$$
\boxed{
MDP + \pi
\Rightarrow
MRP
}
$$

라고 이해할 수 있다.

---

# 13. State Value \(v^\pi(s)\)

MDP에서는 policy에 따라 value가 달라지므로

$$
v^\pi(s)
=
E_\pi[G_t\mid S_t=s]
$$

라고 쓴다.

의미는:

> state \(s\)에서 시작해서 지금부터 policy \(\pi\)를 계속 따를 경우의 기대 return

이다.

MRP에서의

$$
v(s)
$$

와 완전히 대응되는 개념이다.

차이는 MDP에서는 action을 선택해야 하기 때문에 policy가 추가된 것뿐이다.

---

# 14. Action Value \(q^\pi(s,a)\)

MDP에서는 transition 전에 action을 한 번 선택해야 한다.

그래서 state에 대한 value뿐 아니라 **state-action pair에 대한 value**도 정의한다.

$$
q^\pi(s,a)
=
E_\pi[G_t\mid S_t=s,A_t=a]
$$

의미는:

> 현재 state \(s\)에서 첫 action만 \(a\)로 고정하고, 그 이후에는 policy \(\pi\)를 따를 때의 기대 return.

즉:

$$
\boxed{
v^\pi(s)
=
이\ state에서\ policy대로\ 행동하면?
}
$$

$$
\boxed{
q^\pi(s,a)
=
이\ state에서\ 지금\ action\ a를\ 하면?
}
$$

이다.

---

## 14.1 왜 \(q\)라는 새로운 notation이 필요한가

MRP에서는

$$
s\rightarrow s'
$$

한 번의 transition뿐이었다.

MDP에서는

$$
s\rightarrow a\rightarrow s'
$$

라는 중간 단계가 추가되었다.

따라서 value 역시

$$
v^\pi(s)
\rightarrow
q^\pi(s,a)
\rightarrow
v^\pi(s')
$$

처럼 중간 action 단계에 대한 value가 필요해졌다.

즉 \(q\)는 전혀 다른 종류의 개념이라기보다,

> action이라는 중간 단계가 생기면서 필요한 state-action value

라고 보는 것이 자연스럽다.

---

# 15. \(v^\pi\)와 \(q^\pi\)의 관계

State \(s\)에서 policy가 여러 action을 확률적으로 선택한다면

$$
v^\pi(s)
=
\sum_a
\pi(a|s)q^\pi(s,a)
$$

이다.

즉 \(v^\pi(s)\)는 각 action의 \(q^\pi(s,a)\)를 policy 확률로 가중평균한 값이다.

반대로

$$
q^\pi(s,a)
=
R_s^a
+
\gamma
\sum_{s'}P^a_{ss'}v^\pi(s')
$$

이다.

즉

$$
\boxed{
v^\pi
\xrightarrow{\pi}
q^\pi
\xrightarrow{P,R}
v^\pi
}
$$

라는 반복 구조가 만들어진다.

---

# 16. Bellman Expectation Equation for MDP

두 식을 합치면

$$
v^\pi(s)
=
\sum_a
\pi(a|s)
\left[
R_s^a
+
\gamma
\sum_{s'}
P^a_{ss'}v^\pi(s')
\right]
$$

이 된다.

즉 결국 다시

$$
v^\pi(s)
\leftrightarrow
v^\pi(s')
$$

관계를 만든 것이다.

MRP에서는

$$
s\rightarrow s'
$$

였지만,

MDP에서는

$$
s\rightarrow a\rightarrow s'
$$

라는 단계가 하나 더 생겼기 때문에

$$
\sum_a\pi(a|s)
$$

가 추가된 것뿐이다.

---

# 17. Optimal Value

지금까지는 특정 policy \(\pi\)에 대해

$$
v^\pi,\quad q^\pi
$$

를 정의했다.

이제 모든 policy 중 가장 좋은 policy를 생각한다.

$$
v^*(s)
=
\max_\pi v^\pi(s)
$$

$$
q^*(s,a)
=
\max_\pi q^\pi(s,a)
$$

이다.

중요한 점은 \(v^*\)가 여러 \(v^\pi\)를 평균낸 별도의 개념이 아니라는 것이다.

Optimal policy를

$$
\pi^*
$$

라고 하면

$$
v^*(s)=v^{\pi^*}(s)
$$

$$
q^*(s,a)=q^{\pi^*}(s,a)
$$

이다.

즉

$$
\boxed{
v^*,q^*
=
가장\ 좋은\ policy에서의\ 기존\ value
}
$$

라고 이해하면 된다.

---

# 18. \(q^*(s,a)\)의 중요한 의미

$$
q^*(s,a)
$$

는

> state \(s\)에서 이번 첫 action만 \(a\)라고 고정하고, 그 이후에는 가능한 최고의 행동을 계속한다고 가정했을 때의 기대 return

이다.

여기서 중요한 아이디어가 생긴다.

미래 전체에 대해서는 이미

> 이후에는 항상 최적으로 행동한다

라고 \(q^*\)의 정의 안에서 처리해버렸다.

그러면 현재 우리가 비교해야 할 것은 오직 첫 action \(a\)뿐이다.

따라서 optimal policy는

$$
\boxed{
\pi^*(s)
=
\arg\max_a q^*(s,a)
}
$$

가 된다.

즉 미래 전략 전체를 다시 비교할 필요 없이, 현재 state에서 \(q^*\)를 가장 크게 만드는 action 하나만 선택하면 된다.

---

# 19. Bellman Optimality Equation

Optimal state value는

$$
v^*(s)
=
\max_a q^*(s,a)
$$

이다.

그리고

$$
q^*(s,a)
=
R_s^a
+
\gamma
\sum_{s'}
P^a_{ss'}v^*(s')
$$

이므로,

$$
\boxed{
v^*(s)
=
\max_a
\left[
R_s^a
+
\gamma
\sum_{s'}
P^a_{ss'}v^*(s')
\right]
}
$$

이 된다.

기존 Bellman Expectation Equation에서는

$$
\sum_a\pi(a|s)(\cdots)
$$

로 **현재 policy대로 action들을 평균**냈다.

Bellman Optimality Equation에서는

$$
\max_a(\cdots)
$$

로 **가능한 action 중 가장 좋은 것을 선택**한다.

즉:

$$
\boxed{
Bellman\ Expectation
=
현재\ policy를\ 평가
}
$$

$$
\boxed{
Bellman\ Optimality
=
최적으로\ 행동했을\ 때의\ value를\ 정의
}
$$

이다.

---

# 20. \(q^*\)의 Bellman Optimality Equation

다음 state에서의 optimal value는

$$
v^*(s')
=
\max_{a'}q^*(s',a')
$$

이므로,

$$
q^*(s,a)
=
R_s^a
+
\gamma
\sum_{s'}
P^a_{ss'}
\max_{a'}q^*(s',a')
$$

라고 쓸 수 있다.

즉:

> 지금 \(a\)를 한 결과
>
> * 다음 state부터 다시 가장 좋은 action을 선택했을 때의 value

라는 구조다.

---

# 21. Lecture 2 전체를 한 흐름으로 보기

Lecture 2에서 계속 새로운 notation이 등장하지만 실제 구조는 상당히 일관되어 있다.

### 1. Markov Process

$$
s\rightarrow s'
$$

$$
P_{ss'}
$$

### 2. MRP

$$
s\rightarrow s'
$$

여기에 immediate reward를 추가.

$$
R_s
$$

미래 전체를 평가하기 위해

$$
G_t
$$

와

$$
v(s)
$$

를 정의.

### 3. MDP

중간에 action이 추가.

$$
s\rightarrow a\rightarrow s'
$$

따라서

$$
\pi(a|s)
$$

와

$$
P^a_{ss'}
$$

가 필요해지고,

action 단계의 value를 표현하기 위해

$$
q^\pi(s,a)
$$

가 추가된다.

### 4. Optimality

특정 policy가 아니라 가능한 최고의 policy를 선택하면

$$
v^\pi,q^\pi
$$

가

$$
v^*,q^*
$$

로 바뀐다.

그리고 현재 action을 고르는 optimal policy는

$$
\pi^*(s)
=
\arg\max_a q^*(s,a)
$$

가 된다.

---

# 22. Lecture 2가 결국 하고 있는 일

이 강의를 이해하면서 가장 중요했던 관점은 다음과 같다.

Lecture 2는 아직 PPO, SAC, Q-learning 같은 알고리즘을 배우는 강의가 아니다.

먼저

$$
v^\pi,\quad q^\pi,\quad v^*,\quad q^*,\quad\pi^*
$$

를 정의해서

> 강화학습 알고리즘이 궁극적으로 무엇을 알아내려고 하는가?

를 수학적으로 규정한다.

실제 환경에서는 \(v^*\)나 \(q^*\)를 바로 알고 있지 않다.

그래서 이후 강의에서:

* Dynamic Programming
* Monte-Carlo
* Temporal-Difference Learning
* Q-learning
* Sarsa
* Policy Gradient
* Actor-Critic

등을 이용해 이 값 또는 policy를 계산하거나 추정한다.

현대 알고리즘인 PPO, SAC 역시 결국

$$
\boxed{
좋은\ policy를\ 찾아\ optimal\ policy에\ 가까워지는\ 방법
}
$$

이라는 큰 틀 안에 위치한다.

---

# 최종 Remind

Lecture 2를 다시 볼 때 가장 먼저 이 구조를 떠올린다.

$$
\boxed{
R_{t+1}
\rightarrow
R_s
}
$$

실제 1-step reward 확률변수 → 1-step 기대 reward

$$
\boxed{
G_t
\rightarrow
v(s)
}
$$

실제 trajectory의 all-step return → all-step 기대 return

그리고 action이 추가되면

$$
\boxed{
s
\xrightarrow{\pi}
a
\xrightarrow{P,R}
s'
}
$$

가 되고,

$$
\boxed{
v^\pi(s)
\rightarrow
q^\pi(s,a)
\rightarrow
v^\pi(s')
}
$$

라는 구조가 생긴다.

마지막으로 모든 미래를 직접 펼치는 대신

$$
현재\ value
\leftrightarrow
다음\ value
$$

의 재귀 관계를 이용하는 것이 Bellman equation의 핵심이다.

$$
\boxed{
무한한\ 미래를\ 한\ step의\ 재귀관계로\ 압축한다.
}
$$

이 관점이 Lecture 2의 가장 중요한 중심축이다.
