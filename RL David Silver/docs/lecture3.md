# Lecture 3 — Planning by Dynamic Programming 정리

David Silver의 Lecture 3는 Lecture 2에서 정의했던

$$
v^\pi,\quad q^\pi,\quad v^*,\quad \pi^*
$$

를 **환경 모델 \(P,R\)을 알고 있다는 가정 아래 실제로 계산하는 방법**을 다룬다.
즉 Lecture 2가 “무엇을 구하고 싶은가?”를 정의했다면, Lecture 3는 “그 값을 어떻게 계산할 것인가?”를 다룬다.

---

# 1. Dynamic Programming의 전제

Dynamic Programming은 MDP를 알고 있다고 가정한다.

$$
\langle S,A,P,R,\gamma\rangle
$$

즉 다음을 알고 있다.

$$
P_{ss'}^a
=
P(S_{t+1}=s'\mid S_t=s,A_t=a)
$$

$$
R_s^a
=
E[R_{t+1}\mid S_t=s,A_t=a]
$$

따라서 agent가 환경을 실제로 여러 번 실행하지 않아도 Bellman equation을 이용해 value를 계산할 수 있다.

이 점이 이후 model-free RL과의 가장 큰 차이다.

$$
\boxed{
DP = P,R을 알고 있는 planning
}
$$

---

# 2. 왜 Dynamic Programming이 가능한가

강의에서는 MDP가 두 가지 특성을 갖기 때문에 DP를 적용할 수 있다고 설명한다.

## 2.1 Optimal Substructure

큰 문제를 작은 subproblem들의 결과를 이용해 풀 수 있다.

Bellman equation 자체가 이 구조다.

$$
v(s)
=
R_s+\gamma\sum_{s'}P_{ss'}v(s')
$$

즉 현재 state의 value를 다음 state들의 value로 분해한다.

## 2.2 Overlapping Subproblems

여러 state의 value를 계산할 때 같은 \(v(s')\)가 반복적으로 필요하다.

따라서 한 번 계산한 value를 저장해서 다시 사용할 수 있다.

즉 DP는

$$
\boxed{
문제를 subproblem으로 나누고
+
계산한 subproblem의 값을 저장해서 재사용
}
$$

하는 방식이다.

---

# 3. Policy Evaluation

먼저 policy \(\pi\)가 주어졌다고 하자.

우리가 알고 싶은 것은

$$
v^\pi(s)
$$

이다.

즉:

> policy \(\pi\)를 계속 따랐을 때 state \(s\)에서 기대되는 return

을 계산하고 싶다.

이 과정이 **Policy Evaluation**이다.

---

# 4. 왜 policy를 아는데 value를 계산하는가

Policy를 안다는 것은 단지

> 각 state에서 어떤 action을 할 것인가

를 안다는 뜻이다.

하지만 그 policy가 좋은지 나쁜지는 모른다.

따라서

$$
v^\pi(s)
$$

를 계산해서 policy를 평가해야 한다.

그리고 이 value를 바탕으로

$$
q^\pi(s,a)
$$

를 계산하면 현재 policy보다 더 좋은 action이 있는지 확인할 수 있다.

즉:

$$
\boxed{
policy를 실행만 하려면 value가 필요 없지만,
policy를 개선하려면 policy evaluation이 필요하다.
}
$$

---

# 5. Iterative Policy Evaluation

Bellman expectation equation은

$$
v^\pi(s)
=
\sum_a\pi(a|s)
\left[
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v^\pi(s')
\right]
$$

이다.

문제는 왼쪽의 \(v^\pi(s)\)를 구하려는데 오른쪽에도 다시 \(v^\pi(s')\)가 있다는 것이다.

그래서 처음부터 정확한 값을 구하지 않고 초기값에서 시작한다.

예를 들어

$$
v_0(s)=0
$$

으로 둔다.

그다음

$$
v_{k+1}(s)
=
\sum_a\pi(a|s)
\left[
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v_k(s')
\right]
$$

로 반복한다.

즉:

$$
v_0
\rightarrow
v_1
\rightarrow
v_2
\rightarrow
\cdots
\rightarrow
v^\pi
$$

이다.

---

# 6. \(t\)와 \(k\)는 전혀 다른 개념이다

이 부분에서 가장 중요한 혼동이 있었다.

Bellman equation의

$$
S_t\rightarrow S_{t+1}
$$

에서 \(t\)는 **환경의 실제 시간축**이다.

반면

$$
v_k\rightarrow v_{k+1}
$$

에서 \(k\)는 **계산 iteration 횟수**다.

즉

$$
\boxed{
t\rightarrow t+1
=
agent가 환경에서 한 step 진행
}
$$

$$
\boxed{
k\rightarrow k+1
=
컴퓨터가 value estimate를 한 번 더 계산
}
$$

이다.

Iterative Policy Evaluation에서 agent가 실제 trajectory를 진행하는 것은 아니다.

\(P,R,\pi\)를 알고 있기 때문에 Bellman equation을 이용해서 value table을 반복 계산할 뿐이다.

---

# 7. 한 번의 iteration이 의미하는 것

Synchronous Iterative Policy Evaluation에서는 한 iteration 동안 모든 state를 한 번씩 업데이트한다.

$$
v_k
\rightarrow
v_{k+1}
$$

이 한 번의 iteration이다.

모든 state \(s\)에 대해

$$
v_{k+1}(s)
$$

를 계산할 때는 이전 iteration의

$$
v_k(s')
$$

만 사용한다.

즉 새로운 value table 전체를 만든 뒤 한꺼번에 교체한다.

$$
\boxed{
한 iteration
=
현재 value table 전체로
새 value table 전체를 한 번 생성
}
$$

---

# 8. 초기값이 실제 value라면

만약 우연히

$$
v_0=v^\pi
$$

였다면 Bellman equation을 이미 만족하므로

$$
T^\pi v_0=v_0
$$

이다.

따라서

$$
v_1=v_0
$$

가 된다.

하지만 일반적으로 초기값은 실제 value와 다르므로

$$
v_1\neq v_0
$$

이고,

$$
v_0
\rightarrow
v_1
\rightarrow
v_2
\rightarrow
\cdots
$$

식으로 계속 치환하면서 계산한다.

즉 iterative policy evaluation은 Bellman equation의 fixed point를 찾는 과정이다.

---

# 9. Policy Improvement

현재 policy \(\pi\)의 value를 구했다면,

$$
q^\pi(s,a)
=
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v^\pi(s')
$$

를 계산할 수 있다.

그다음 각 state에서 가장 좋은 action을 선택한다.

$$
\pi'(s)
=
\arg\max_a q^\pi(s,a)
$$

이게 **Greedy Policy Improvement**다.

즉:

$$
\boxed{
현재 policy가 만들어낸 미래를 기준으로,
현재 state에서 더 좋은 action이 있는지 확인
}
$$

하는 것이다.

---

# 10. 왜 한 번의 Policy Improvement로 끝나지 않는가

중요한 이유는

$$
q^\pi(s,a)
$$

가

> 현재 state에서 첫 action만 \(a\)로 바꾸고, 그 이후에는 기존 policy \(\pi\)를 따른다

는 가정으로 계산된 값이기 때문이다.

즉 policy improvement를 통해

$$
\pi'
$$

를 만들고 실제로 \(\pi'\)를 사용하기 시작하면 이후 state들에서의 행동도 모두 달라진다.

그러면

$$
v^\pi
\neq
v^{\pi'}
$$

가 되고,

$$
q^\pi
\neq
q^{\pi'}
$$

도 된다.

따라서 새 policy를 다시 평가해야 한다.

$$
\boxed{
\pi
\rightarrow
v^\pi
\rightarrow
q^\pi
\rightarrow
\pi'
\rightarrow
v^{\pi'}
\rightarrow
q^{\pi'}
\rightarrow
\pi''
\rightarrow\cdots
}
$$

이 반복이 Policy Iteration이다.

---

# 11. Policy Improvement에서 state들이 독립인가?

Policy improvement는

$$
\pi'(s)
=
\arg\max_a q^\pi(s,a)
$$

를 각 state별로 계산하기 때문에 state들을 독립적으로 보는 것처럼 느껴질 수 있다.

하지만 실제로는 그렇지 않다.

왜냐하면

$$
q^\pi(s,a)
=
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v^\pi(s')
$$

안에 이미 다음 state들의 value가 들어 있다.

즉 \(q^\pi(s,a)\)에는 미래 state들과의 연관성이 모두 반영되어 있다.

따라서 정확한 해석은:

$$
\boxed{
state들이 독립인 것이 아니라,
q^\pi가 미래 state들의 영향을 이미 요약하고 있기 때문에
각 state의 action 선택만 독립적으로 할 수 있다.
}
$$

---

# 12. Synchronous Policy Improvement

표준적인 synchronous policy improvement에서는 한 번의 sweep 동안 기존 policy를 고정한다.

즉

$$
\pi_k
$$

를 기준으로 모든 state에 대해

$$
\pi_{k+1}(s)
=
\arg\max_a q^{\pi_k}(s,a)
$$

를 계산한다.

예를 들어 먼저

$$
\pi_{k+1}(s_1)
$$

를 계산했다고 해서 그 새 값을 바로 이용해

$$
\pi_{k+1}(s_2)
$$

를 계산하지 않는다.

모든 state는 공통적으로 기존 \(\pi_k\)를 기준으로 계산된다.

다 만든 뒤에야

$$
\pi_{k+1}
$$

전체가 새로운 policy가 된다.

---

# 13. Policy Iteration 전체 구조

Lecture 3에서 가장 중요하게 기억할 프로세스는 이것이다.

$$
\boxed{
\pi_k
\rightarrow
v^{\pi_k}
\rightarrow
q^{\pi_k}
\rightarrow
\pi_{k+1}
}
$$

그리고 다시 반복한다.

좀 더 자세히 보면:

### 1. Policy Evaluation

현재 policy를 고정한다.

$$
\pi_k
$$

그리고

$$
v_0
\rightarrow
v_1
\rightarrow
v_2
\rightarrow\cdots
\rightarrow
v^{\pi_k}
$$

까지 여러 번 Bellman backup을 반복한다.

### 2. \(q^\pi\) 계산

$$
q^{\pi_k}(s,a)
=
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v^{\pi_k}(s')
$$

### 3. Policy Improvement

$$
\pi_{k+1}(s)
=
\arg\max_a q^{\pi_k}(s,a)
$$

### 4. 다시 Policy Evaluation

이번에는

$$
\pi_{k+1}
$$

를 기준으로 \(v^{\pi_{k+1}}\)를 계산한다.

---

# 14. 한 화살표 안에 여러 iteration이 들어 있다

Policy Iteration을 간단히

$$
\pi_k
\rightarrow
v^{\pi_k}
\rightarrow
q^{\pi_k}
\rightarrow
\pi_{k+1}
$$

라고 썼지만, 실제 계산량은 균일하지 않다.

특히

$$
\pi_k
\rightarrow
v^{\pi_k}
$$

안에는

$$
v_0
\rightarrow
v_1
\rightarrow
v_2
\rightarrow
\cdots
\rightarrow
v^{\pi_k}
$$

라는 여러 번의 내부 iteration이 들어 있다.

반면

$$
v^{\pi_k}\rightarrow q^{\pi_k}
$$

는 값 계산 한 sweep이면 되고,

$$
q^{\pi_k}\rightarrow\pi_{k+1}
$$

도 각 state에서 argmax를 한 번 계산하면 된다.

즉:

$$
\boxed{
Policy Evaluation은 내부 반복이 크고,
Policy Improvement는 한 sweep이면 된다.
}
$$

---

# 15. Modified Policy Iteration

여기서 자연스럽게 질문이 나온다.

> policy improvement를 하기 전에 정말 \(v^\pi\)가 완전히 수렴할 때까지 기다려야 하는가?

답은 아니다.

Policy Evaluation을 몇 번만 수행하고

$$
v_1,v_2,\ldots,v_k
$$

정도의 근사값만 얻은 상태에서 바로 policy improvement를 해도 된다.

즉

$$
\pi
\rightarrow
v_1
\rightarrow
v_2
\rightarrow\cdots
\rightarrow
v_k
\rightarrow
\pi'
$$

처럼 할 수 있다.

이것이 Modified Policy Iteration이다.

---

# 16. Value Iteration

Modified Policy Iteration을 더 극단적으로 가져가서,

> Policy Evaluation을 딱 한 번만 하고 바로 improvement

하면 Value Iteration으로 연결된다.

Value Iteration의 update는

$$
v_{k+1}(s)
=
\max_a
\left[
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v_k(s')
\right]
$$

이다.

즉 명시적인 policy를 유지하지 않고 value만 업데이트한다.

$$
v_0
\rightarrow
v_1
\rightarrow
v_2
\rightarrow
\cdots
\rightarrow
v^*
$$

---

# 17. Value Iteration에서는 policy가 사라진 것처럼 보인다

Policy Evaluation에서는

$$
\sum_a\pi(a|s)(\cdots)
$$

로 policy에 따라 action들을 평균냈다.

Value Iteration에서는

$$
\max_a(\cdots)
$$

를 사용한다.

즉 현재 value estimate 기준으로 가장 좋아 보이는 action을 바로 선택한다.

이를 policy notation으로 굳이 표현하면,

$$
\pi_k(a|s)
=
\begin{cases}
1 & a=\arg\max_a(\cdots)\\
0 & \text{otherwise}
\end{cases}
$$

처럼 현재 \(v_k\)에 대한 deterministic greedy policy를 암묵적으로 사용한다고 볼 수 있다.

하지만 이 policy를 별도로 저장하거나 평가하지 않는다.

$$
\boxed{
Value Iteration은
중간 policy를 명시적으로 관리하지 않고
max 연산에 녹여버린다.
}
$$

---

# 18. Value Iteration이 하는 일

한 state \(s\)에서 가능한 action이

$$
a_1,a_2,a_3
$$

라면 각각에 대해

$$
R_s^{a_1}
+
\gamma\sum_{s'}P_{ss'}^{a_1}v_k(s')
$$

$$
R_s^{a_2}
+
\gamma\sum_{s'}P_{ss'}^{a_2}v_k(s')
$$

$$
R_s^{a_3}
+
\gamma\sum_{s'}P_{ss'}^{a_3}v_k(s')
$$

를 계산한다.

그중 가장 큰 값을

$$
v_{k+1}(s)
$$

로 넣는다.

즉:

$$
\boxed{
현재 value estimate를 기준으로
각 action의 candidate value를 계산하고
그중 최고값만 남긴다.
}
$$

---

# 19. Value Iteration의 수렴

Value Iteration은

$$
v_{k+1}=T^*v_k
$$

를 반복한다.

실제로는

$$
v_{k+1}(s)\approx v_k(s)
$$

가 모든 state에서 성립할 때 수렴했다고 본다.

보통

$$
\max_s|v_{k+1}(s)-v_k(s)|<\epsilon
$$

같은 기준을 사용할 수 있다.

중요한 점은

> max를 한 번 했는데 같은 action이 나오는가

가 아니라,

> Bellman optimality backup을 다시 수행해도 value function 전체가 거의 변하지 않는가

이다.

---

# 20. Policy Iteration vs Value Iteration

Policy Iteration:

$$
\pi_k
\rightarrow
v^{\pi_k}
\rightarrow
\pi_{k+1}
$$

* policy를 명시적으로 유지
* policy evaluation을 충분히 수행
* 그다음 policy improvement

Value Iteration:

$$
v_k
\rightarrow
v_{k+1}
\rightarrow\cdots
\rightarrow
v^*
$$

* 중간 policy를 명시적으로 유지하지 않음
* Bellman optimality backup을 반복
* 최종 \(v^*\)에서 \(\pi^*\)를 추출

즉 두 알고리즘은 완전히 동떨어진 것이 아니라

> policy evaluation을 얼마나 많이 한 뒤 improvement하느냐

라는 연속선상에서 볼 수 있다.

---

# 21. Asynchronous Dynamic Programming

지금까지는 synchronous 방식이었다.

즉 한 iteration마다 모든 state를 업데이트했다.

Asynchronous DP에서는 이 제약을 없앤다.

$$
\boxed{
state를 하나씩 원하는 순서로 update할 수 있다.
}
$$

다만 중요한 수렴 조건은:

$$
\boxed{
장기적으로 어떤 state도 영원히 제외되면 안 된다.
}
$$

즉 모든 state가 계속 선택되어야 한다.

이것은

> 모든 state를 한 번씩 다 업데이트한 뒤에야 이전 state를 다시 업데이트해야 한다

는 뜻은 아니다.

예를 들어

$$
s_1,s_1,s_4,s_2,s_1,s_7,\ldots
$$

처럼 불균형하게 업데이트해도 괜찮다.

다만 아주 긴 시간으로 보면 어떤 state도 완전히 버려져서는 안 된다.

---

# 22. In-place Dynamic Programming

Synchronous 방식에서는

$$
v_{\text{old}}
$$

와

$$
v_{\text{new}}
$$

를 따로 둔다.

반면 In-place DP에서는 value table을 하나만 둔다.

$$
v(s)
\leftarrow
\max_a
\left[
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v(s')
\right]
$$

즉 state 하나를 계산하면 그 자리에서 바로 덮어쓴다.

그래서 다음 state를 계산할 때 방금 갱신된 value를 사용할 수도 있다.

개념 자체는 바뀌지 않고

$$
\boxed{
언제 새 값을 반영하는가
}
$$

만 달라진 것이다.

---

# 23. Prioritised Sweeping

Asynchronous DP에서는 어떤 state를 먼저 업데이트할지 자유롭게 정할 수 있다.

Prioritised Sweeping은

> value가 많이 틀려 있는 state부터 먼저 update하자

는 아이디어다.

이를 위해 Bellman error를 사용한다.

$$
\left|
\max_a
\left[
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v(s')
\right]
-
v(s)
\right|
$$

이 값이 크다면,

> 현재 저장된 \(v(s)\)와 Bellman equation이 요구하는 값의 차이가 크다

는 뜻이다.

따라서 priority를 높인다.

---

# 24. Predecessor가 중요한 이유

어떤 state \(s\)의 value가 크게 바뀌었다고 하자.

그리고

$$
p_1\rightarrow s
$$

$$
p_2\rightarrow s
$$

가 가능하다면 \(p_1,p_2\)는 \(s\)의 predecessor다.

\(v(s)\)가 크게 변경되면

$$
v(p_1),v(p_2)
$$

를 계산할 때 사용되는 Bellman target도 크게 바뀔 수 있다.

따라서

$$
\boxed{
현재 state의 value가 크게 수정됨
\Rightarrow
그 state로 들어오는 predecessor들도
다시 계산할 필요가 커짐
}
$$

이라고 볼 수 있다.

Prioritised Sweeping은 이런 predecessor들의 priority도 갱신한다.

---

# 25. Real-Time Dynamic Programming

RTDP는 asynchronous DP에서

> 어떤 state를 업데이트할 것인가

를 실제 agent trajectory를 이용해 정한다.

즉 agent가 실제로 방문한 state를 중심으로 Bellman backup을 수행한다.

중요한 점은 여전히 DP라는 것이다.

실제로 방문한 transition 하나만 보고 value를 업데이트하는 model-free 방식이 아니다.

현재 방문한 \(S_t\)에 대해 알고 있는 전체 \(P,R\)을 사용해서 Bellman backup을 한다.

즉:

$$
\boxed{
RTDP
=
실제로 방문한 state를 우선적으로 선택하는
Asynchronous DP
}
$$

이다.

State space가 매우 클 때 실제로 도달 가능하거나 중요한 영역에 계산을 집중할 수 있다.

---

# 26. Approximate Dynamic Programming

기존 DP는 모든 state에 대해

$$
v(s)
$$

를 table로 저장했다.

하지만 state space가 너무 크거나 continuous하면 이것이 어렵다.

그래서

$$
v(s)
\approx
\hat v(s,w)
$$

라는 function approximation을 사용한다.

예를 들어 neural network 역시 이런 function approximator가 될 수 있다.

다만 Lecture 3의 Approximate DP는 여전히

$$
P,R
$$

을 알고 있는 planning 문제다.

즉 Bellman target을 환경 모델을 이용해 직접 계산할 수 있다.

$$
\boxed{
Lecture\ 3:
Model\ known
+
Planning
+
Function\ Approximation
}
$$

이다.

이후 Lecture 4~6에서는 \(P,R\)을 모르는 model-free 상황으로 넘어가고, 실제 experience를 이용해 value를 학습한다.

---

# 27. Model이란 무엇인가

여기서 말하는 environment model은

$$
P
$$

와

$$
R
$$

이다.

즉

$$
\boxed{
Model
=
State\ Transition\ Model
+
Reward\ Model
}
$$

이다.

\(P,R\)을 안다는 것은

> state \(s\)에서 action \(a\)를 했을 때 다음 state가 어떤 분포로 나오고, reward가 어떻게 나오는지 안다

는 뜻이다.

그래서 agent가 실제로 실행하지 않아도 planning이 가능하다.

반대로 \(P,R\)을 명시적으로 알거나 사용하지 않고 경험에서 \(V,Q,\pi\)를 직접 학습하는 것이 model-free RL이다.

---

# 28. Contraction Mapping

Lecture 3 마지막 부분은

> Iterative Policy Evaluation과 Value Iteration이 왜 수렴하는가?

를 설명한다.

Policy Evaluation의 Bellman operator를

$$
T^\pi(v)
=
R^\pi+\gamma P^\pi v
$$

라고 하자.

두 value function \(u,v\) 사이의 거리를

$$
\|u-v\|_\infty
=
\max_s|u(s)-v(s)|
$$

로 정의한다.

그러면

$$
\|T^\pi u-T^\pi v\|_\infty
\le
\gamma\|u-v\|_\infty
$$

가 성립한다.

즉 Bellman backup을 한 번 적용하면 두 value function 사이의 최대 거리가 \(\gamma\)배 이하가 된다.

---

# 29. 왜 contraction이 되는가

$$
T^\pi(u)
=
R^\pi+\gamma P^\pi u
$$

$$
T^\pi(v)
=
R^\pi+\gamma P^\pi v
$$

이므로 차이를 보면

$$
T^\pi(u)-T^\pi(v)
=
\gamma P^\pi(u-v)
$$

이다.

Reward는 양쪽에서 동일하므로 사라진다.

그리고 \(P^\pi\)는 다음 state value들의 확률 가중평균이다.

확률 가중평균은 원래 존재하던 최대 차이보다 더 큰 차이를 새로 만들 수 없다.

따라서

$$
\|P^\pi(u-v)\|_\infty
\le
\|u-v\|_\infty
$$

이고,

$$
\|T^\pi u-T^\pi v\|_\infty
\le
\gamma\|u-v\|_\infty
$$

가 된다.

---

# 30. Fixed Point와 수렴

진짜 value \(v^\pi\)는 Bellman equation을 만족하므로

$$
T^\pi v^\pi=v^\pi
$$

이다.

즉 \(v^\pi\)는 Bellman operator의 fixed point다.

그리고 contraction mapping theorem에 의해 contraction operator는 유일한 fixed point를 갖고, 반복 적용하면 그 fixed point로 수렴한다.

따라서

$$
v_0
\rightarrow
T^\pi v_0
\rightarrow
(T^\pi)^2v_0
\rightarrow\cdots
$$

는 결국

$$
v^\pi
$$

로 수렴한다.

Value Iteration 역시 Bellman optimality operator \(T^*\)가 contraction이므로

$$
v_k\rightarrow v^*
$$

로 수렴한다.

---

# 31. \(\gamma=1\)인 경우

Contraction 증명에서 핵심은

$$
\gamma<1
$$

이다.

왜냐하면

$$
\|T u-T v\|
\le
\gamma\|u-v\|
$$

에서 \(\gamma<1\)이어야 거리가 실제로 줄어든다고 보장할 수 있기 때문이다.

만약

$$
\gamma=1
$$

이면

$$
\|T u-T v\|
\le
\|u-v\|
$$

일 뿐이므로 contraction argument를 그대로 사용할 수 없다.

그렇다고 \(\gamma=1\)이면 무조건 수렴하지 않는 것은 아니다.

Episode가 반드시 종료되는 episodic MDP에서는 \(\gamma=1\)을 사용할 수 있고, 별도의 조건으로 수렴할 수 있다.

즉:

$$
\boxed{
\gamma<1
\Rightarrow
contraction으로 깔끔하게 수렴 보장
}
$$

$$
\boxed{
\gamma=1
\Rightarrow
이 contraction proof는 그대로 사용할 수 없음
}
$$

이다.

---

# 32. Lecture 3 전체 흐름

Lecture 3을 한 번에 다시 떠올리려면 다음 구조를 기억하면 된다.

## Prediction

현재 policy가 주어짐:

$$
\pi
$$

그 policy를 평가:

$$
\pi
\rightarrow
v^\pi
$$

## Control — Policy Iteration

$$
\boxed{
\pi_k
\rightarrow
v^{\pi_k}
\rightarrow
q^{\pi_k}
\rightarrow
\pi_{k+1}
}
$$

그리고 반복.

## Control — Value Iteration

중간 policy를 명시적으로 제거:

$$
\boxed{
v_k
\xrightarrow{\text{Bellman Optimality Backup}}
v_{k+1}
\rightarrow
\cdots
\rightarrow
v^*
}
$$

마지막에

$$
\pi^*
$$

를 추출.

---

# 33. Lecture 2와 Lecture 3의 연결

Lecture 2에서는

$$
v^\pi,\quad
q^\pi,\quad
v^*,\quad
q^*,\quad
\pi^*
$$

를 정의했다.

즉:

> 우리가 궁극적으로 원하는 값이 무엇인가?

를 정의했다.

Lecture 3에서는

$$
P,R
$$

을 알고 있다는 전제 아래 이 값들을 실제로 계산한다.

즉:

$$
\boxed{
Lecture\ 2
=
정답이 무엇인지 정의
}
$$

$$
\boxed{
Lecture\ 3
=
model을 알고 있을 때 그 정답을 계산하는 방법
}
$$

이다.

그리고 다음 Lecture 4부터는

$$
P,R
$$

을 모르는 상황으로 이동한다.

즉 agent가 실제 experience를 통해 value를 학습하는 **Model-Free Prediction**이 시작된다.

---

# 최종 Remind

Lecture 3에서 가장 먼저 떠올릴 것은 이 두 식이다.

### Policy Iteration

$$
\boxed{
\pi_k
\rightarrow
v^{\pi_k}
\rightarrow
q^{\pi_k}
\rightarrow
\pi_{k+1}
}
$$

여기서

$$
\pi_k\rightarrow v^{\pi_k}
$$

안에는 실제로

$$
v_0\rightarrow v_1\rightarrow v_2\rightarrow\cdots
$$

라는 긴 Policy Evaluation iteration이 들어 있다.

### Value Iteration

$$
\boxed{
v_{k+1}(s)
=
\max_a
\left[
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v_k(s')
\right]
}
$$

중간 policy를 따로 유지하지 않고 현재 value estimate에서 greedy action을 바로 반영한다.

그리고 Lecture 3 전체의 전제는:

$$
\boxed{
P,R을 알고 있다.
}
$$

따라서 실제 agent trajectory를 샘플링하지 않고도 Bellman equation만으로 planning할 수 있다.

마지막으로 수렴성의 핵심은:

$$
\boxed{
Bellman\ operator는
value\ function을
fixed\ point\ 방향으로 contraction시킨다.
}
$$

이다.
