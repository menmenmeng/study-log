# Lecture 4 — Model-Free Prediction

David Silver, Reinforcement Learning

## 0. Lecture 4의 목적

Lecture 3에서는 환경 모델인 Transition Probability \(P\)와 Reward Function \(R\)을 알고 있다고 가정했다. 따라서 Bellman Equation을 이용해 State Value를 직접 계산할 수 있었다.

Lecture 4에서는 이 가정을 제거한다.

**환경 모델 \(P,R\)을 모르는 상황에서, 실제 Experience를 이용해 Value Function을 추정하는 방법**을 배우는 강의다.

전체 흐름은 다음과 같다.

**Monte-Carlo → Temporal-Difference → n-step TD → TD(λ)**

새로운 알고리즘이 등장할 때마다 무엇을 바꾸었는지 이해하는 것이 중요하다.

---

# 1. Dynamic Programming과 Model-Free Prediction

## 1.1 Dynamic Programming

Lecture 3의 DP에서는 MDP를 알고 있다고 가정했다.

$$
MDP = \langle S,A,P,R,\gamma \rangle
$$

따라서 현재 Policy가 주어졌다면 Bellman Expectation Equation을 통해 Value Function을 계산할 수 있었다.

$$
v^\pi(s)
=
\sum_a \pi(a|s)
\left[
R_s^a
+
\gamma \sum_{s'}P_{ss'}^a v^\pi(s')
\right]
$$

이 방법은 가능한 다음 State와 각각의 Transition Probability를 이용해 Expectation을 직접 계산한다.

## 1.2 Model-Free Prediction

하지만 현실에서는 일반적으로 다음 두 값을 알 수 없다.

$$
P_{ss'}^a
=
P(S_{t+1}=s'|S_t=s,A_t=a)
$$

$$
R_s^a
=
E[R_{t+1}|S_t=s,A_t=a]
$$

이때 agent는 환경을 실제로 실행해 Experience를 수집할 수 있다.

$$
S_t,A_t,R_{t+1},S_{t+1},\ldots
$$

이 경험을 이용해서 다음 값을 추정한다.

$$
v^\pi(s)=E_\pi[G_t|S_t=s]
$$

따라서 핵심적인 차이는 다음과 같다.

| | Dynamic Programming | Model-Free Prediction |
|---|---|---|
| 환경 모델 | 알고 있음 | 모름 |
| 학습에 사용하는 정보 | \(P,R\) | 실제 Experience |
| Expectation | 모델로 계산 | Sample로 추정 |
| 주요 목표 | \(v^\pi\) 계산 | \(v^\pi\) 추정 |

## 1.3 동전 던지기와의 비교

DP와 MC를 이해하기 위해 우리가 사용했던 비교가 있었다.

동전이 공정하다는 사실을 알고 있다면 앞면의 확률을 직접 계산할 수 있다.

$$
P(H)=0.5
$$

반면 동전이 공정한지 모르는 경우, 100,000번 던져서 앞면이 50,021번 나왔다면 다음처럼 추정한다.

$$
\hat P(H)=\frac{50021}{100000}
$$

즉 DP는 **확률분포를 알고 기댓값을 계산하는 것**이고, MC는 **확률분포를 모르지만 Sample Average로 기댓값을 추정하는 것**이다.

다만 RL에서 평균내는 대상은 단순한 Reward 하나가 아니라 Trajectory 전체의 Return이라는 점이 다르다.

---

# 2. Reward를 연구자가 설계하는데 왜 R을 모르는가?

이 부분에서 한 가지 중요한 질문이 있었다.

> Reward는 연구자가 설계하는 영역인데, 그렇다면 R은 항상 알고 있는 것 아닌가?

여기서는 Reward Rule과 Reward Model을 구분해야 한다.

예를 들어 연구자가 다음과 같은 Reward Rule을 만들었다고 하자.

- 성공하면 +10
- 실패하면 -2

하지만 특정 State에서 어떤 Action을 수행했을 때 성공할 확률을 모를 수 있다.

성공확률을 \(p\)라고 하면 다음과 같다.

$$
R_s^a=p(10)+(1-p)(-2)
$$

Reward Rule 자체는 알고 있지만 \(p\)를 모르면 \(R_s^a\)를 계산할 수 없다.

따라서 구분하면 다음과 같다.

**Reward Rule:** 어떤 결과에 어떤 보상을 부여할지 정한 규칙

**Reward Model:** State-Action Pair에서 발생할 Reward의 확률적 특성을 나타내는 환경 모델

Model-Free RL에서는 Reward Signal을 관측할 수 있지만, 모든 State-Action Pair의 Reward Model을 미리 알고 있을 필요는 없다.

---

# 3. Monte-Carlo Prediction

## 3.1 기본 아이디어

Lecture 2에서 State Value를 정의했다.

$$
v^\pi(s)=E_\pi[G_t|S_t=s]
$$

여기서 Return은 다음과 같다.

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

즉 Value는 현재 State에서 시작했을 때의 Return 기댓값이다.

그렇다면 특정 State를 여러 번 방문해서 그때마다 Return을 기록하고 평균내면 된다.

$$
V(s)
=
\frac{1}{N(s)}
\sum_{i=1}^{N(s)}G^{(i)}(s)
$$

여기서 \(G^{(i)}(s)\)는 State \(s\)에서 관측한 i번째 Return Sample을 의미한다.

**Monte-Carlo의 핵심: Value = Mean Return**

## 3.2 MC는 Episode가 끝나야 한다

우리가 한 번 더 짚었던 중요한 점이 있다.

> A를 방문하는 순간에는 아직 미래 Reward가 발생하지 않았는데, 어떻게 G를 계산할 수 있는가?

정확히 맞는 의문이다.

MC는 State를 방문하는 순간 바로 Value를 업데이트하지 않는다.

예를 들어 다음 Trajectory가 있다고 하자.

$$
A\rightarrow B\rightarrow C\rightarrow Terminal
$$

A를 방문하는 순간에는 앞으로 발생할 Reward를 모른다.

Terminal에 도착한 후에야 각 방문 시점의 Return을 계산할 수 있다.

따라서 실행 순서는 다음과 같다.

1. Episode 전체 실행
2. Terminal 도달
3. 방문 시점별 Return 계산
4. Value Function 업데이트

이것이 Monte-Carlo가 **Complete Episode**를 필요로 하는 이유다.

---

# 4. First-Visit과 Every-Visit MC

## 4.1 같은 State를 여러 번 방문할 수 있다

예를 들어 Episode가 다음과 같다고 하자.

$$
A\rightarrow B\rightarrow A\rightarrow C\rightarrow Terminal
$$

여기서 A는 두 번 등장한다.

첫 번째 A에서의 Return은 전체 남은 Trajectory에 대한 것이다.

$$
G_{t_1}
=
R_{t_1+1}
+
\gamma R_{t_1+2}
+\cdots
$$

두 번째 A에서의 Return은 두 번째 A 이후의 Reward들에 대해서만 계산된다.

$$
G_{t_2}
=
R_{t_2+1}
+
\gamma R_{t_2+2}
+\cdots
$$

따라서 같은 State A에 대한 Return Sample이 두 개 존재한다.

## 4.2 First-Visit MC

First-Visit은 한 Episode에서 **각 State의 첫 번째 방문만** 사용한다.

앞의 예시에서는 다음 방문들이 사용된다.

- 첫 번째 A
- B
- C

두 번째 A는 사용하지 않는다.

여기서 우리가 처음에 혼동했던 부분은 First-Visit이 Episode 전체에서 State 하나만 사용한다는 해석이었다.

그것은 아니다.

**Episode에서 등장한 각각의 State에 대해 첫 방문만 사용하는 것**이다.

## 4.3 Every-Visit MC

Every-Visit은 같은 Episode의 모든 방문을 사용한다.

따라서 앞의 예시에서는 두 번째 A도 사용한다.

A에 대한 누적 Return은 다음처럼 갱신된다.

$$
S(A)
\leftarrow
S(A)+G_{t_1}+G_{t_2}
$$

그리고 Sample Count도 두 번 증가한다.

$$
N(A)\leftarrow N(A)+2
$$

둘의 차이는 오직 **어떤 방문에서 얻은 Return Sample을 평균 계산에 포함하는가**다.

## 4.4 같은 Reward가 중복 반영되는가?

우리가 Every-Visit을 이해하면서 사용했던 중요한 해석이 있다.

첫 번째 A의 Return에는 두 번째 A 이후의 Reward도 포함된다.

두 번째 A의 Return에도 그 이후의 Reward가 포함된다.

따라서 두 번째 A 이후의 Reward가 두 Return에 모두 기여할 수 있다.

하지만 두 Return에서 동일한 Reward에 적용되는 Discount Factor는 다를 수 있다.

첫 번째 A에서 보면 더 먼 미래의 Reward이고, 두 번째 A에서 보면 더 가까운 미래의 Reward이기 때문이다.

즉 단순히 같은 숫자를 두 번 더하는 것이 아니라 **각 방문 시점을 기준으로 서로 다른 Return을 계산하는 것**이다.

---

# 5. Markov Property와 Every-Visit

Every-Visit에서 두 번째 A를 다시 Sample로 사용할 수 있는 이유는 Markov Property와 연결된다.

Markov Property는 현재 State가 주어졌을 때 미래의 분포가 이전 History에 추가로 의존하지 않는다는 의미였다.

따라서 두 번째로 A에 도착했다고 해도, 현재 State가 A라면 그 이후의 Return은 여전히 A의 Value를 추정하는 데 사용할 수 있다.

중요한 전제는 State가 미래 예측에 필요한 정보를 충분히 담고 있어야 한다는 것이다.

즉 **A에 도착하기까지의 과거 경로를 별도로 구분하지 않고도 A의 Value를 추정할 수 있다.**

이 개념은 뒤에서 TD가 다른 State의 Value를 재사용하는 이유와도 연결된다.

---

# 6. Incremental Monte-Carlo

## 6.1 기존의 평균 계산

우리는 State마다 다음 두 값을 유지할 수 있었다.

$$
N(s)=\text{Sample Count}
$$

$$
S(s)=\text{Return Sum}
$$

그리고 Value를 계산했다.

$$
V(s)=\frac{S(s)}{N(s)}
$$

하지만 새로운 Sample이 하나 들어올 때마다 누적합을 유지하지 않고 평균 자체를 갱신할 수 있다.

## 6.2 Incremental Mean

일반적인 평균은 다음처럼 갱신할 수 있다.

$$
\mu_k
=
\mu_{k-1}
+
\frac{1}{k}(x_k-\mu_{k-1})
$$

이것을 MC에 적용하면 다음과 같다.

$$
V(S_t)
\leftarrow
V(S_t)
+
\frac{1}{N(S_t)}
\left[
G_t-V(S_t)
\right]
$$

이 방법은 기존 평균과 새로운 Sample의 차이만 이용한다.

따라서 누적합을 따로 유지할 필요가 없다.

우리가 이해했던 것처럼 기본적으로는 **같은 평균을 계산하는 방식을 효율적으로 바꾼 것**이다.

## 6.3 RL의 공통 Update 형태

이때 매우 중요한 형태가 등장한다.

$$
\text{New Estimate}
=
\text{Old Estimate}
+
\alpha(\text{Target}-\text{Old Estimate})
$$

이후 등장하는 TD, Sarsa, Q-learning 등의 Update에도 같은 구조가 사용된다.

## 6.4 고정 Step Size

정확한 Sample Mean을 계산할 때는 다음을 사용한다.

$$
\alpha=\frac{1}{N(S_t)}
$$

반면 일정한 Step Size를 사용하면 다음과 같다.

$$
V(S_t)
\leftarrow
V(S_t)+\alpha[G_t-V(S_t)]
$$

이 경우에는 최근 경험에 상대적으로 더 큰 비중을 부여한다.

따라서 Non-Stationary Environment에서 환경 변화를 반영하는 데 유용할 수 있다.

---

# 7. Temporal-Difference Learning

## 7.1 MC의 제약

MC에서는 실제 Return을 Target으로 사용했다.

$$
\text{MC Target}=G_t
$$

그러나 이 Return을 알기 위해서는 Episode가 끝나야 한다.

TD는 이 문제를 해결하기 위해 미래 전체를 실제로 관측하지 않는다.

## 7.2 Return의 재귀 관계

Lecture 2에서 배웠던 다음 관계를 다시 생각한다.

$$
G_t
=
R_{t+1}+\gamma G_{t+1}
$$

TD는 여기서 실제 \(G_{t+1}\) 대신 다음 State의 현재 Value Estimate를 사용한다.

$$
G_{t+1}\approx V(S_{t+1})
$$

따라서 TD Target은 다음과 같다.

$$
\text{TD Target}
=
R_{t+1}+\gamma V(S_{t+1})
$$

즉 미래 전체의 실제 Return 대신 일부를 현재 추정된 Value로 대체하는 것이다.

## 7.3 TD(0) Update

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
\right]
$$

여기서 중요한 것은 TD Target이 실제로 관측한 Reward와 기존 Value Estimate로 만들어진다는 점이다.

따라서 Episode가 끝나지 않았더라도 Transition 하나가 끝나면 Update할 수 있다.

---

# 8. V(S_t)와 V(S_{t+1})의 차이

TD를 공부하면서 우리가 혼동했던 중요한 개념이다.

처음에는 Value가 하나의 숫자로 계속 갱신되는 것으로 이해했었다.

하지만 Value Function은 State마다 값을 갖는다.

$$
V(s_1),V(s_2),V(s_3),\ldots
$$

현재 State가 A이고 다음 State가 B라면

$$
S_t=A,\qquad S_{t+1}=B
$$

이다.

이때

$$
V(S_t)=V(A)
$$

이고,

$$
V(S_{t+1})=V(B)
$$

이다.

따라서 TD(0)는 두 State의 Value를 이용해 현재 State의 Value를 업데이트한다.

$$
V(A)
\leftarrow
V(A)+\alpha[R_{t+1}+\gamma V(B)-V(A)]
$$

**업데이트 대상은 V(A)이고, V(B)는 TD Target을 만들기 위한 참고값이다.**

---

# 9. TD Error

TD Error는 다음과 같다.

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

즉 방금 관측한 Transition을 이용해 만든 TD Target과 현재 Value Estimate의 차이다.

이때 TD Error가 반드시 실제 Return과의 차이를 의미하는 것은 아니다.

MC의 Update Error는 다음과 같다.

$$
G_t-V(S_t)
$$

TD Error는 다음과 같다.

$$
R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

우리가 나중에 Eligibility Trace를 공부할 때 다시 확인했듯이, TD Error는 특별한 사건이 발생했을 때만 계산하는 것이 아니다.

**매 Transition마다 계산되는 값이다.**

---

# 10. MC와 TD의 실제 Update 시점

예를 들어 다음 Episode를 생각한다.

$$
A\rightarrow B\rightarrow C\rightarrow D\rightarrow Terminal
$$

MC는 Terminal에 도달한 후 A, B, C, D 각각에 대한 Return을 계산해서 Value를 업데이트한다.

반면 TD는 다음처럼 학습할 수 있다.

- A에서 B로 전이한 직후 V(A) 업데이트
- B에서 C로 전이한 직후 V(B) 업데이트
- C에서 D로 전이한 직후 V(C) 업데이트
- D에서 Terminal로 전이한 직후 V(D) 업데이트

다만 **MC도 한 Episode에서 여러 State의 Value를 업데이트한다**는 것을 기억해야 한다.

TD의 차별점은 한 Episode에서 더 많은 State를 업데이트한다는 것이 아니라, **Episode 종료 전에 학습이 가능하다는 것**이다.

---

# 11. Bootstrapping

TD는 자기 자신의 Value Estimate를 이용해서 다른 Value Estimate를 갱신한다.

이것을 Bootstrapping이라고 한다.

예를 들어

$$
V(A)\leftarrow R+\gamma V(B)
$$

에서 \(V(B)\) 역시 아직 정확하지 않은 추정값일 수 있다.

그럼에도 이를 이용해서 \(V(A)\)를 갱신하는 것이다.

David Silver 강의의 핵심 표현은 다음과 같다.

**Update a guess towards a guess.**

MC는 실제 Return을 Target으로 사용하기 때문에 이러한 Bootstrapping을 하지 않는다.

---

# 12. MC와 TD의 Bias-Variance Trade-off

## 12.1 Monte-Carlo

MC는 실제 Return을 사용한다.

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+\cdots
$$

이 Return은 주어진 State에서의 실제 Value에 대한 Unbiased Sample이다.

$$
E_\pi[G_t|S_t=s]=v^\pi(s)
$$

하지만 미래의 Action, Transition, Reward에서 발생하는 Randomness가 모두 포함된다.

따라서 Variance가 클 수 있다.

**MC: Unbiased Target, High Variance**

## 12.2 Temporal-Difference

TD는 다음 Target을 사용한다.

$$
R_{t+1}+\gamma V(S_{t+1})
$$

미래 전체의 Randomness를 직접 Sample하지 않으므로 일반적으로 Variance가 작다.

하지만 \(V(S_{t+1})\)가 잘못된 추정값일 수 있기 때문에 Target에 Bias가 생길 수 있다.

**TD: Potentially Biased Target, Lower Variance**

우리가 이해했던 표현으로 정리하면 다음과 같다.

MC는 실제로 끝까지 진행한 결과를 사용한다. 각 Sample은 많이 흔들릴 수 있지만 평균적으로 올바른 Value를 가리킨다.

TD는 미래의 일부를 현재 만들어진 평균적 추정값으로 대신한다. 따라서 Randomness는 줄지만, 그 추정값 자체가 틀렸을 가능성이 있다.

---

# 13. Driving Home Example

David Silver는 집에 가는 데 걸리는 시간을 예측하는 예제를 사용한다.

출발할 때는 30분이 걸릴 것이라고 예측했다.

하지만 5분 후 차에 도착했더니 비가 오고 있었고, 앞으로 35분 더 걸릴 것으로 예측했다.

이때 TD는 출발 당시의 예측을 다음 값으로 수정할 수 있다.

$$
5+35=40
$$

즉 실제 도착하기 전부터 기존의 30분 예측을 40분 방향으로 수정할 수 있다.

반면 MC는 집에 실제로 도착해서 최종 소요 시간이 43분이라는 사실을 확인한 뒤 업데이트한다.

**MC는 최종 Outcome을 보고 학습한다.**

**TD는 다음 State에서 만들어진 Prediction을 보고 중간에 학습한다.**

---

# 14. Random Walk Example

강의에서는 일렬로 나열된 State 사이를 무작위로 움직이는 Random Walk 환경을 사용한다.

왼쪽 Terminal에서는 Reward 0, 오른쪽 Terminal에서는 Reward 1을 받는다.

중간 State들은 A, B, C, D, E로 구성된다.

Agent는 왼쪽이나 오른쪽으로 무작위 이동한다.

오른쪽 Terminal에 가까울수록 Value가 높아진다.

이 환경에서 MC와 TD를 비교하면 TD가 보통 더 빠르게 True Value에 접근하는 모습을 볼 수 있다.

강의의 중요한 메시지는 TD가 **Markov Structure를 이용해 Experience를 효율적으로 활용할 수 있다**는 것이다.

그러나 TD가 모든 문제에서 MC보다 항상 우월하다는 뜻은 아니다.

---

# 15. Batch MC vs Batch TD

## 15.1 Finite Experience의 문제

MC와 TD는 충분히 많은 Experience를 얻으면 모두 True Value로 수렴할 수 있다.

하지만 한정된 Experience만 주어졌을 때는 서로 다른 해에 도달할 수 있다.

강의에서는 A와 B 두 State를 사용한 예제로 설명한다.

총 8개의 Episode가 있고, 그중 하나는 다음과 같다.

$$
A\xrightarrow{0}B\xrightarrow{0}Terminal
$$

나머지 7개는 B에서 시작해 Terminal로 전이한다.

B에서의 Reward를 모두 합치면 총 6이고, B의 전체 방문은 8번이다.

따라서 경험적으로 다음과 같다.

$$
V(B)=\frac68=0.75
$$

## 15.2 Batch MC의 결과

A에서 시작한 Episode는 단 하나이고, 그 Return은 0이다.

따라서 Batch MC는 다음 값을 선택한다.

$$
V(A)=0
$$

즉 실제로 관측된 A의 Return을 가장 잘 맞춘다.

## 15.3 Batch TD의 결과

TD는 A에서 B로 전이했다는 사실을 이용한다.

그리고 B에서 얻은 많은 경험을 통해 다음 값을 학습했다.

$$
V(B)=0.75
$$

따라서 A의 Value도 Bellman 관계에 따라 다음처럼 계산한다.

$$
V(A)=0+\gamma V(B)
$$

이 예제에서는 \(\gamma=1\)이므로

$$
V(A)=0.75
$$

가 된다.

---

# 16. TD가 Markov Property를 활용하는 이유

여기서 우리가 중요하게 이해했던 부분이 있다.

> 왜 B에서 학습한 Value를 A로 Propagate할 수 있는가?

Markov Property에 의하면 현재 State가 B일 때, B 이후의 미래 분포를 결정하기 위해 과거 State가 A였는지 C였는지를 추가로 알 필요가 없다.

따라서

$$
V(B)
$$

는 B 이후의 미래를 요약하는 재사용 가능한 정보다.

B에 대한 경험이 많다면 A에서 B로 한 번 전이한 경험만 있어도 그 정보를 이용할 수 있다.

즉

$$
V(A)\approx R+\gamma V(B)
$$

로 B의 Value를 A에 Propagate할 수 있다.

우리가 이해한 가장 중요한 표현은 다음과 같았다.

**Markov Property가 있으므로, State B의 Value는 이전 경로와 무관하게 재사용 가능한 미래 정보다.**

이 때문에 TD는 Markov 환경에서 효율적으로 작동할 수 있다.

---

# 17. Sampling과 Bootstrapping으로 보는 MC, TD, DP

David Silver는 세 알고리즘을 두 가지 기준으로 비교한다.

**Sampling:** Expectation을 실제 Sample로 추정하는가?

**Bootstrapping:** 기존 Value Estimate를 이용해 새로운 Value Estimate를 업데이트하는가?

| 알고리즘 | Sampling | Bootstrapping |
|---|---|---|
| Dynamic Programming | X | O |
| Monte-Carlo | O | X |
| Temporal-Difference | O | O |

DP는 환경 모델로 Expectation을 계산하고, 다음 State의 Value Estimate를 사용한다.

MC는 실제 Experience를 사용하지만 기존 Value Estimate로 Bootstrap하지 않는다.

TD는 실제 Experience를 사용하고, 다음 State의 Value Estimate로 Bootstrap한다.

따라서 TD는 MC와 DP의 특징을 하나씩 결합한 형태라고 볼 수 있다.

---

# 18. n-step TD

## 18.1 TD(0)에서 출발

TD(0)는 한 Step의 실제 Reward를 확인한 뒤 Bootstrap한다.

$$
G_t^{(1)}
=
R_{t+1}+\gamma V(S_{t+1})
$$

그런데 반드시 한 Step 뒤에서 Bootstrap해야 하는 것은 아니다.

## 18.2 2-step TD

두 Step의 Reward를 실제로 관측한다.

$$
G_t^{(2)}
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2V(S_{t+2})
$$

## 18.3 3-step TD

$$
G_t^{(3)}
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+
\gamma^3V(S_{t+3})
$$

## 18.4 일반화

$$
G_t^{(n)}
=
\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+
\gamma^nV(S_{t+n})
$$

이 값이 n-step Return이다.

Update는 다음과 같다.

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
G_t^{(n)}-V(S_t)
\right]
$$

단, Episode가 n Step보다 먼저 종료되면 실제 종료 시점까지의 Reward만 사용하고 Terminal 이후의 Bootstrap 항은 제거한다.

---

# 19. TD와 MC 사이의 연속적인 구조

n이 작으면 실제 Reward를 짧게 관측하고 빨리 Bootstrap한다.

n이 커지면 실제 Reward를 더 길게 관측하고 Bootstrap을 늦춘다.

정리하면:

- 1-step Return: TD(0)
- 2-step Return: 2-step TD
- 3-step Return: 3-step TD
- Episode 끝까지의 Return: MC

즉 TD와 MC는 완전히 분리된 것이 아니라 **Bootstrap을 어느 시점에 수행하는가**에 따라 연결된다.

---

# 20. TD(λ) — Forward View

## 20.1 왜 λ가 필요한가?

n-step TD에서는 n을 직접 선택해야 한다.

하지만 반드시 하나의 n만 선택할 필요는 없다.

여러 n-step Return을 가중평균할 수 있다.

이것이 TD(λ)의 Forward View다.

## 20.2 λ-return

일반적인 표현은 다음과 같다.

$$
G_t^\lambda
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}G_t^{(n)}
$$

여기서 \(0\le\lambda<1\)이다.

각 Return에 주어지는 가중치는 다음과 같다.

| Return | Weight |
|---|---|
| 1-step | \(1-\lambda\) |
| 2-step | \((1-\lambda)\lambda\) |
| 3-step | \((1-\lambda)\lambda^2\) |
| 4-step | \((1-\lambda)\lambda^3\) |

가중치 합은 1이다.

$$
(1-\lambda)
\sum_{n=0}^{\infty}\lambda^n=1
$$

따라서 여러 n-step Return의 Weighted Average가 된다.

## 20.3 Episodic Environment에서의 정확한 표현

Episode가 시점 T에서 종료된다면 n-step Return은 남은 Episode 길이에 의해 제한된다.

이 경우 Forward View의 λ-return은 다음처럼 표현할 수 있다.

$$
G_t^\lambda
=
(1-\lambda)
\sum_{n=1}^{T-t-1}
\lambda^{n-1}G_t^{(n)}
+
\lambda^{T-t-1}G_t
$$

마지막 항은 Episode 종료까지 실제로 관측한 Full Return이다.

이 표현에서는 \(\lambda=1\)을 직접 대입하면 MC Return이 된다.

## 20.4 λ의 의미

λ가 작으면 짧은 n-step Return에 큰 가중치를 부여한다.

λ가 크면 더 긴 n-step Return의 영향이 커진다.

즉 λ는 **얼마나 멀리 실제 Reward를 관측하고, 언제 Bootstrap할지에 대한 가중치를 조절하는 파라미터**다.

---

# 21. Credit Assignment Problem

TD(λ)의 Backward View로 넘어가면서 Credit Assignment Problem이 등장한다.

문제는 다음과 같다.

> 지금 좋은 Reward나 큰 TD Error가 관측되었다면, 과거에 방문했던 어떤 State들이 그 결과에 얼마나 기여했는가?

예를 들어 다음 Trajectory를 생각한다.

$$
A\rightarrow B\rightarrow C\rightarrow D
$$

D로 전이하는 순간 큰 Positive TD Error가 발생했다고 하자.

TD(0)는 그 직전의 State만 업데이트한다.

하지만 더 이전에 방문했던 A와 B도 최종 결과에 영향을 주었을 수 있다.

그렇다면 현재의 학습 신호를 과거 State들에게도 전달하는 방법이 필요하다.

그 방법이 Eligibility Trace다.

---

# 22. Eligibility Trace

## 22.1 State별 Trace를 유지한다

각 State마다 Eligibility Trace를 하나씩 유지한다.

$$
E_t(s)
$$

이 값은 현재 시점에 발생한 TD Error를 State s에 얼마나 반영할지 결정하는 가중치다.

처음에는 모든 Trace를 0으로 둔다.

$$
E_{\text{initial}}(s)=0
$$

그리고 매 Step 다음과 같이 갱신한다.

$$
E_t(s)
=
\gamma\lambda E_{t-1}(s)
+
\mathbf{1}(S_t=s)
$$

## 22.2 수식의 의미

첫 번째 항은 기존 Trace를 감소시킨다.

$$
\gamma\lambda E_{t-1}(s)
$$

두 번째 항은 현재 State를 방문한 경우 1을 추가한다.

$$
\mathbf{1}(S_t=s)
=
\begin{cases}
1 & S_t=s\\
0 & S_t\ne s
\end{cases}
$$

따라서 최근에 방문한 State는 높은 Trace를 가지며, 오래전에 방문한 State의 Trace는 점차 감소한다.

같은 State를 반복 방문했다면 기존 Trace 위에 값이 추가된다.

이것이 강의에서 설명하는 **Recency Heuristic과 Frequency Heuristic의 결합**이다.

---

# 23. Eligibility Trace 계산 예시

다음과 같이 가정한다.

$$
\gamma\lambda=0.8
$$

그리고

$$
A\rightarrow B\rightarrow C
$$

순서로 방문한다.

초기 상태에서는 모든 Trace가 0이다.

A를 방문한 시점:

$$
E(A)=1
$$

B를 방문한 시점:

$$
E(A)=0.8,\qquad E(B)=1
$$

C를 방문한 시점:

$$
E(A)=0.64,\qquad E(B)=0.8,\qquad E(C)=1
$$

따라서 C에서 TD Error가 계산되면 A, B, C에 서로 다른 비율로 학습 신호를 전달할 수 있다.

---

# 24. TD(λ) — Backward View

## 24.1 TD Error 계산

매 Step 다음 값을 계산한다.

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

## 24.2 과거 State들의 Value 업데이트

TD(0)에서는 현재 State만 업데이트한다.

반면 Backward View TD(λ)는 모든 State에 대해 다음 Update를 적용한다.

$$
V(s)
\leftarrow
V(s)
+
\alpha\delta_tE_t(s)
$$

Trace가 0인 State는 업데이트되지 않고, Trace가 클수록 현재 TD Error의 영향을 크게 받는다.

이것이 Backward View의 핵심이다.

**현재 발생한 TD Error를 과거 State들에게 Eligibility Trace에 비례해 분배한다.**

---

# 25. TD Error가 발생한다는 표현의 의미

우리가 중간에 질문했던 부분이다.

> TD Error는 당연히 있을 수밖에 없는데, 왜 “TD Error가 발생하면”이라고 표현하는가?

정확히는 **TD Error를 매 Step마다 계산한다**고 표현해야 한다.

TD Error는 특별한 사건이 아니라 현재 Value Estimate와 TD Target의 차이다.

따라서 매 Transition마다 계산할 수 있고, 그 값이 양수일 수도, 음수일 수도, 0일 수도 있다.

Backward View에서는 이 값을 Eligibility Trace에 따라 과거 State들에게 전달한다.

---

# 26. Forward View와 Backward View의 관계

## 26.1 Forward View

Forward View는 현재 State를 기준으로 미래의 여러 Return을 살펴본다.

$$
G_t^{(1)},G_t^{(2)},G_t^{(3)},\ldots
$$

그리고 이 Return들을 가중평균한다.

$$
G_t^\lambda
$$

즉 **미래의 Return 정보를 현재 State의 학습 Target으로 가져오는 관점**이다.

## 26.2 Backward View

Backward View는 매 Step 발생하는 TD Error를 과거 State들에게 분배한다.

$$
V(s)
\leftarrow
V(s)+\alpha\delta_tE_t(s)
$$

즉 **현재의 학습 신호를 과거 State들에게 전달하는 관점**이다.

## 26.3 두 관점의 Equivalence

Lecture 4에서는 두 관점이 Episode 전체에 걸쳐 동일한 총 Update로 연결될 수 있음을 설명한다.

다만 이 동등성은 고정된 Value Estimate를 사용하는 Offline 계산에서 정확하게 성립하는 형태이며, 일반적인 Online Backward View에서는 학습 도중 Value가 변하기 때문에 Forward View와 차이가 발생할 수 있다.

따라서 두 관점은 다음처럼 기억하는 것이 좋다.

**Forward View:** 여러 n-step Return을 결합하는 이론적 관점

**Backward View:** Eligibility Trace를 이용해 Online Update를 수행하는 구현 관점

---

# 27. λ=0과 λ=1

## 27.1 λ=0

λ가 0이면 1-step Return에만 가중치가 부여된다.

$$
G_t^0=G_t^{(1)}
$$

따라서 TD(0)가 된다.

Backward View에서도 기존 Trace가 사라져 현재 방문한 State만 Update된다.

$$
E_t(s)=\mathbf{1}(S_t=s)
$$

즉

**λ=0 → TD(0)**

## 27.2 λ=1

λ가 1이면 Episodic Forward View에서 전체 Episode Return이 사용된다.

$$
G_t^1=G_t
$$

따라서 MC와 연결된다.

Backward View의 경우에도 Episode 전체에서의 총 Update가 MC Return을 이용한 Update와 연결된다. 단, Online 구현에서 매 Step Value가 변하면 정확한 동등성에 주의해야 한다.

즉

**λ=1 → Monte-Carlo Return**

---

# 28. Lecture 4 전체 알고리즘 비교

| 알고리즘 | Target | Update 시점 | Bootstrapping |
|---|---|---|---|
| MC | Full Return | Episode 종료 후 | X |
| TD(0) | 1-step Return | 매 Step | O |
| n-step TD | n-step Return | n-step 관측 후 | O (종료 전) |
| TD(λ) Forward | λ-return | 미래 Return 필요 | 혼합 |
| TD(λ) Backward | TD Error와 Trace | 매 Step | O |

여기서 MC부터 TD(λ)까지 학습 목표는 동일하다.

$$
v^\pi(s)
$$

즉 현재 Policy를 따랐을 때의 State Value Function을 추정한다.

차이는 **그 Value를 어떤 Target과 어떤 Update 방식으로 학습하는가**다.

---

# 29. Lecture 2, 3, 4의 연결

## Lecture 2 — What to Estimate?

Lecture 2에서는 다음 값을 정의했다.

$$
v^\pi(s)=E_\pi[G_t|S_t=s]
$$

즉 우리가 구하려는 값이 무엇인지를 수학적으로 정의했다.

## Lecture 3 — Planning with a Known Model

Lecture 3에서는 \(P,R\)을 알고 있기 때문에 Bellman Equation으로 Value를 계산했다.

대표적으로 Iterative Policy Evaluation은 다음과 같다.

$$
v_{k+1}(s)
=
\sum_a\pi(a|s)
\left[
R_s^a+
\gamma\sum_{s'}P_{ss'}^a v_k(s')
\right]
$$

여기서는 실제 Agent Experience를 Sample하지 않아도 된다.

## Lecture 4 — Learning without a Model

Lecture 4에서는 \(P,R\)을 모르므로 실제 Experience로 Value를 학습했다.

MC에서는 다음과 같다.

$$
V(S_t)
\leftarrow
V(S_t)+\alpha[G_t-V(S_t)]
$$

TD에서는 다음과 같다.

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
[
R_{t+1}+\gamma V(S_{t+1})-V(S_t)
]
$$

즉 Lecture 2에서 정의했던 \(v^\pi\)를 Lecture 3에서는 Model로 계산하고, Lecture 4에서는 Experience로 추정한 것이다.

---

# 30. 최종 Remind

Lecture 4를 다시 복습할 때 가장 먼저 떠올릴 것은 다음 세 가지다.

## 첫 번째 — Monte-Carlo

$$
V(S_t)
\leftarrow
V(S_t)+\alpha[G_t-V(S_t)]
$$

실제로 Episode를 끝까지 진행한 Return을 이용해 Value를 추정한다.

**미래를 끝까지 실제로 보고 학습한다.**

## 두 번째 — Temporal-Difference

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
[
R_{t+1}+\gamma V(S_{t+1})-V(S_t)
]
$$

미래의 일부를 기존 Value Estimate로 대체한다.

**한 Step만 실제로 보고 나머지는 현재 추정된 Value로 Bootstrap한다.**

## 세 번째 — TD(λ)

$$
G_t^\lambda
=
(1-\lambda)
\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

여러 n-step Return을 가중평균한다.

그리고 Backward View에서는 Eligibility Trace를 사용한다.

$$
E_t(s)
=
\gamma\lambda E_{t-1}(s)+\mathbf{1}(S_t=s)
$$

$$
V(s)\leftarrow V(s)+\alpha\delta_tE_t(s)
$$

**현재의 TD Error를 이전에 방문했던 State들에게 분배해 학습한다.**

---

# Lecture 4의 가장 중요한 결론

Lecture 4의 핵심은 결국 다음과 같다.

**Environment Model을 모르는 상태에서, 어떻게 Value Function을 학습할 것인가?**

MC는 실제 Return의 Sample Average를 사용한다.

TD는 실제 Reward와 기존 Value Estimate를 결합한다.

n-step TD는 실제 Reward를 얼마나 멀리까지 볼지 결정한다.

TD(λ)는 여러 n-step Return을 결합하고, Eligibility Trace를 통해 과거 State들에 학습 신호를 전달한다.

따라서 전체 흐름은 다음과 같다.

**MC → TD(0) → n-step TD → TD(λ)**

이것이 Lecture 4 — Model-Free Prediction의 중심 구조다.

다음 Lecture 5에서는 단순히 Policy의 Value를 추정하는 Prediction을 넘어, Experience를 이용해 Policy 자체를 개선하는 **Model-Free Control**을 배우게 된다.