# Lecture 2 — Video Models
## 장기 복습용 Remind Note

> 목표: 슬라이드를 단순히 압축하는 것이 아니라,  
> **강의의 흐름 + 교수님이 실제로 강조한 포인트 + 내가 질문하면서 이해한 방식 + 헷갈렸던 개념의 정정**까지 함께 남긴다.
>
> 특히 이번 강의에서는 `temporal layer`, `mapping network`, `regularization`, `VAE latent`, `3D attention`, `conditioning`, `trajectory`, `open/closed loop`와 함께,  
> 내가 따로 정리한 **attention / self-attention / cross-attention의 기본 개념**을 복습 포인트로 포함한다.

---

# 0. 이 강의는 무엇을 설명하려는가?

이번 강의의 큰 흐름은 두 덩어리다.

1. **이미지 생성 모델을 어떻게 비디오 생성 모델로 확장했는가**
2. **비디오를 사용자가 원하는 방식으로 어떻게 제어할 것인가**

즉:

```text
Image Generation
    ↓
Video Generation
    ↓
Video-native Architecture
    ↓
Conditioning / Control
    ↓
Action / Closed-loop
    ↓
World Model로 연결
```

이번 강의는 수식 전개보다는 **모델 설계의 변화와 conditioning 방식**이 중심이다.

---

# 1. 3장 — Generating images vs videos

이미지는 기본적으로:

\[
H \times W \times C
\]

비디오는 시간축이 추가되어:

\[
T \times H \times W \times C
\]

가 된다.

예를 들어 30fps 비디오는 1초만 해도 이미지 30장이다.

따라서 비디오 모델은 이미지 모델보다:

- 연산량이 훨씬 크고
- 데이터가 훨씬 많이 필요하며
- 시간축의 일관성까지 유지해야 한다

는 문제가 생긴다.

## 핵심

\[
\boxed{\text{Video generation}=\text{image generation}+\text{temporal consistency}}
\]

단순히 프레임마다 좋은 이미지를 만드는 것으로는 부족하다.

---

# 2. 4장 / 8장 — Inflating text-to-image model

초기 비디오 생성 연구의 중요한 아이디어:

> **이미 잘 학습된 T2I 모델은 최대한 그대로 두고, 시간축을 처리하는 layer만 추가하자.**

구조는 대략:

```text
기존 image block
      ↓
temporal layer
      ↓
기존 image block
      ↓
temporal layer
      ↓
...
```

기존 image model은 freeze하고, 새로 추가한 temporal layer만 학습한다.

---

## 2.1 Temporal layer를 내가 이해한 방식

기존 image model은 각 frame을 사실상 독립적으로 처리한다.

예:

\[
h_1,\;h_2,\;h_3
\]

가 각각 frame 1, 2, 3의 중간 feature라면,

기존 image model에서는 서로 직접 정보를 주고받지 않는다.

Temporal layer는:

\[
(h_1,h_2,h_3)
\rightarrow
(\tilde h_1,\tilde h_2,\tilde h_3)
\]

처럼 **여러 frame의 feature를 함께 보고 서로 영향을 주도록 만드는 layer**다.

즉:

> 기존 image block은 한 frame 내부의 공간 구조를 잘 만들고,  
> temporal layer는 여러 frame 사이의 시간 관계를 반영한다.

구현은 꼭 하나가 아니며:

- temporal convolution
- temporal self-attention
- 기타 frame interaction module

등이 가능하다.

## 내가 중간에 헷갈린 포인트

Temporal layer가 많아질수록:

\[
h \rightarrow \tilde h
\]

가 반복되고, 결국 기존 image model이 원래 잘 다루던 hidden feature distribution에서 멀어질 수 있다.

즉:

> **시간적으로는 잘 이어지는데, 한 frame 한 frame의 image quality나 semantic이 망가질 수 있지 않나?**

이 걱정이 맞고, 그래서 뒤에서 regularization loss가 등장한다.

---

# 3. 9장 — Mapping Network

초기 비디오 모델에서는 frame마다 random noise를 쓰게 되는데, 서로 독립적으로 뽑으면 인접 frame의 noise가 매우 다를 수 있다.

이미지 모델은 같은 class condition을 받아도 서로 다른 noise를 넣으면 서로 다른 이미지를 만들 수 있다.

하지만 비디오에서 인접 frame은 단순히 “같은 Husky class” 정도가 아니라:

- 같은 개체
- 같은 배경
- 거의 같은 조명
- 조금씩 변하는 pose/motion

정도의 **강한 temporal similarity**가 필요하다.

그래서 mapping network를 둔다.

```text
raw video noise
      ↓
Mapping Network
      ↓
image model이 다루기 좋은 latent/noise 구조
```

핵심:

\[
\boxed{\text{독립적인 video noise를, temporal continuity를 만들기 쉬운 latent로 보정}}
\]

강의에서는 mapping network 내부의 정확한 수학적 transformation까지 깊게 다루지는 않는다.

---

# 4. 10장 — Training

기본적으로 두 방향의 loss가 나온다.

---

## 4.1 Denoising loss

기존 diffusion image model처럼:

> noisy input에서 signal/noise를 잘 복원하도록 학습

즉 비디오 생성 자체를 잘하도록 하는 기본 loss다.

---

## 4.2 Regularization loss

여기서 중요한 연결:

Temporal layer를 학습하면, 기존 image model의 hidden feature가 변한다.

예:

\[
h \rightarrow \tilde h
\]

다음 frozen image block은 원래 \(h\)를 보던 모델인데, 이제 \(\tilde h\)를 받는다.

Temporal layer가 너무 강하게 개입하면:

- motion은 생기지만
- 기존 image prior가 무너지고
- frame quality나 semantic이 망가질 수 있다

따라서:

> **Temporal modeling은 하되, 기존 image model의 능력에서 너무 멀리 가지 말자**

라는 제약을 건다.

개념적으로:

\[
L
=
L_{\text{denoise}}
+
\lambda L_{\text{reg}}
\]

와 같은 trade-off다.

핵심:

\[
\boxed{\text{Denoising loss}=\text{video를 잘 만들게}}
\]

\[
\boxed{\text{Regularization loss}=\text{기존 image prior에서 너무 멀어지지 않게}}
\]

Regularization은 완전한 보장이 아니라 **타협책**이다.

---

# 5. 11장 — More losses: smoothness

비디오의 인접 frame은 서로 비슷해야 한다.

여기서는 단순히 output pixel을 비교하는 게 아니라:

> **인접 frame의 self-attention map이 비슷해야 한다**

는 prior를 loss로 넣는다.

---

# 6. Attention / Self-Attention 기본 정리

이번 강의를 이해하면서 attention을 다시 정리할 필요가 있었다.

입력:

\[
a_1,a_2,a_3
\]

각 입력에서:

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

를 만든다.

---

## 6.1 Q, K, V의 직관

### Value \(V\)

실제로 가중합될 **정보/content**다.

\[
v_i=W_Va_i
\]

내가 이해한 표현:

> \(a_i\)를 attention에서 실제 전달할 정보 형태로 projection한 것.

---

### Query \(Q\)

현재 token이:

> **“나는 어떤 정보를 찾고 있는가?”**

를 표현하는 vector.

---

### Key \(K\)

각 token이:

> **“나는 이런 정보다”**

라고 자신을 매칭하기 위한 vector.

---

## 6.2 Attention weight

각 query와 key 사이의 dot product:

\[
q_i^\top k_j
\]

를 계산하고 softmax:

\[
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)
\]

여기서 \(A\)가 바로 **attention map / attention weight matrix**다.

즉:

\[
A_{ij}
\]

는:

> token \(i\)가 token \(j\)를 얼마나 참고하는가

를 나타낸다.

---

## 6.3 Attention output

\[
O=AV
\]

각 output:

\[
o_i=\sum_j A_{ij}v_j
\]

즉:

> Q/K로 “누구를 얼마나 볼지” 정하고, 실제 내용은 V에서 가져온다.

내가 최종적으로 이해한 attention:

\[
\boxed{
\text{Q,K로 weight 계산}
\rightarrow
\text{그 weight로 V 가중합}
\rightarrow
\text{새 latent 생성}
}
\]

---

# 7. Self-Attention이란?

Q, K, V가 **같은 입력 집합 \(X\)** 에서 나온다.

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

즉:

> 같은 token 집합 안에서 서로를 참고한다.

예를 들어 video frame 내부의 patch token들이 서로를 참고하는 경우.

---

# 8. 11장의 Self-Attention Map은 무엇인가?

11장에서 말하는 self-attention map은 새로운 latent가 아니다.

\[
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)
\]

이 **weight matrix 자체**다.

즉:

- \(Q,K,V\): attention 계산에 사용되는 vector
- \(A\): attention map
- \(O=AV\): attention 적용 후 output feature

11장에서는 인접 frame \(j-1,j\)에 대해:

\[
A_{j-1}\approx A_j
\]

가 되도록 loss를 건다.

의미:

> 인접 frame에서 모델이 “어디를 얼마나 참고하는가”라는 관계 구조도 비슷했으면 좋겠다.

---

# 9. 12장 — Contrastive loss

같은 비디오 안의 frame들은 같은 object/scene semantics를 공유할 가능성이 높다.

따라서:

\[
\text{same video frame pair}
\rightarrow
\text{feature similarity 높게}
\]

다른 비디오끼리는:

\[
\text{different video}
\rightarrow
\text{feature similarity 낮게}
\]

학습한다.

여기서 feature는 **VAE latent가 아니라 neural network 내부의 hidden representation**에 가깝다.

강의에서는 softmax + cross entropy 기반 contrastive learning을 설명하고, CLIP과 같은 계열의 학습을 연결해서 언급한다.

---

# 10. 13장 — Frame-wise text tokens

같은 text condition을 모든 frame에 똑같이 넣으면, 기존 image model 관점에서는 비슷한 output을 반복하려는 경향이 생길 수 있다.

예:

\[
e_{\text{text}}
\]

하나만 모든 frame에 똑같이 주는 대신:

\[
e_{\text{text}}+P_1
\]

\[
e_{\text{text}}+P_2
\]

\[
e_{\text{text}}+P_3
\]

처럼 frame별 positional-like vector를 추가한다.

이건 LLM의 positional embedding과 비슷한 발상이다.

차이:

- LLM positional embedding: token 순서
- frame-wise text token: frame index / temporal freedom

핵심:

\[
\boxed{\text{같은 prompt라도 frame마다 약간 다른 conditioning을 줘 motion freedom을 만든다}}
\]

---

# 11. Cross-Attention 정리

Self-attention과 계산 방식은 비슷하지만 Q와 K/V가 다른 source에서 온다.

예를 들어 video가 text를 참고:

\[
Q=\text{video tokens}
\]

\[
K,V=\text{text tokens}
\]

출력:

\[
O=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V
\]

여기서 output은 수학적으로 text value의 가중합이다.

그런데 query가 video token마다 있으므로:

\[
o_i
\]

는:

> **video token \(i\)가 필요로 하는 text 정보**

가 된다.

실제 block에서는 보통 residual을 통해 기존 video latent와 결합된다.

\[
h_i' = h_i + o_i
\]

따라서 결과적으로:

> video latent가 text condition을 반영해서 업데이트된다.

---

# 12. 15장 — References of inflation

초기 비디오 모델 발전 흐름:

- Video Diffusion Models: 3D U-Net, space/time factorization
- Imagen Video / Make-A-Video: cascade
- Align Your Latents: image LDM freeze + temporal layer
- Stable Video Diffusion: data curation 강조
- AnimateDiff: transferable motion module
- HARIVO: mapping network + temporal inductive bias

핵심:

> 초기 연구는 새 비디오 모델을 처음부터 만드는 대신, 강한 image model을 재사용하려 했다.

---

# 13. 16장 — Limitations

Inflated image model 방식의 한계 세 가지.

### 1. 2D VAE
시간축 중복을 압축하지 못한다.

### 2. Factorized attention
공간축과 시간축을 따로 보면, 시간과 공간이 동시에 변하는 motion relation을 직접 연결하기 어렵다.

### 3. Frozen image prior
기존 image backbone이 너무 강하게 남아 있어서 큰 motion을 만들기 어렵다.

이 세 문제 때문에 이후에는 **video-native architecture**로 넘어간다.

---

# 14. 19장 — Axis 1: The latent

## 14.1 Image model의 VAE

고해상도 image를 pixel space에서 바로 diffusion/flow model로 처리하면 비싸다.

따라서:

\[
\text{image}
\rightarrow
\text{VAE encoder}
\rightarrow
z
\]

작은 latent space에서 생성하고:

\[
z
\rightarrow
\text{VAE decoder}
\rightarrow
\text{image}
\]

로 복원한다.

---

## 14.2 2D VAE가 비디오에서 비효율적인 이유

frame-wise 2D VAE:

\[
F_1\to z_1,\quad
F_2\to z_2,\quad
F_3\to z_3
\]

하지만 인접 frame은 대부분 정보가 중복된다.

그래서 video VAE는 여러 frame을 함께 받아:

\[
(F_1,F_2,F_3,F_4)
\rightarrow
z_1
\]

처럼 **시간축까지 함께 압축**한다.

중요:

> 여러 frame을 압축한다고 해서 전체 비디오가 vector 하나가 되는 건 아니다.

출력은 여전히:

\[
Z\in\mathbb R^{T'\times H'\times W'\times C}
\]

와 같은 시공간 구조를 가진다.

다만:

\[
T'<T,\quad H'<H,\quad W'<W
\]

로 압축되어 있을 뿐이다.

---

# 15. 20장 — Axis 2: Architecture

Video VAE의 latent:

\[
T'\times H'\times W'\times C
\]

를 Transformer에 넣기 위해 patchify한다.

---

## 15.1 3D Patchify

latent 전체를 token 하나로 만드는 것이 아니다.

작은 시공간 영역:

\[
P_T\times P_H\times P_W
\]

으로 나눈다.

각 위치마다 channel \(C\)개가 있으므로 patch 하나에는:

\[
P_TP_HP_WC
\]

개의 값이 있다.

이를 flatten + linear projection해서 token 하나로 만든다.

결과:

\[
T'\times H'\times W'\times C
\rightarrow
N\times D
\]

즉:

\[
\boxed{\text{3D video latent 하나} \rightarrow \text{여러 spatiotemporal patch} \rightarrow \text{여러 token}}
\]

---

## 15.2 Full 3D Attention

각 token은 비디오의 작은 시공간 patch를 표현한다.

예:

```text
token 1 = t=0, 좌상단 patch
token 2 = t=0, 중앙 patch
...
token 20 = t=1, 다른 위치 patch
```

Full 3D attention에서는 이 모든 token이 서로 직접 attention할 수 있다.

즉:

\[
\boxed{\text{공간+시간 전체 token 사이 self-attention}}
\]

초기 factorized attention은 공간과 시간을 따로 봤지만, full 3D attention은 시간/공간이 함께 변하는 motion relation도 직접 연결할 수 있다.

---

# 16. 21장 — Axis 3: Conditioning Architecture

condition을 video transformer에 넣는 대표 방식 두 가지.

---

## 16.1 Cross-Attention

\[
Q=\text{video}
\]

\[
K,V=\text{text}
\]

video token이 text condition을 참고한다.

---

## 16.2 Token concatenation / in-context

```text
[text tokens][image tokens][video tokens]
```

을 하나의 sequence로 넣고 self-attention한다.

LLM이 여러 종류의 token을 한 sequence에 넣는 것과 비슷하다.

강의에서는 어느 방식이 완전히 정답으로 정착된 건 아니라고 설명한다.

---

# 17. 13장과 21장의 관계

이 둘이 처음에 헷갈렸다.

### 13장
HARIVO라는 특정 inflated model에서:

> frame마다 text condition을 조금씩 다르게 만드는 구체적 trick

### 21장
video-native architecture 전체 관점에서:

> condition을 Transformer에 어떤 구조로 넣을 것인가?

즉:

\[
\boxed{\text{13장}=\text{특정 conditioning trick}}
\]

\[
\boxed{\text{21장}=\text{conditioning architecture의 일반적인 설계 선택}}
\]

---

# 18. 22장 — Axis 4: Data

비디오 모델에서는 데이터가 매우 중요하다.

필요한 것:

- 좋은 비디오 수집
- 좋은 caption
- filtering
- watermark 제거
- 이상한 transition 제거
- motion이 충분한 영상
- aesthetic quality

---

## 18.1 Recaptioning

사람이 모든 비디오에 caption을 쓰는 건 너무 비싸다.

그래서 VLM을 사용해서:

\[
\text{video}
\rightarrow
\text{VLM}
\rightarrow
\text{detailed caption}
\]

을 만든다.

모델은 결국 **이 caption distribution**을 학습한다.

그래서 뒤에서 prompt rewriter가 필요해진다.

---

# 19. Part 2 — Describing the video

강의 후반부의 핵심 질문:

> **사용자가 원하는 비디오를 만들도록 무엇을, 언제 condition으로 줄 것인가?**

두 축:

\[
\boxed{\text{What}}
\]

어떤 정보를 줄 것인가?

\[
\boxed{\text{When}}
\]

언제 줄 것인가?

---

# 20. 27장 — Text

사용자는 짧게:

> `"a cat on a sofa"`

라고 쓰지만, 실제 학습 데이터의 VLM caption은 훨씬 자세할 수 있다.

그래서:

```text
short user prompt
     ↓
Prompt Rewriter
     ↓
training caption style과 비슷한 상세 prompt
```

로 바꿔준다.

Prompt Rewriter를 내가 이해한 방식:

> **사용자가 쓴 짧은 prompt를 모델이 학습 때 많이 봤던 caption distribution에 맞춰주는 adapter**

---

# 21. 28–29장 — Image condition

## First frame

\[
(I_0,\text{text})
\rightarrow
\text{future frames}
\]

첫 frame을 주면 문제는 완전 생성보다 **future prediction**에 가까워진다.

텍스트는:

> 무슨 일이 일어나는가

를 주고,

first frame은:

> 어디서 시작하는가

를 고정한다.

---

## First + Last frame

\[
I_{\text{start}}
\rightarrow
\text{middle frames}
\rightarrow
I_{\text{end}}
\]

video interpolation 문제에 가까워진다.

---

# 22. 31–32장 — Temporal / Structural condition

텍스트:

> “곰이 빠르게 고개를 돌린다”

만으로는:

- 얼마나 빠른가
- 어느 방향인가
- 어떤 trajectory인가
- 가속하는가

를 정확히 못 정한다.

따라서 motion을 더 명시적으로 condition으로 준다.

예:

- skeleton
- pose sequence
- depth
- structural map

Skeleton을 모든 frame에 대해 제공하면:

> 각 frame이 어떤 pose를 가져야 하는지 직접 지정

할 수 있다.

---

# 23. Sparse / Dense condition에 대한 주의

강의에서 용어가 약간 애매하게 사용됐다.

Skeleton sequence는 모든 frame에 condition이 있으므로 **dense**라고 부른다.

하지만 교수님도:

> 사실 inference 전에 한 번에 전부 준 것이므로, 다른 관점에서는 여전히 sparse라고 볼 수도 있다

고 언급했다.

따라서 더 명확한 구분은:

\[
\boxed{\text{sparse/dense}=\text{조건의 시간적 해상도}}
\]

\[
\boxed{\text{open/closed-loop}=\text{조건이 실제로 언제 도착하는가}}
\]

이다.

---

# 24. 33–36장 — Trajectory / MotionPrompting

Skeleton보다 일반적으로:

> 특정 point가 frame별로 어디로 움직일지 직접 지정

할 수 있다.

trajectory:

\[
(x_1,y_1)
\rightarrow
(x_2,y_2)
\rightarrow
\cdots
\rightarrow
(x_T,y_T)
\]

point가 \(N\)개라면:

\[
N\times T\times2
\]

---

## Trajectory encoding

각 trajectory에 embedding vector를 할당하고, 해당 시간/공간 좌표에 그 embedding을 배치한다.

결국:

\[
T\times H\times W\times C
\]

같은 condition tensor를 만들 수 있다.

이를 ControlNet-like branch로 주입한다.

---

# 25. ControlNet에 대해 강의에서 요구한 수준

교수님은 **ControlNet 내부 구조는 자세히 설명하지 않겠다**고 명시했다.

이 강의에서 가져갈 정도:

> trajectory/structure condition을 별도 branch로 처리해서 원래 생성 모델의 중간 feature에 영향을 줄 수 있다.

그 이상은 별도 ControlNet 공부가 필요하다.

---

# 26. 37–39장 — Wan-Move

MotionPrompting보다 단순한 방식.

별도 trajectory embedding 대신:

> **첫 frame의 feature를 이후 trajectory 위치에 복사**

한다.

그리고 별도 ControlNet branch 대신:

\[
\text{video latent}
\oplus
\text{condition feature}
\]

를 concat해서 하나의 model에서 처리한다.

장점:

- 별도 auxiliary model 없음
- memory 절감
- computation 절감

---

# 27. 41장 — Viewpoint / Camera

물체 motion이 아니라 **카메라 자체의 움직임**을 condition으로 줄 수 있다.

예:

- zoom out
- camera translation
- rotation
- elevation change

교수님은 이 부분은 3D 사전 지식이 많이 필요하므로 깊게 설명하지 않았다.

핵심만:

\[
\boxed{\text{카메라 pose / motion도 video condition이 될 수 있다}}
\]

---

# 28. 43장 — Motion

여기서는 motion-control 연구들을 넓게 묶어 보여준다.

예:

- drag
- point trajectory
- optical flow
- motion field

핵심:

\[
\boxed{\text{결과 video에서 “어떻게 움직일지”를 직접 condition으로 제어}}
\]

---

# 29. 44장 — Audio / Speech / Multi-shot

condition은 visual input만 있는 게 아니다.

### Audio
효과음 등 일반적인 소리.

### Speech
사람이 말하는 음성.

### Multi-shot
여러 shot이 이어지는 storyboard 형태.

예:

```text
shot 1: 터널 진입
shot 2: 터널 내부
shot 3: 터널 밖 풍경
```

---

# 30. 45–47장 — Action / Open-loop / Closed-loop

이번 강의 마지막에서 매우 중요한 구분.

---

## 30.1 Action

게임처럼:

```text
↑ ↑ → ↑ ↓
```

같은 user action을 video generation condition으로 줄 수 있다.

---

## 30.2 Open-loop

생성 전에 condition을 전부 미리 준다.

예:

```text
text
pose sequence
camera path
action sequence
trajectory
     ↓
model
     ↓
전체 video 생성
```

즉:

\[
\boxed{\text{condition이 inference 전에 모두 준비됨}}
\]

---

## 30.3 Closed-loop

모델이 결과를 낸 뒤, 다음 condition/action이 다시 들어온다.

```text
frame 생성
   ↓
user action
   ↓
next frame 생성
   ↓
user action
   ↓
next frame ...
```

즉:

\[
\boxed{\text{output} \rightarrow \text{new condition} \rightarrow \text{next output}}
\]

feedback loop가 생긴다.

교수님은 이 구조를 **World Model로 가는 방향**이라고 설명했다.

---

# 31. 이번 강의의 핵심 구분들

## Image model inflation vs Video-native model

### Inflation
```text
Frozen image model
+ temporal layers
+ temporal losses
```

### Video-native
```text
Video VAE
+ temporal compression
+ 3D patchify
+ full 3D Transformer
```

---

## Image similarity vs Video temporal similarity

이미지 모델에서는:

> 같은 Husky class 정도의 semantic consistency

만으로도 충분할 수 있다.

비디오에서는:

> 같은 object / background / identity가 frame 사이에서 거의 유지되면서 조금씩 motion만 변해야 한다.

요구되는 similarity 수준이 훨씬 강하다.

---

## Attention map vs Attention output

\[
A=\operatorname{softmax}(QK^\top/\sqrt d)
\]

= attention map

\[
O=AV
\]

= attention output feature

---

## Self-Attention vs Cross-Attention

### Self-Attention

\[
Q,K,V
\]

가 모두 같은 input에서 옴.

### Cross-Attention

예:

\[
Q=\text{video}
\]

\[
K,V=\text{text}
\]

video가 text condition을 참고.

---

## VAE latent vs hidden feature

### VAE latent
영상/이미지를 압축한 representation.

### hidden feature
Transformer/U-Net 내부 layer가 계산 중 만든 중간 representation.

둘은 같은 말이 아니다.

---

## Sparse / Dense vs Open / Closed

### Sparse / Dense
조건이 시간축에서 얼마나 촘촘한가.

### Open / Closed-loop
조건이 생성 전에 다 주어지는가, 생성 도중 계속 새로 들어오는가.

---

# 32. 전체 흐름 한 번에 보기

```text
[Image generation]
        ↓
이미지 여러 장만으로 video 만들기엔 temporal consistency 부족

        ↓

[Inflation era]
Frozen T2I model
+ Temporal layers
+ Mapping network
+ Video-specific losses

        ↓

[Problems]
2D VAE temporal redundancy 못 씀
Factorized attention 한계
Frozen image prior가 큰 motion 제한

        ↓

[Video-native design]

1. Video VAE
   spatial + temporal compression

2. 3D Patchify
   video latent → spatiotemporal tokens

3. Full 3D Attention
   모든 시간/공간 token 상호작용

4. Conditioning architecture
   cross-attention / token concat

5. Data
   VLM recaptioning + curation

        ↓

[How to control video?]

Text
Image
Structure
Skeleton
Trajectory
Camera
Motion
Audio
Action

        ↓

[When is condition given?]

Open-loop
vs
Closed-loop

        ↓

[Closed-loop]
World Model로 연결
```

---

# 33. 복습 시 최소한 기억할 것

1. **Temporal layer**
   - 여러 frame의 hidden feature를 서로 보게 한다.

2. **Mapping network**
   - frame-wise random noise를 video generation에 더 적합한 latent로 보정한다.

3. **Regularization loss**
   - temporal modeling 때문에 기존 image prior가 무너지는 것을 막는다.

4. **Self-attention map**
   - \(QK^\top\)로 계산된 “누구를 얼마나 볼지” weight matrix.

5. **Contrastive loss**
   - 같은 video feature는 가깝게, 다른 video feature는 멀게.

6. **Video VAE**
   - 공간뿐 아니라 시간 redundancy도 압축.

7. **3D Patchify + Full Attention**
   - 시공간 patch를 token으로 만들고 모두 서로 attention.

8. **Cross-Attention**
   - video query가 text key/value에서 필요한 정보를 가져온다.

9. **Conditioning**
   - 무엇을 만들지 제어하는 입력.

10. **Trajectory**
    - point의 시간별 위치를 직접 지정하는 dense motion condition.

11. **Open-loop**
    - 모든 condition을 inference 전에 미리 제공.

12. **Closed-loop**
    - output을 본 뒤 다음 action/condition을 계속 입력.

13. **World Model**
    - closed-loop interactive generation 쪽으로 연결.

---

# 34. 강의자가 깊게 요구하지 않은 부분

강의 스크립트 기준으로 교수님이 명시적으로 깊게 들어가지 않은 것:

- VAE 내부 수학
- ControlNet 내부 구조
- Camera / 3D geometry 상세
- 일부 optical flow + depth 방법론

따라서 이 강의 복습에서는:

> **왜 필요한지 / 어떤 문제를 해결하는지 / 전체 구조에서 어디에 들어가는지**

를 이해하는 것이 우선이다.

---

# 35. 최종 한 문장

> **Video model의 발전은 “좋은 image model에 시간 정보를 얹는 방식”에서 시작해, 이후 시간축까지 압축하는 video VAE와 full 3D attention을 사용하는 video-native architecture로 발전했고, 그 이후의 핵심 문제는 text·image·structure·trajectory·camera·action 같은 condition을 무엇을 언제 주어 사용자가 원하는 비디오를 얼마나 정확하게 제어할 것인가가 되었다.**
