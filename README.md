# 2021 Speaker Recognition

주최 : 과학기술정보통신부
주관 : NIA 한국지능정보사회진흥원
운영 : Dacon
후원 : Naver, KT
대회일정

| 일정        | 내용 | 기타           | 결과 |
| ----------- | ---- | -------------- | ---: |
| 9월말       | 예선 | 상위 20팀 선발 | 13등 |
| 10.09~10.24 | 본선 | 상위 10팀 선발 |  5등 |
| 11.01~11.30 | 결승 | 수상 6팀       |  9등 |

## 참여인원

- 2명

## 결과

- 예선 : 13등
- 본선 : 5등
- 결승 : 9등

# 결승 요약

<img src="./png/1.png"
     sizes="(min-width: 600px) 100px, 50vw">

## 방법

데이터 형태
|idx|wav path|speaker id|
|---|---|---:|
|1|1.wav|1|
|2|2.wav|1|
|3|3.wav|2|
|4|4.wav|4|
...

## Overview

<img src="./png/6.png"
height='300' width ='300'>

### 방법 1 : siamese model

| idx | left wav | right wav | label |
| --- | -------- | --------- | ----: |
| 1   | 1.wav    | 2.wav     |     0 |
| 2   | 1.wav    | 3.wav     |     1 |
| 3   | 2.wav    | 4.wav     |     1 |

- siamese에 들어가기 위해, 음성 데이터를 쌍으로 묶어준다. 단 imbalance를 피하기 위해 1과 0의 비율을 0.5로 맞추었다.

### 방법 2 : speaker alone

| idx | wav path | speaker id |
| --- | -------- | ---------: |
| 1   | 1.wav    |          1 |
| 2   | 2.wav    |          1 |

- 480명의 speaker를 space에 represent한다.

## Agumentation(확률적으로 선택)

- AddGaussianNoise
- TimeStretch
- PitchShift
- Shift
- reverse

## Loss Function

### BCEWITHLOGITSLOSS with 방법1

This loss combines a Sigmoid layer and the BCELoss in one single class. \
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss.

<img src="./png/2.png"
width = '300' height='100'>

where sigma is Sigmoid

---

### Constrasive Loss with 방법2

<img src="./png/7.png"
width = '300' height='100'>

---

### Triplet Loss with 방법2

<img src="./png/8.png"
width = '300' height='100'>

(a,p,n) is parameter which means (anchor, positive, negative)

### angleproto with 방법2

<img src="./png/2.png"
width = '300' height='100'>

## Model

### ResNet34

<img src="./png/3.png"
width = '150' height='100'>

### thin-ResNet with SEBlock

<img src="./png/4.png"
width = '150' height='100'>

### patch is all you need

<img src="./png/5.png"
width = '150' height='100'>

## Optimizer

RAdam

## Mixed Precision is Used

Mixed Precision of amp make float32 as float16 to accelerate casting such as Linear, Conv layer, etc.

## requirement

torch_optimizer
torch == 1.10.0
torchaudio == 0.10.0
pytorch-metric-learning
faiss-gpu
soundfile
scipy
audiomentations

## HardWare

- Naver clova NSML
- NVIDIA TITAN 32G
