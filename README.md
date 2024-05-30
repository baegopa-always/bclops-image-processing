# 비클롭스

### 프로젝트 개요

<p align="center">
  <img src="https://file.notion.so/f/f/04e846d6-857c-4ddd-aea7-9dab39f443b8/061d1c7e-83b8-46d7-938a-d0e9c53263dd/bclops-logo.png?id=dd7b396c-8764-4455-8e84-2478bea444fd&table=block&spaceId=04e846d6-857c-4ddd-aea7-9dab39f443b8&expirationTimestamp=1717164000000&signature=nIRwicjn-vciiH8_5lUMn7IWnSGisB9DxQo48bb7MbY&downloadName=bclops-logo.png" height=300>
</p>

### 딥러닝과 이미지 처리 기법을 이용한 터널 내 절리 추출 앱 서비스
졸업 프로젝트였던 캡스톤 디자인 과목에서 진행한 AI 융합 프로젝트입니다.<br>
현재 터널 페이스 매핑 과정은 수기로 진행되는데 그 중 절리(터널 굴착 중 발생하는 암반 면의 균열) 파악 단계에서 <br>
작업자의 안전성과 작업 효율의 향상을 위해 실시간 스마트폰 촬영으로 절리를 분석할 수 있는 앱을 개발하였습니다.
<br><br>
📚 <a href="https://baegopa.notion.site/2-fab23219dd4e4cd8a77936b7f9194830?pvs=4">프로젝트 상세 문서</a>

---
### 역할 (이미지 전처리 모델 개발 및 서버 구축)
- OpenCV를 활용하여 이미지 전처리 모델 개발
- 이미지 프로세싱 모델 REST API 개발

---
### 트러블 슈팅
#### 1. 절리(edge) 추출 시 무분별한 line 문제 개선
  - 기존
      - 이미지 밝기에 따라 결과 값이 다른 문제
  - 개선 방안
      - adaptive equalization 적용하여 이미지 명암비 조절
      - bilateral filter 적용하여 두꺼운 edge 우선적으로 추출
      - closing 연산 추가하여 가장 긴 절리 우선 추출
      - AI 모델과 이미지 프로세싱 모델 결과 AND 연산하여 대표 절리 추출 강화

#### 2. 이미지 해상도로 인한 결과 차이 개선
  - 기존
      - 같은 이미지라도 해상도에 따라 결과 값이 다른 문제 발생
  - 개선 방안
      - 피사체와의 거리 값을 추가 파라미터로 받아 edge 두께를 일정하도록 수정

#### 3. 분석 응답 시간 개선
  - 기존
      - AI 모델 및 이미지 프로세싱 모델 AND 연산 시간이 약 CPU 에 따라 최대 약 20초 소요
  - 개선 방안
      - 입력 이미지를 리사이징 후 연산량을 대폭 감소 시켜 최대 5초 내외 응답

#### 4. REST API 응답 시간 및 AWS 예산 문제 개선
  - 기존
      - EC2에 모델 배포 시 둘 이상의 요청이 들어올 경우 응답 시간 지체
      - CPU가 항상 활성화되어 리소스 낭비
  - 개선 방안
      - Docker와 ECR을 통해 배포하여 api gateway로 연결 후 기존 최대 5초 응답에서 3~4초로 개선
      - api gateway 요청할 경우에만 리소스 사용하여 예산 절감

---
### 아키텍처
<p align="center">
  <img src="https://file.notion.so/f/f/04e846d6-857c-4ddd-aea7-9dab39f443b8/6f92fb79-1210-4d80-8fdf-f038ff22324a/Untitled.png?id=33a49613-1e7d-48d8-9965-f54e4e32e035&table=block&spaceId=04e846d6-857c-4ddd-aea7-9dab39f443b8&expirationTimestamp=1717164000000&signature=VfDCSnnxOsJJzCHuN2eSFbiMDIdsnvUFJg2yEBQH6d0&downloadName=Untitled.png" height=400>
</p>

---
### 프로젝트 구성
- 팀 구성 : 7인 1팀 <br>
- 개발 기간 : <n>2023. 03. - 2023. 12. </n>

---
### 기술 스택
- AWS (ECR, EC2, Lambda function, API gateway, S3)
- Python, OpenCV, Flask
- Docker
