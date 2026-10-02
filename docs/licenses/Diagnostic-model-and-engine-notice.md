# 진단용 모델·이전 엔진 출처 고지

확인일: 2026-10-01. 이 문서는 프로젝트의 출처·상태 기록이며 원저자의 라이선스 원문이 아니다. 현재 완료된 로컬 영상 시각시험은 모두 기대한 시각 품질 검증에 실패했다. 다음 구성요소는 비교 진단용으로 보존하며 제작용 기본 모델·엔진으로 선정한 것으로 표시하지 않는다. 라이선스 확인은 생성 품질 보증이 아니다.

## TAESD 전체 가중치

- 공식 저장소: `madebyollin/taesd`.
- 모델 revision: `614f76814bbe30edbe2e627ace1c2234c81a2c0e`.
- 원래 파일: `diffusion_pytorch_model.safetensors`.
- 로컬 파일: `D:\CursorVideoLocal\models\taesd_sd15.safetensors`.
- 크기: 9,793,292 바이트.
- SHA-256: `db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549`.
- 변경: 파일 이름만 변경. 공식 전체 모델의 바이트는 그대로 사용한다.
- 공개 라이선스: MIT, Copyright (c) 2023 Ollin Boer Bohan. 상업 사용·변형·재배포를 허용하며 저작권·허가·면책 고지를 유지한다.
- [고정된 공식 모델 카드](https://huggingface.co/madebyollin/taesd/blob/614f76814bbe30edbe2e627ace1c2234c81a2c0e/README.md), [원저자 MIT 원문](https://github.com/madebyollin/taesd/blob/027d3ef404a04a141e60d5ec0fee149c2c8bdd74/LICENSE).

동봉 원문: `TAESD-MIT-LICENSE.txt`. SD1/2용이며 Wan용 모델이 아니다. 미세 묘사 손실과 영상 프레임별 깜박임 가능성이 있으며 이 컴퓨터에서의 디코딩 속도 향상을 보장하지 않는다.

## 원형 AnimateDiff v3

- 공식 저장소: `guoyww/animatediff`.
- 모델 revision: `fdfe36afa161e51b3e9c24022b0e368d59e7345e`.
- 파일: `D:\CursorVideoLocal\models\v3_sd15_mm.ckpt`.
- 크기: 1,673,262,583 바이트.
- SHA-256: `2412711886f61091846f53204aabc38aa6e09356d62a9808abe4daa802168343`.
- 변경: 공식 원형 파일을 그대로 사용한다.
- 공개 라이선스: 저자 모델 카드의 Apache-2.0. 원저자 LICENSE.txt도 표준 Apache-2.0이다. 상업 사용 허여와 재배포·출처·변경 고지 조건을 따르며 해당 NOTICE도 유지한다. 결합한 SD1.5의 CreativeML Open RAIL-M 조건은 별도로 유지한다.
- [고정된 공식 모델 카드](https://huggingface.co/guoyww/animatediff/blob/fdfe36afa161e51b3e9c24022b0e368d59e7345e/README.md), [원저자 Apache-2.0 원문](https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/LICENSE.txt).

동봉 원문: `AnimateDiff-stock-Apache-2.0-LICENSE.txt`. 원저자 README의 학술용 공개 안내도 함께 기록한다. 해당 안내를 명시적인 비상업 전용 라이선스라고 단정하지 않는다. 별도 GUI·WebUI 확장 프로그램을 실행하지 않는다.

## 이전 CPU 엔진 비교 빌드

- 공식 저장소: `leejet/stable-diffusion.cpp`.
- tag: `master-841-6b3edaa`.
- commit: `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`.
- 공식 ZIP: `sd-master-6b3edaa-bin-win-cpu-x64.zip`.
- 크기: 24,084,786 바이트.
- SHA-256: `a36edb067de09fc9f70fcd193e519ff62592f860744558d7918762c7c3401050`.
- 실행 폴더: `D:\CursorVideoLocal\engines\diagnostic-841`.
- 엔진 라이선스: MIT, Copyright 2023 leejet. GGML의 해당 MIT 고지도 별도 보존한다.
- [공식 릴리스](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-841-6b3edaa), [고정 소스 라이선스](https://github.com/leejet/stable-diffusion.cpp/blob/6b3edaaf32cc19e5bb2d819c788bd557eddc8eba/LICENSE).

동봉 원문: `stable-diffusion.cpp-diagnostic-841-MIT-LICENSE.txt`, `stable-diffusion.cpp-diagnostic-841-runtime-LICENSE.txt`, `ggml-diagnostic-841-MIT-LICENSE.txt`. 소스·배포물의 원래 고지를 그대로 복사했다. 최신 `master-929-3f8527a`의 기존 라이선스 사본과 lock를 대체하지 않는다. 바이너리에 포함된 다른 라이브러리의 고지를 포함한 완전한 외부 배포 의무 확인은 별도다.

라이선스 사본의 취득 출처·해시·확인일은 `sources.json`을 따른다.
