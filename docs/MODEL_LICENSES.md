# 모델·런타임 라이선스 고지

확인일: 2026-10-01. 실행 위치: `D:\CursorVideoLocal`. 이 문서는 외부 구성요소의 권리 출처와 사용 조건을 기록한다. 모델의 성능, LG 그램에서의 실행 시간, 생성 결과물의 모든 제삼자 권리를 보증하는 문서가 아니다.

## 2026-10-07 Intel GPU 기본 장시간 모델: LTX-Video 2B 0.9.8 distilled

| 구성 | 적용 조건 |
|---|---|
| LTX-Video 가중치 | [LTX-Video Open Weights License](licenses/LTX-Video-Open-Weights-License-0.X.txt). 무상·비독점 사용권. 재배포하거나 원격 서비스로 제공할 때 이 원문을 함께 주고(3.2), 변경한 파일에 변경 고지를 하며(3.3), Attachment A의 이용 제한을 사용·배포 계약에 강제 조항으로 넣어야 한다(3.1). 프로젝트 LICENSE 5(b)가 이 이용 제한을 사용자에게 적용한다. 생성 결과물에 대해 라이선스 제공자는 권리를 주장하지 않지만(5), 결과물과 그 사용의 책임은 사용자에게 있다. 우리가 보관한 0.X 원문에는 매출 상한 조항이 없다. |
| T5 텍스트 인코더 | LTX 저장소에 함께 있는 Google T5 v1.1 XXL 인코더, Apache-2.0. 설치 시 BF16으로 변환해 로컬에만 둔다(재배포하지 않음). |
| 실행 방식 | 설치기가 고정 revision·해시로 공식 저장소에서 직접 내려받는다. 이 저장소는 가중치를 포함하거나 재배포하지 않는다. 프로젝트 코드는 diffusers·transformers의 공개 API를 호출하고 일부 메서드를 실행 중에 감쌀 뿐, 그 소스 코드를 복사해 넣지 않는다. |
| 첫 프레임 | Cursor 내장 이미지 생성(GenerateImage) 결과는 사용자 실행 폴더(D)에만 두며 저장소에 넣지 않는다. 실존 인물·상표·로고·실제 현장 표시를 요청하지 않는다. |
| SkyReels-V2(NVIDIA 전용) | [Skywork Community License](licenses/SkyReels-V2-Skywork-LICENSE.txt). 그 조건과 이용 제한을 함께 따른다. |

## 2026-10-02 현재 기본 모델

기본 제작 경로를 SD 1.5 Q4 첫 이미지 + Neodragon 채널별 INT8 영상으로 바꿨다. 기존 Lightning·Wan은 진단 기록으로만 보존한다. 영상 49프레임의 시간 변화는 Neo가 생성하며 이전 이미지 전환·보간 방식은 실행하지 않는다.

| 구성 | 적용 조건과 고정 출처 |
|---|---|
| Neodragon 가중치 | BSD-3-Clause-Clear **및** Qualcomm Responsible AI License. 공식 revision `5bdc87a4895a4fd148ca14c03181345d278d1040`. 일반 창작에 대한 포괄적 비상업 전용 제한은 RAIL 본문에서 확인되지 않았으며 허용 라이선스와 RAIL의 금지 용도를 함께 따른다. 무제한 MIT/Apache 가중치로 표시하지 않는다. |
| 원저자 코드 | `d2abbe99f46577c4e1db682ad3c26832ec0c23b9`. `engines/experimental-neodragon`의 LICENSE와 파일별 BSD·Apache·MIT 고지를 보존한다. 원저자 Python 파일을 수정하지 않았다. |
| 채널별 INT8 변경본 | `transformer-cpu-int8-v2.pt`, 변환명 `neodragon-cpu-per-channel-int8-v2`. 선형 계층의 출력 채널별 스케일을 사용한다. 원본은 보존하며 변경 내용·원본 해시·출력 해시·Torch 버전·라이선스를 `.provenance.json`에 기록한다. |
| 첫 이미지 | 기존 SD 1.5 Q4 보존 원본을 사용한다. CreativeML Open RAIL-M. Lightning용 변경 스케줄이나 동작 모듈을 사용하지 않는다. |
| 출력 검사 | CompVis/stable-diffusion-safety-checker, MIT, revision `cb41f3a270d63d454d385fc2e4f571c487c253c5`. 일반 생성에 요구된 검사 기능을 유지한다. 버전 차이로 바뀐 position_ids 저장 여부는 값 일치와 엄격한 나머지 가중치 로딩으로 처리한다. |
| CPU 실행 의존성 | `neodragon-runtime.lock.json`에 CPU-only Torch 등 36개 파일의 정확한 버전·URL·SHA256을 고정했다. 각 설치 패키지의 저작권·라이선스·dist-info는 D runtime에 보존한다. 이를 모델의 MIT 허가로 일괄 대체하지 않는다. |

[공식 모델 고지](https://huggingface.co/Qualcomm-AI-Research/Neodragon), [공식 코드 LICENSE](https://github.com/Qualcomm-AI-research/neodragon/blob/d2abbe99f46577c4e1db682ad3c26832ec0c23b9/LICENSE.txt), [Qualcomm RAIL](https://www.qualcomm.com/site/responsible-ai-license), [출력 검사 모델 카드](https://huggingface.co/CompVis/stable-diffusion-safety-checker).

RAIL은 일반 생성 모델에 함께 적용되며 README를 위반하는 사용, 불법·위해·사칭·차별 등의 용도를 금지한다. 양자화로 해당 조건을 제거하지 않는다. 모델·수정본을 배포할 때 원문, 저작권·출처와 변경 고지를 전달해야 한다. 일반 생성용 안전 검사를 끄지 않는다. 공식 RAIL 본문과 그 아래 웹사이트의 일반 TOU를 같은 조항으로 혼동하지 않는다. 생성 미디어와 입력 자료의 제삼자 권리는 모델 허가만으로 일괄 해결되지 않는다.

프로젝트 내부에서 별도 FFmpeg 프로세스를 실행해 개인·상업 프로젝트의 영상을 만드는 것과 GPL FFmpeg 실행 파일 자체를 제삼자에게 배포하는 것은 적용 의무가 다르다. 실행 파일을 외부에 재배포한다면 정확한 빌드의 해당 소스·의존 라이브러리·빌드 설정을 제공하는 등 GPL 방식으로 의무를 이행해야 한다. 단순한 라이선스·소스 링크 목록을 완전한 바이너리 배포 의무 이행으로 표시하지 않는다. 현재 작업은 로컬 실행 환경과 고지를 준비한 것이며 공개 바이너리 배포는 하지 않았다.

이하 기존 진단 구성의 고지는 당시 파일과 측정 이력의 기록이다.
## 기존 진단 구성: AnimateDiff-Lightning + Stable Diffusion 1.5

현재 완료된 로컬 영상 시각시험은 모두 기대한 시각 품질 검증에 실패했다. 파일 로드·영상 인코딩·연결 기능의 성공은 제작용 영상 품질의 성공을 뜻하지 않는다. 아래 Lightning 조합과 추가 진단 모델은 제작용 기본 모델 선정이 완료된 것으로 표시하지 않는다. 라이선스 확인과 출력 품질 검증은 별개다.

**두 가중치는 CreativeML Open RAIL-M 조건을 지키면 상업용 프로젝트에 사용할 수 있다.** 비상업 전용 또는 매출액 상한 조항은 확인되지 않았다. 자유로운 MIT/Apache 가중치와는 다르며, 용도 제한이 있는 공개 가중치로 표시한다. ByteDance 라이선스 §§2–5는 사용·변형·배포와 유상 판매 및 원격 서비스를 포함한 권리 허여와 조건을 규정한다. 원문: [ByteDance의 해당 revision 라이선스](https://huggingface.co/ByteDance/AnimateDiff-Lightning/blob/027c893eec01df7330f5d4b733bc9485ee02e8b2/LICENSE.md), [Stable Diffusion 원저자 라이선스](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).

| 구성요소 | 고정된 출처 | 파일 | 크기 | 라이선스·출처 수준 |
|---|---|---|---:|---|
| SD 1.5 기반 모델의 보존 원본 | `gpustack/stable-diffusion-v1-5-GGUF` / `dcac270609fffc0ce7c7d41a3c0e752721859f7c` | `stable-diffusion-v1-5-Q4_0.gguf` | 1,747,190,784 바이트 | CreativeML Open RAIL-M. GPUStack의 제삼자 양자화이며 원저자의 공식 배포본은 아니다. CLIP·VAE는 FP16으로 포함한다. |
| 실제 추론용 기반 모델 | 위 원본에서 로컬로 준비, `lightning-linear-alphas-gguf-v1` | `stable-diffusion-v1-5-Q4_0-lightning-linear.gguf` | 1,747,194,848 바이트 | 같은 CreativeML Open RAIL-M. Lightning의 공식 linear beta 설정에 맞춘 스케줄 텐서만 추가한 변경본. |
| 4단계 동작 모델 | `ByteDance/AnimateDiff-Lightning` / `027c893eec01df7330f5d4b733bc9485ee02e8b2` | `animatediff_lightning_4step_comfyui.safetensors` | 908,929,664 바이트 | CreativeML Open RAIL-M, Copyright 2024 Bytedance Inc. 공식 ByteDance 가중치. 원래 AnimateDiff의 텐서 이름과 시간 위치 인코더를 포함한다. ComfyUI 프로그램은 사용하지 않는다. |

실제 추론에 참여하는 두 모델의 합계는 **2,656,124,512 바이트, 약 2.66 GB / 2.47 GiB**다. 변경 전 원본도 보존하므로 세 파일의 저장 공간은 **4,403,315,296 바이트, 약 4.40 GB / 4.10 GiB**다. 설치 기록·런타임·시험 파일은 별도이며 이 수치는 실행 중 메모리 사용량이 아니다. CLIP·VAE가 포함되어 있어 기본 조합에 별도 T5를 다운로드하지 않는다. 파일의 정확한 SHA-256·revision·역할은 프로젝트의 `lightning.lock.json`과 실행 폴더의 모델 lock에 기록되어 있다. 설치기는 다운로드 후 파일 크기와 SHA-256을 검증한다.

### 변경본과 형식 선택 기록

원본 SD 파일의 SHA-256은 `c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b`, 추론용 변경본은 `b4a990b50d0700a9b80687adde0ba3cf545137bbe230b2605bbf0bb366d0c61f`다. 변경본에는 이름 `alphas_cumprod`, 형식 F32, 길이 1,000인 텐서를 추가했다. 설정은 beta_start=0.00085, beta_end=0.012, beta_schedule=linear이며 [ByteDance의 공식 추론 예제](https://huggingface.co/ByteDance/AnimateDiff-Lightning/blob/027c893eec01df7330f5d4b733bc9485ee02e8b2/README.md)를 따른다. 원본의 텐서 1,130개의 데이터 바이트는 그대로 보존했고 공통 payload의 SHA-256은 `c34518cc07686f8e371574c66d81b46e487bb473af9cba1462c749aea5474920`이다. 학습 가중치를 바꾸지 않았고 원본 파일도 유지한다. 준비 버전과 보존 검증 결과는 `D:\CursorVideoLocal\models\stable-diffusion-v1-5-Q4_0-lightning-linear.provenance.json` 및 동봉한 변경 고지에 기록한다. 변경본을 제공할 때도 원본의 라이선스와 출처, 이 변경 사실을 함께 제공한다.

기본 동작 모델의 SHA-256은 `aeb66ae8ff4a868d31379c3bde3e5e7e510a4d4b565a06ee4f1297e93a561dc5`다. 같은 공식 저장소의 `animatediff_lightning_4step_diffusers.safetensors`(907,702,248 바이트, SHA-256 `8f3330914edd8fc2ee5659e6944323858feb0ec21cc913789202d043f69b5e17`)는 형식 호환성 시험에 사용했으며 기본 추론에서는 제외한다. pinned 엔진은 `temporal_transformer.proj_in.weight`라는 키로 시간 축 모듈을 감지하고 활성화하지만 Diffusers 형식에는 이 계층 이름이 없다. 공식 ComfyUI 형식은 필요한 이름과 위치 인코더를 포함한다. [감지 및 모듈 생성 코드](https://github.com/leejet/stable-diffusion.cpp/blob/3f8527a/src/model/diffusion/unet.hpp#L91), [시간 축 모듈 코드](https://github.com/leejet/stable-diffusion.cpp/blob/3f8527a/src/model/diffusion/animatediff.hpp). 파일 이름의 ComfyUI 표시는 가중치 포맷이며 외부 GUI 사용을 뜻하지 않는다. 이 형식 교정과 스케줄 교정은 생성 품질을 보증하는 문장이 아니다.

SD 1.5 카드에는 연구를 주된 용도로 설명하는 문장이 있다. 이것을 숨기지 않으며, 법적 사용 허여는 함께 게시된 Open RAIL-M의 실제 조항과 용도 제한을 따른다. [GPUStack 모델 카드](https://huggingface.co/gpustack/stable-diffusion-v1-5-GGUF)는 원본의 미러임을 밝히고 같은 라이선스를 표시한다. [ByteDance 모델 카드](https://huggingface.co/ByteDance/AnimateDiff-Lightning)는 4단계 모델과 그 공식 추론 설정을 제공한다.

### 지켜야 하는 조건

- 모델·변형 모델을 제삼자에게 제공하거나 원격 서비스로 제공할 때 라이선스 원문, 저작권·출처 고지를 제공한다. 변경한 모델 파일에는 변경 사실을 표시한다. 양자화도 원본 모델의 조건을 없애지 않는다.
- 모델 또는 변형 모델을 사용하는 이용자에게 §5 및 Attachment A의 제한을 적용한다. 모델·변형 모델의 배포 계약에는 제한이 집행 가능한 조건으로 들어가야 한다.
- Attachment A에는 불법 행위, 아동 위해, 타인을 해칠 목적의 허위 정보·개인정보 생성, 명예훼손·괴롭힘, 법적 권리나 의무를 결정하는 완전 자동 의사결정, 차별·취약성 악용, 의료 조언, 사법·법집행·이민·망명 절차용 특정 정보 생성 등의 금지 용도가 있다. 요약보다 함께 보존한 원문이 우선한다.
- §6에서 배포자는 생성 결과물에 대한 권리를 주장하지 않는다고 규정한다. 결과물의 사용 책임과 입력 자료·상표·실존 인물·저작물에 관한 제삼자 권리는 별도로 남는다. 라이선스만으로 모든 출력의 저작권 소유 또는 비침해가 자동 확정되지는 않는다.

원문 사본: `docs/licenses/AnimateDiff-Lightning-LICENSE.md`, `docs/licenses/SD15-CreativeML-OpenRAIL-M-LICENSE.txt`. SD 라이선스 사본은 CompVis 원저자 저장소의 공통 원문이며, GPUStack 가중치의 별도 공식 발행을 뜻하지 않는다.

## 추가 진단 모델: TAESD와 원본 AnimateDiff v3

다음 두 파일은 디코딩 비용과 Lightning 이외의 동작 모듈을 비교하기 위한 진단 자료다. 현재 제작용 기본 모델로 선정하지 않았고, TAESD를 적용한 시험을 포함해 영상 품질 검증에 성공했다고 표시하지 않는다.

| 구성요소 | 공식 모델 출처 / revision | 원래 파일 → 로컬 파일 | 크기 | SHA-256 | 공개 라이선스 |
|---|---|---|---:|---|---|
| SD 1.5용 TAESD 전체 가중치 | `madebyollin/taesd` / `614f76814bbe30edbe2e627ace1c2234c81a2c0e` | `diffusion_pytorch_model.safetensors` → `taesd_sd15.safetensors` | 9,793,292 바이트 | `db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549` | MIT |
| 원본 AnimateDiff v3 동작 모듈 | `guoyww/animatediff` / `fdfe36afa161e51b3e9c24022b0e368d59e7345e` | `v3_sd15_mm.ckpt` → 같은 이름 | 1,673,262,583 바이트 | `2412711886f61091846f53204aabc38aa6e09356d62a9808abe4daa802168343` | Apache-2.0 |

TAESD는 저자의 [공식 모델 카드](https://huggingface.co/madebyollin/taesd/blob/614f76814bbe30edbe2e627ace1c2234c81a2c0e/README.md)에 MIT로 표시되어 있으며 [원저자 라이선스](https://github.com/madebyollin/taesd/blob/027d3ef404a04a141e60d5ec0fee149c2c8bdd74/LICENSE)는 Copyright (c) 2023 Ollin Boer Bohan이다. 상업 사용·변형·재배포를 허용하며 저작권·허가·면책 고지를 유지한다. 로컬 파일은 식별하기 쉽게 이름만 바꿨고 위 SHA-256의 공식 전체 가중치를 그대로 사용한다. 엔진의 [공식 TAESD 사용법](https://github.com/leejet/stable-diffusion.cpp/blob/3f8527a/docs/taesd.md)도 이 전체 파일을 지정한다. SD1/2용 디코더이며 Wan용 디코더가 아니다. 저자는 세부 묘사 손실과 프레임별 깜박임 가능성을 설명한다. 속도 향상 비율이나 LG 그램에서의 실행 시간을 보장하지 않는다.

원본 AnimateDiff v3는 [저자의 공식 모델 카드](https://huggingface.co/guoyww/animatediff/blob/fdfe36afa161e51b3e9c24022b0e368d59e7345e/README.md)에서 Apache-2.0을 가중치 저장소에 직접 지정한다. [원저자 LICENSE.txt](https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/LICENSE.txt)도 표준 Apache-2.0이며 라이선스 원문·저작권·출처·해당 NOTICE 유지, 변경 사실 표시 등의 조건을 따른다. 원저자의 [README Disclaimer](https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/README.md#disclaimer)에는 학술용 공개라는 안내도 있다. 이를 함께 고지하되 명시적인 비상업 전용 라이선스라고 단정하지 않는다. 이 동작 모듈을 SD1.5와 결합할 때 SD1.5의 CreativeML Open RAIL-M 조건이 없어지는 것은 아니다. 별도 GUI 또는 제삼자 WebUI 확장 프로그램은 사용하지 않는다.

원문 사본: `docs/licenses/TAESD-MIT-LICENSE.txt`, `AnimateDiff-stock-Apache-2.0-LICENSE.txt`. 진단 파일의 출처·해시·선정 상태는 `docs/licenses/Diagnostic-model-and-engine-notice.md`에 함께 기록한다.

## 선택 후보: Wan 2.1 1.3B

Wan 조합은 기본 경량 모델보다 자원 소모가 크므로 선택 프로필이다. 코드·공식 모델은 Apache-2.0이며 원본 UMT5-XXL도 Apache-2.0이다. 로컬 파일은 제삼자가 변환·양자화한 포맷이라는 사실을 함께 기록한다. [Wan 공식 라이선스](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt), [Wan 공식 모델 카드](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B), [Google UMT5-XXL 모델 카드](https://huggingface.co/google/umt5-xxl).

| 역할 | 고정된 출처 revision | 파일 | 크기 |
|---|---|---|---:|
| 영상 확산 모델 | `calcuis/wan-1.3b-gguf` / `0652f175f44055eb60cca26dd7cd89c14abe22ce` | `wan2.1_t2v_1.3b-q4_0.gguf` | 916,895,968 바이트 |
| 텍스트 인코더 | `city96/umt5-xxl-encoder-gguf` / `b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7` | `umt5-xxl-encoder-Q4_K_S.gguf` | 3,497,596,768 바이트 |
| 영상 VAE | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` / `123acf1cc74bccbb9bfff8ac1ee72edc08c2341d` | `split_files/vae/wan_2.1_vae.safetensors` | 253,815,318 바이트 |

정확한 SHA-256은 `models.lock.json`을 따른다. Comfy-Org 파일을 직접 읽으며 ComfyUI 프로그램은 실행하지 않는다. Apache-2.0 모델·코드를 재배포할 때 라이선스와 관련 저작권·특허·출처·NOTICE를 유지하고 변경 사실을 표시한다. 상표 사용권과 비침해 보증은 포함되지 않는다. 원문 사본은 `docs/licenses/Wan2.1-Apache-2.0-LICENSE.txt`다.

## 실행 구성요소

| 구성요소 | 실제 고정 버전 | 라이선스·고지 |
|---|---|---|
| stable-diffusion.cpp | `master-929-3f8527a` CPU 빌드, Vulkan은 선택 빌드 | MIT, Copyright 2023 leejet. 배포물에 저작권·허가·면책 고지 유지. [해당 소스 라이선스](https://github.com/leejet/stable-diffusion.cpp/blob/3f8527a/LICENSE). ZIP 크기·SHA-256은 `models.lock.json`에 있다. |
| stable-diffusion.cpp 이전 CPU 진단 빌드 | `master-841-6b3edaa`, commit `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba` | MIT. `D:\CursorVideoLocal\engines\diagnostic-841`에 분리 보존한다. 기존 최신 엔진의 선택·고지를 대체하지 않는 비교 진단용이다. [별도 공식 라이선스](https://github.com/leejet/stable-diffusion.cpp/blob/6b3edaaf32cc19e5bb2d819c788bd557eddc8eba/LICENSE). |
| GGML | 위 엔진 배포물에 포함된 버전 | MIT. 실제 배포물의 `ggml.txt`를 보존했다. |
| Python | Windows 임베디드 3.12.10 | PSF 및 포함 구성요소 고지. 실제 `LICENSE.txt` 전체를 보존했다. [Python 배포 출처](https://www.python.org/ftp/python/3.12.10/). ZIP의 취득 후 SHA-256은 `D:\CursorVideoLocal\audits\runtime-source.json`에 있다. |
| Pillow | 12.3.0 | MIT-CMU 계열 라이선스. 저작권·허가 고지 유지, 저작자 이름의 무단 광고 사용 금지. [해당 버전 라이선스](https://github.com/python-pillow/Pillow/blob/12.3.0/LICENSE). |
| imageio-ffmpeg 파이썬 래퍼 | 0.6.0 | BSD-2-Clause. 소스·바이너리 배포의 고지 유지. [해당 버전 라이선스](https://github.com/imageio/imageio-ffmpeg/blob/v0.6.0/LICENSE). |
| 포함 FFmpeg 실행 파일 | 7.1 essentials, gyan.dev 빌드 | **GPL-3.0-or-later**. 실제 실행 파일의 `-L` 출력으로 확인. `--enable-gpl --enable-version3`와 x264 등이 포함된다. 래퍼의 BSD 조건과 서로 다르다. |

FFmpeg는 프로젝트 안에서 영상 포장·인코딩에 사용하는 실행 구성요소다. 별도 외부 서비스·GUI 프로그램을 이용하지 않는다. GPL도 상업적 사용을 허용한다. 이 바이너리를 포함하여 제삼자에게 배포하면 해당 바이너리와 포함 GPL 구성요소에 대한 GPL 고지·대응 소스 제공 등 적용되는 배포 의무를 이행해야 한다. 이 프로젝트 전체 또는 포함 FFmpeg를 일괄 MIT로 표시하지 않는다. 별도 프로세스로 인코더를 호출하는 사실만으로 배포 의무가 없어지는 것은 아니다. [FFmpeg의 라이선스 설명](https://ffmpeg.org/legal.html), [해당 FFmpeg GPL 원문](https://github.com/FFmpeg/FFmpeg/blob/n7.1/COPYING.GPLv3).

`docs/licenses/ffmpeg-7.1-build.txt`와 `ffmpeg-7.1-license-output.txt`는 설치한 실행 파일에서 직접 얻었다. 실행 파일 SHA-256은 `runtime-licenses.json`에 기록되어 있다. 실제 배포물을 외부에 전달할 경우 엔진·Python·코덱에 딸린 다른 라이브러리의 고지도 함께 유지한다. 이 문서 자체는 완전한 배포물 SBOM이나 대외 배포 적합성 인증이 아니다.

이전 CPU 진단 ZIP은 `sd-master-6b3edaa-bin-win-cpu-x64.zip`, 24,084,786 바이트, SHA-256 `a36edb067de09fc9f70fcd193e519ff62592f860744558d7918762c7c3401050`이다. [공식 릴리스](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-841-6b3edaa)의 digest와 설치용 ZIP의 실제 해시가 일치한다. 이 버전의 소스 MIT 원문, 실제 배포물의 `stable-diffusion.cpp.txt` 및 `ggml.txt`를 각각 `docs/licenses/stable-diffusion.cpp-diagnostic-841-MIT-LICENSE.txt`, `stable-diffusion.cpp-diagnostic-841-runtime-LICENSE.txt`, `ggml-diagnostic-841-MIT-LICENSE.txt`로 따로 보존했다. 최신 `master-929-3f8527a`의 기존 원문 사본은 그대로 유지한다.

## 기록과 갱신

0.6.0 현장 참고 자료는 모델 라이선스와 별도다. ŠJů의 레일 현장 영상 2개는 **CC BY 4.0**, KEmel49의 굴착기/트럭 영상은 **CC BY-SA 4.0** 게시 조건을 확인했다. 원본·9개 참고 구간과 출처/해시는 D의 `datasets/construction-v1`에 함께 둔다. 저작자·출처·라이선스·변경을 표시하고 BY-SA 변형을 배포할 때는 동일/호환 조건도 유지한다. [자료별 원출처와 변경](SITE_ADAPTATION.md), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 새 모델의 신경망 훈련이나 재학습 가중치 배포 허여를 일괄 확인한 것이 아니다.

릴리스의 `examples/site-draft.mp4`는 CC BY 참고 프레임을 조건으로 모델이 새 움직임을 생성한 **실험 초안**이다. 같은 폴더의 `site-draft-ATTRIBUTION.md`와 모델의 이용 조건을 함께 유지한다. 프로젝트 라이선스(LICENSE, 개인 학습 무료·기관 교육 유료·변경 금지)는 이 영상·참고 자료·모델 가중치의 원래 라이선스를 바꾸지 않으며, CC BY 출처 표시를 제한하지 않는다.

`docs/licenses/sources.json`에는 다운로드한 라이선스 사본의 출처 URL·확인일·크기·SHA-256이 있다. 프로젝트 코드는 모델·라이선스를 자동 변경하지 않는다. 모델이나 런타임 버전을 바꿀 때 lock의 revision·파일 해시, 이 고지, 해당 원문 사본을 함께 갱신한다. 새 LoRA·현실 모델·음성 모델을 추가할 때 기존 모델의 사용 가능성이 새 파일의 사용 허여를 대신하지 않는다.
