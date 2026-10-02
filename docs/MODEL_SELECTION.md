> 0.7.0 현재 기준: LG 그램 16GB·Intel 내장 GPU를 기준으로 한 저메모리 OpenCL 스트리밍 경로와 C/D SSD 선택을 추가했다. 기본 preview는 384×256이며 GPU 버퍼 64MiB·작업 RAM 5GiB 상한이다. 아래는 이전 버전의 조사·측정 이력이다. RTX의 시간은 그램 성능이 아니며 현장 사실감과 Iris 실행은 미검증이다. 현재 변경과 실측은 [LOW_RESOURCE_GPU.md](LOW_RESOURCE_GPU.md)를 우선한다.

> 2026-10-02 리빌드 업데이트: 모델은 Neodragon 채널별 INT8 + SD 1.5 Q4를 유지했다. 0.5.0은 첫 이미지 캐시를 사용한 새 추론에서 119.703초, 조건 계산도 재사용한 새 추론에서 87.922초를 기록했다. 최대 상주는 약 3.95GiB다. [비교 조건과 최적화](REBUILD_OPTIMIZATION.md), [최신 검증](VALIDATION.md). 이는 그램 최적 모델의 확정이나 Sora 품질 달성이 아니며 Iris 속도는 미측정이다. LTX·NOVA는 여전히 비교 후보다. 아래의 0.4.0 수치와 선정 당시 조사·사용자 후기는 비교 근거로 보존했다.
# LG 그램용 로컬 영상 모델 재선정

조사일: **2026-10-01~02**. 대상은 **Windows 포터블 환경, D SSD, RAM 약 16GB이며 평소 여유가 적음, Intel Iris 내장 그래픽, 외장 GPU 없음**이다. 정확한 CPU·Iris 세대와 가용 RAM은 아직 확인되지 않았다.

**현재 기본은 실제 CPU 생성을 확인한 Neodragon 채널별 INT8이다.** 기존 AnimateDiff-Lightning 조합은 제작용 기본에서 제외했다. Neodragon은 해변 한 장면에서 512×320·49프레임을 생성했고, 영상 추론·복원을 단계별로 나눠 최대 상주 3.947GiB를 관측했다. LTX-Video 2B distilled와 NOVA 0.6B는 추가 비교 후보다. **실제 그램에서 최적으로 작동한다고 확인된 모델은 아직 없다.** 아래 후기와 공식 배포 조사는 후보 선정의 근거이며, 현재 호스트의 실측과 구분한다.

## 판단 기준

영상 부분 파라미터 수보다 **전체 텍스트 인코더·영상 모델·VAE·실행 환경의 메모리**, 성공한 결과 한 개까지 걸리는 시간, 문장 지시와 동작의 일치, CPU/Intel 실행 근거, 한국에서 사용할 배포의 조건을 우선한다. 짧고 빠른 실패를 반복하는 모델은 효율적인 모델로 평가하지 않는다.

공식 데모·VBench·개발사의 사용자 비교 시험과 실제 이용자의 실행 경험을 구분한다. Reddit/Hugging Face의 경험담은 개인의 자가 보고다. 좋아요·별 수는 품질 점수로 사용하지 않는다. 동일 모델·해상도·프레임·장치 조건이 없는 시간을 나누어 속도 순위를 만들지 않는다.

## 우선순위

| 후보 | 강점과 이용자 평가 | 그램용 판단 |
|---|---|---|
| **Neodragon hybrid** | 큰 T5를 작은 DistilT5로 바꾸고 단계·디코더도 경량화. 휴대폰 CPU/GPU 실행 경험 발견 | **현재 구현한 기본 모델.** Windows CPU·채널별 INT8의 짧은 해변 결과를 확인. 그램 실측과 복잡한 동작은 미검증 |
| **LTX-Video 2B distilled 0.9.6 / 0.9.8** | 적은 추론 단계와 공식 Intel 경로. 0.9.6은 빠른 반복·I2V 결과에 대한 실제 후기가 다수. 흐림·인체 변형·T2V 약점도 보고 | **다음 통합 비교 후보.** 전체 구성 INT4와 단계별 메모리 반환을 검증. 최신 0.9.8도 비교하되 0.9.6 후기·설정을 그대로 적용하지 않음 |
| **NOVA 0.6B** | 작은 영상 본체, Apache-2.0 원문. 공식 예제·저자 벤치마크는 있음 | **3순위 연구 후보.** Phi-2와 디코더·많은 반복 계산을 포함해야 함. 독립 그램/CPU 성능 사례를 이번 조사에서 확보하지 못함 |
| FastWan 2.1 1.3B | 3단계 증류, Apache-2.0. 발표 속도는 H200/4090 | 여유 RAM이 충분할 때 비교. 기존 일반 Wan의 CPU 진단은 약 6.86GiB 상주 메모리·시각 실패를 기록. 그 진단을 증류 FastWan의 품질·속도 측정으로 사용하지 않음 |

이 순서는 **현재 구현 상태와 추가 검증 우선순위**이며 동일 조건의 실측 순위가 아니다. Neodragon은 실제 결과를 확보해 기본으로 연결했으며, LTX의 Intel 경로·후기와 NOVA의 전체 구성은 후속 비교 근거다. 다른 모델이 같은 장치·메모리 범위에서 더 나은 결과를 보이면 교체할 수 있도록 별도 모델 경로와 기록을 유지한다.

## LTX: 속도 호평을 문장 생성 품질로 오해하지 않기

[0.9.6 사용자 스레드](https://www.reddit.com/r/StableDiffusion/comments/1k1o4x8/the_new_ltxvideo_096_distilled_model_is_actually/)에서 작성자는 다수의 결과가 쓸 만하다고 평가했지만 **H100 원격 장치·이미지 입력**을 사용했다. RTX 3060의 빠른 결과를 보고한 댓글도 후속 답변에서 I2V이며 T2V는 좋지 않다고 밝혔다. 다른 댓글은 흐림을 지적했다. “초 단위 생성”을 그램의 예상 시간으로 사용하지 않는다. [버전 비교 후기](https://www.reddit.com/r/StableDiffusion/comments/1k2tj4w/)에도 속도·프롬프트 개선 평가와 품질 불만이 함께 있다.

문장 입력만 받으면서 내부에서 첫 장면을 생성하고 시간축 모델에 조건으로 줄 수 있다. 이후 프레임은 실제 영상 모델이 생성해야 한다. T2V 단독과 자동 첫 프레임 기반 I2V를 **같은 스크립트**로 비교해 추가 이미지 계산이 성공률을 충분히 높이는지 판단한다. 정지 이미지 전환·확대·보간으로 영상 실패를 대체하지 않는다.

[공식 0.9.6 설정](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.6-distilled.yaml)은 8단계·CFG 1의 기본 파이프라인이다. [0.9.8 2B 설정](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.8-distilled.yaml)은 7단계 첫 처리와 3단계 후처리·공간 업스케일러를 명시한다. 같은 2B라는 이유로 샘플링 설정을 섞지 않는다. 프롬프트 보강용 Florence·Llama를 추가 상주 모델로 도입하지 않고 Cursor의 장면 프롬프트를 사용한다.

[Intel 공식 예제](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ltx-video/ltx-video.ipynb)는 OpenVINO의 INT4 변환과 CPU/GPU 선택을 제공한다. 그러나 일반 `Lightricks/LTX-Video`를 사용하므로 **정확한 2B distilled 체크포인트·스케줄러·VAE 조합의 검증이 아니다.** 최신 예제에도 timestep 입력 호환을 위한 변환기 버전 고정이 있다. [과거 Core Ultra 7 iGPU 오류](https://github.com/openvinotoolkit/openvino_notebooks/issues/2830)도 확인했다. 당시 장치는 Iris 그램과 다르며 과거 오류를 현재 모든 Intel GPU로 일반화하지 않는다.

공식 2B 단일 체크포인트는 6,340,744,028바이트이며 텍스트 인코더는 별도다. [city96 변환본](https://huggingface.co/city96/LTX-Video-0.9.6-distilled-gguf/tree/main)의 Q4_0 영상 부분은 1,177,743,872바이트지만 VAE는 2,493,857,180바이트다. **영상 부분 1.18GB가 전체 설치·RAM 1.18GB를 뜻하지 않는다.** GGUF를 OpenVINO가 그대로 실행한다고 가정하지 않는다. 현재 앱의 sd.cpp에 이 구형 LTX 2B 실행을 연결하지 않았다.

## Neodragon: NPU 수치와 범용 실행을 구분하기

[공식 논문·페이지](https://qualcomm-ai-research.github.io/neodragon/)의 3.5GB 최대 RAM·약 6.7초는 **Qualcomm Hexagon NPU를 위한 최적화**다. [공개 모델 카드](https://huggingface.co/Qualcomm-AI-Research/Neodragon)의 큰 T5를 작은 DistilT5로 바꾸는 구조는 그램의 메모리 문제에 맞는 방향이다. 영상 DiT 파일만 약 3.14GB이고 context adapter·CLIP·VAE·첫 프레임 생성 구성도 필요하다. 공개 가중치를 그대로 불러오는 Windows PyTorch 실행이 3.5GB라는 뜻은 아니다.

[실제 Android CPU/GPU 보고](https://www.reddit.com/r/StableDiffusion/comments/1wb45e4/local_video_generation_on_android/)는 OnePlus 12에서 Neodragon INT8·Z Image Turbo INT4로 512×320·49프레임을 약 550~600초에 생성했다고 한다. **개인의 자가 보고이며 Intel Windows 측정이 아니다.** NPU 없이 실행한 근거로는 의미가 있지만 광범위한 호평·그램 속도를 입증하지 않는다. 이 실행의 재현 가능한 변환 코드·Intel 배포는 공개 글에서 확보하지 못했다.

[공식 코드](https://github.com/Qualcomm-AI-research/neodragon)는 연구용 PyTorch 경로다. [Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile)은 Qualcomm NPU 전용이며 앱 코드가 CC BY-NC 4.0이다. [Mobile-OV](https://huggingface.co/leduy99/Mobile-OV)는 CPU 명령을 제공하지만 DreamLite가 CC BY-NC 4.0이므로 상업 사용 가능한 기본 번들로 도입하지 않는다. 이 앱이나 패키지를 외부 프로그램으로 설치시키지 않는다.

## NOVA: 영상 본체 0.6B, 전체 구성은 더 큼

[NOVA 공식 배포](https://huggingface.co/BAAI/nova-d48w1024-osp480)의 실제 파일 크기는 아래와 같다. 합계는 저장 용량이며 동시에 사용하는 RAM 측정이 아니다.

| 부품 | 원본 배포 크기 |
|---|---:|
| 영상 transformer | 1,291,088,736바이트 |
| Phi-2 텍스트 인코더 2개 shard 합계 | 5,559,417,400바이트 |
| 영상 VAE | 478,407,926바이트 |
| 합계 | **7,328,914,062바이트: 약 7.33GB / 6.83GiB** |

저자 예제는 CUDA와 기본 AR 반복 64·확산 단계 25를 명시한다. CPU로 옮길 수 있는 연산과 빠른 CPU 구현은 다르다. 본체만 보고 가장 효율적이라고 선택하지 않는다. [Apache-2.0 고지](https://github.com/baaivision/NOVA)는 유리하며 모델 카드의 연구용 의도·제약도 기록한다.

## 추가로 검토했으나 기본에서 제외한 후보

| 후보 | 판단 근거 |
|---|---|
| Motif-Video 2B | [공식 배포](https://huggingface.co/Motif-Technologies/Motif-Video-2B)는 큰 T5Gemma2와 GPU 중심 수치. [사용자 스레드](https://www.reddit.com/r/StableDiffusion/comments/1smonvh/motifvideo2b/)에는 작은 모델치고 좋다는 의견·인코더 메모리 불만이 함께 있음. 일부는 직접 시험하지 않았다고 명시. 그램용 검증 우선순위에서 제외 |
| SANA-Video 2B | [공식 구현](https://github.com/NVlabs/Sana/blob/main/asset/docs/sana_video.md)은 GPU 실행과 별도 텍스트 모델 필요. [사용자 스레드](https://www.reddit.com/r/StableDiffusion/comments/1rz153l/nvidia_sana_video_2b/)에는 연구 가치 호평·로딩 메모리·품질 불만이 공존. 긴 영상 KV 개선을 전체 RAM 절감으로 해석하지 않음 |
| Kandinsky 5 Video Lite 2B | [공식 최신 코드](https://github.com/kandinskylab/kandinsky-5)는 GPU 12GB 구성을 안내. 오래된 24/48GB 후기를 현재 최소 사양으로 사용하지 않음. [4GB GPU I2V 후기](https://www.reddit.com/r/StableDiffusion/comments/1v1lawz/kandinsky5_lite_i2v_low_vram_workflow_4gb_gpus_5s/)도 외장 GPU·offload 사례이며 그램 총 RAM 근거가 아님 |
| CogVideoX 2B | [공식 배포](https://huggingface.co/THUDM/CogVideoX-2b)의 텍스트 인코더 파일만 약 9.52GB. CUDA offload로 낮춘 VRAM 수치를 시스템 RAM과 동일시하지 않음 |
| MobileWan | [공식 공개 모델](https://huggingface.co/Qualcomm-AI-Research/mobilewan)은 5B와 추가 구성 필요. [80% 선호도](https://qualcomm-ai-research.github.io/MobileWan/)는 개발사의 시험이며 독립 그램 사용자 평가가 아님. 모바일 전용 성능을 Windows에 적용하지 않음 |
| FastWan-QAD / TurboDiffusion | [QAD 카드](https://huggingface.co/FastVideo/FastWan-QAD-1.3B)는 NVFP4·SageAttention3. [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)은 CUDA 최적화. NVIDIA 전용 성능을 Iris의 효율로 인정하지 않음 |
| LTX-2 / 2.3 / 2.5 | 구형 LTX-Video 2B와 별개인 큰 영상·음성 모델. [공식 데스크톱 구현](https://github.com/Lightricks/ltx-desktop)은 Windows 저사양에서 API 경로를 사용하므로 요구와 불일치 |
| SnapGen-V / Mobile Video Diffusion | [SnapGen-V 페이지](https://snap-research.github.io/snapgen-v/)와 [MobileVD 저장소](https://github.com/Qualcomm-AI-research/mobile-video-diffusion)에서 필요한 배포 체크포인트·Intel 실행 경로를 확보하지 못함. 모바일 논문의 속도만으로 바로 설치 가능하다고 표시하지 않음 |
| HunyuanVideo 1.5 | [라이선스의 Territory](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE)가 한국을 제외. 한국용 기본 배포에서 제외 |
| VideoComposer / 비상업 전용 구형 모델 | [공식 고지](https://github.com/ali-vilab/videocomposer)가 연구·비상업 전용. 일반 프로젝트의 상업 배포까지 허용된 모델로 취급하지 않음 |

## 이용 조건

**NOVA의 Apache-2.0 원문을 확인했다.** 일반 FastWan은 [공식 가중치 모델 카드](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers)에서 Apache-2.0 지정과 [FastVideo 코드의 원문](https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE)을 확인했다. 가중치 저장소에는 별도 LICENSE 파일을 확보하지 못했으므로 모델 카드와 코드 라이선스의 출처를 구분한다. LTX 2B는 [현재 원본 배포가 연결하는 LTXV Open Weights License 0.X](https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt)를 따르며 용도·재배포 조건이 있다. 연 매출 **미화 1,000만 달러 이상인 법인의 상업 이용은 별도 유료 계약 대상**이다. 그 미만도 원문 조건을 따라야 한다. 0.9.6 이름의 별도 라이선스 저장소만 받으면 가중치가 설치되는 것은 아니다.

Neodragon·MobileWan은 BSD-3-Clause-Clear와 [Qualcomm Responsible AI License](https://www.qualcomm.com/site/responsible-ai-license)를 함께 따른다. 제한 없는 BSD로 표시하지 않는다. 실제 RAIL과 같은 페이지 아래의 일반 웹사이트 이용 약관을 구분했다. 웹사이트의 비상업 문구를 모델의 비상업 전용 조건으로 혼동하지 않는다. Neodragon README가 요구하는 일반 생성용 안전 검사도 임의 제거하지 않는다. 첫 프레임·인코더·VAE·코드의 각 고지도 보존해야 한다.

## 실제 비교와 통과 기준

1. 모델·설정·변환기·가중치 해시를 고정하고 준비 파일을 D에 둔다. 현재 MCP의 동작 옵션에 등록하기 전에 검증한다.
2. 외장 GPU 없이 같은 스크립트로 파도, 동물이 걷는 장면, 사람이 물건을 드는 장면을 각각 2개 seed로 생성한다. 순수 T2V와 자동 첫 프레임 I2V는 별도로 비교한다.
3. 텍스트 처리·추론·디코딩을 포함한 **총 시간, 시스템 여유 RAM, 프로세스 최대 상주 메모리, Intel 공유 RAM**, 취소 시간을 기록한다. 파일 크기·GPU VRAM만 기록하지 않는다.
4. 저해상도에서 피사체가 식별되고 요청한 동작이 이어져야 한다. 격자·색면·노이즈 변화를 동작 성공으로 판정하지 않는다. 인물 일관성·손·접촉도 눈으로 확인한다.
5. 현재 보호 범위 안에서 품질을 통과한 후보끼리 성공률과 **성공한 클립 한 개까지의 총 시간**을 비교한다. CPU 결과 후 실제 그램 Iris도 검증하고 제작용 기본을 정한다.

모델을 동시에 상주시키지 않는다. 텍스트 임베딩은 D에 캐시하고 텍스트 부품을 종료한 뒤 영상 모델을 실행한다. 영상 모델을 내린 뒤 디코더를 실행한다. 한 장면씩 검토하고 검증한 장면을 캐시한다. Cursor는 장면 요청과 짧은 상태에만 토큰을 쓰고 영상 계산은 로컬에서 처리한다. RAM이 거의 가득 찬 상태를 D SSD만으로 해결한다고 약속하지 않는다.

## 조사 증거와 구현 상태

공개 Hugging Face API에서 **가중치를 다운로드하지 않고** revision, 파일 크기·LFS 해시, 작은 설정·모델 카드·라이선스를 수집했다. [D의 메타데이터 원본](<D:/CursorVideoLocal/audits/model-survey-2026-10-01/catalog.json>)과 저장소별 파일이 근거다. 스냅샷은 조사 시점의 상태이며 앞으로 변할 `main`을 고정 revision과 혼동하지 않는다.

현재 MCP 앱의 정상 제작은 SD 1.5 Q4 첫 이미지와 Neodragon의 시간축 추론을 사용한다. SD/AnimateDiff·Wan은 실패 기록을 보존한 진단 전용 경로다. 환경 준비·MCP 통신과 영상 품질 성공을 구분한다. [최신 결과와 기존 시험](VALIDATION.md)

Neodragon을 처음 준비할 때는 연구 구성 41개 파일, 약 **10.27GB**와 출력 검사기를 고정 revision·해시로 받았다. 초기 시험은 다른 프로세스가 메모리를 사용해 여유 RAM **0.693GiB**에서 시작 보호 기준에 걸렸으며, 다른 프로세스를 종료하지 않았다. 이후 메모리가 확보된 상태에서 계층 공통 INT8의 세부 손실을 확인하고 채널별 INT8과 순차 복원으로 개선했다. 현재 제작용 lock은 SSD 연구 부품을 제외한 29개 파일, **8.16GiB**다. 최종 `22f99f0c7776`은 첫 이미지 캐시 사용 시 **128.640초**, CPU 추론 하위 프로세스 최대 상주 **3.947GiB**를 기록했다. 이는 i9-13900HX CPU 4스레드의 실측이며 그램이나 NPU의 측정이 아니다. [준비·차단·개선 이력](NEODRAGON_CPU_PROBE.md)
