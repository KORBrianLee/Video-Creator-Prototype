# Cursor 로컬 영상 생성기 0.7.3

> **License notice (worldwide).** Free **only for an individual's own personal learning**. Companies, private academies, universities and other institutions that use this software in teaching (for example vision AI courses) **with state or public funding, or while charging students any fee, must buy a license or obtain prior written permission**. Commercial use, modification, derivative works, redistribution and hosted services also require prior written permission. Violations will be pursued for damages to the fullest extent permitted by applicable law. Full terms (Korean and English, equally authentic): [LICENSE](LICENSE). Contact: [KORBrianLee](https://github.com/KORBrianLee).
>
> **라이선스 안내.** 개인의 학습 목적 사용만 무료입니다. 국가·공공 지원을 받거나 수강료를 받고 수업을 진행하는 회사·학원·대학·기관은 라이선스 구입 또는 사전 서면 허락이 필요합니다. 전문은 [LICENSE](LICENSE)를 보세요.

**설계 기준은 LG 그램: RAM 약 16GB, Intel Iris 내장 GPU, 평소 RAM 여유가 적은 Windows 노트북이다.** 더 좋은 컴퓨터에서는 아래 자원 한계가 자동으로 넓어지고, 그램에서는 검증된 값으로 동작한다.

## 설치와 업데이트

1. 저장소를 원하는 폴더에 clone한다. 실행 폴더(`D:\VideoCreator\CursorVideoRuntime`) 안에 clone하지 않는다.
2. 처음 한 번 `setup.cmd`를 실행한다. Python·CPU 엔진·Vulkan GPU 엔진·모델 약 9GB를 해시 검증하며 받고, 현장 참고 자료를 준비하고, 실행 폴더 `D:\VideoCreator\CursorVideoRuntime\app`에 배포한다. 모델·캐시·임시 파일·연산은 모두 D 실행 폴더에서 처리하며, D 볼륨이 없는 PC에서만 `C:\CursorVideoRuntime`을 쓴다. 다른 D 폴더는 `setup.cmd -RuntimeDir <D 경로>`(그램은 `setup-gram.cmd`).
3. 이후에는 `update.cmd`만 실행한다. `git pull` 후 코드만 다시 배포하고, lock 파일이 바뀐 경우에만 모델을 받아 검증한다. Vulkan GPU 엔진이 없는 PC(예전 설치)는 업데이트할 때 자동으로 받는다. 첫 장면 이미지는 이 엔진으로 GPU에서 만들고, 엔진이 없거나 실패하면 CPU로 자동 전환한다. GPU 엔진을 받지 않으려면 `installer.py ... --no-vulkan`으로 실행한다. 실행 폴더의 `video.config.json` 설정은 유지된다.
4. Cursor에서 `D:\VideoCreator\CursorVideoRuntime\app` 폴더를 열고 `local-video` MCP를 다시 시작한 뒤 `video_doctor`로 확인한다.

Cursor에서 설치된 `app` 폴더를 열고 `local-video` MCP를 사용한다. 대본은 Cursor가 한 번 짧은 장면 지시로 바꾸고, 모델 추론은 컴퓨터에서 진행한다. 외부 제작 GUI·클라우드 생성 API는 사용하지 않는다.

기본 `auto`는 감지한 외장 GPU를 우선 선택하고, 없으면 내장 GPU를 사용한다. 설치기는 이 컴퓨터의 주 GPU를 감지해 맞는 PyTorch GPU 런타임 하나만 받는다(해시 고정, 각각 별도 폴더). NVIDIA는 CUDA(`runtime/cuda-site`, Turing 이상·RTX 50은 CUDA 12.8, Maxwell~Volta는 12.6), Intel 내장·Arc는 XPU(`runtime/xpu-site`)다. AMD와 GPU 없는 PC는 받지 않고 Vulkan·OpenCL·CPU 경로를 쓴다. 설치 직후 GPU 동작 확인에 통과하면 Neo 영상 모델의 트랜스포머(Linear·attention·정규화)와 디코더(3D 컨볼루션)를 그 GPU에서 계산한다. 외장 GPU는 여유 VRAM에 모델 가중치가 들어가면 한 번만 올려 상주시키고, 모자라거나 내장 GPU(공유 RAM)면 단계마다 올린다. 어텐션 조각 크기도 외장은 여유 VRAM, 내장은 여유 RAM에 맞춘다. GPU 단계가 실행 중 실패하면 그 단계를 CPU/OpenCL 경로로 다시 실행하고, 그 백엔드는 1시간 동안 쓰지 않는다. `video_doctor`의 `accelerator`에서 선택 결과와 사유를 볼 수 있다. 첫 장면 이미지는 Vulkan 엔진이 GPU에서 만든다. 전체 모델을 GPU에 올리지 않고 SSD에 매핑한 BF16 행렬을 단계마다 올려 FP32로 펴서 곱한다. 이 인텔 드라이버는 자료형 변환 커널이 실패하고 FP64가 없어서, BF16은 비트 연산으로 정확히 펴고 작은 FP64 텐서는 CPU에서 만든 뒤 FP32로 옮긴다. GPU 커널은 처음 쓸 때 드라이버가 컴파일하므로 첫 실행만 몇 분 더 걸린다. GPU 확인이 실패하면 OpenCL GPU 행렬 연산과 CPU로 자동 전환한다(`--no-gpu-torch`로 설치를 건너뛸 수 있다). CUDA 경로는 이 개발 PC(Intel 내장 GPU)에서 실제 하드웨어로 시험하지 못했고, 코드 논리와 폴백만 테스트로 검증했다. 텍스트 인코더·VAE 인코더·안전 검사는 CPU에서 실행한다. 드라이버 추가 메모리는 GPU 버퍼 수치에 포함되지 않으며 시스템 여유 RAM을 별도로 감시한다.

## 적응형 자원 한계

`video.config.json`의 자원 값은 기본 `"auto"`이며, 각 단계를 시작할 때마다 지금 컴퓨터 상태로 다시 계산한다. 숫자를 넣으면 그 값을 고정 한계로 쓴다. `video_doctor`의 `resource_plan`에서 계산 결과를 확인할 수 있다.

| 항목 | `auto` 계산 | 16GB 그램 예 |
|---|---|---|
| `threads` | 성능 코어 수(하이브리드 CPU의 효율 코어 제외), 최대 8 | 4 |
| `reserve_ram_gib` | 전체 RAM의 9%, 1~4GiB | 약 1.4GiB |
| `minimum_free_ram_gib` | 모델 요구량(Neo 3GiB)과 보호 RAM+1GiB 중 큰 값 | 3GiB |
| `maximum_working_set_gib` | 지금 여유 RAM−보호 RAM, 5GiB~전체 RAM의 50% | 5GiB |
| `gpu_buffer_mib` | 내장 GPU: (여유 RAM−보호 RAM)/24, 32~256MiB. 외장 GPU: VRAM의 1/4, 최대 1GiB | 64~100MiB |

실행 중에도 대응한다. 내장 GPU 버퍼는 여유 RAM이 보호 RAM+1GiB 아래로 내려가면 절반씩 줄이고(최소 32MiB) 회복되면 다시 늘린다. 여유 RAM이 보호 기준 아래로 잠깐 내려가는 것은 15초까지 허용하고, 보호 기준의 75% 아래로 떨어지면 바로 중지한다. RAM 부족으로 중지된 작업은 RAM이 회복될 때까지 최대 5분 기다린 뒤 두 번까지 이어서 다시 시도하며, 완료된 장면은 캐시에서 재사용한다. 버퍼 크기는 계산 결과를 바꾸지 않으므로 캐시 키에 넣지 않는다. 단계별 RAM 학습값은 그 단계 프로세스 자신의 사용량을 넘지 않게 해서, 작업 중 다른 앱이 늘어난 몫을 요구량으로 배우지 않는다.

스로틀링 대응: 엔진 프로세스는 데스크톱이 느려지지 않게 낮은 우선순위로 실행하되, Windows 전원 스로틀링(EcoQoS)은 끈다. 창 없는 백그라운드 작업이라는 이유로 효율 코어·낮은 클럭으로 밀리지 않는다. CPU 성능 제한(`% Performance Limit`)을 실행 중에 기록하고, 새 단계를 시작할 때 발열·전력 한계로 60% 아래로 묶여 있으면 최대 90초 식힌 뒤 시작한다. `video_doctor`의 `resource_plan.power`에 AC/배터리·절전 모드를 표시한다. 배터리·절전 모드에서는 Windows와 펌웨어가 클럭을 낮추므로 AC 전원을 권장한다. 발열 자체는 하드웨어 한계라 완전히 없앨 수 없다.

기본 `preview`는 **384×256·49프레임·24fps, 약 2초**다. `quality`는 512×320이다. 증류 모델의 실제 생성 단계는 3단계를 유지한다. GPU 경로는 원본 BF16을 FP32로 계산하여 기존 INT8의 활성값 양자화 손실을 피한다. 이 변경이 작업자 형태나 Sora2 사실감을 보장하지 않는다. 정지 이미지 연결·프레임 보간으로 대체하지 않는다.

`video_doctor`로 준비·GPU 선택·여유 RAM·품질 기록을 확인한다. RAM이 부족하면 시작을 거절하거나 이 작업만 중지한다. 위 한계는 보호 정책이며 실제 최소 사양이 아니다. 가능하면 여유 RAM 5~6GiB를 확보한다. SSD가 RAM 부족을 없애지는 않는다. GPU를 강제 지정했는데 감지하지 못하면 CPU로 몰래 바꿔 실행하지 않는다.

현장 영상은 `video_site_presets`의 8개 장면 또는 `domain="construction"`의 직접 작성 장면을 사용한다. **중장비와 작업자의 자연스러운 협업·형태 유지 품질은 아직 제작 검증을 통과하지 않았다.** 사용자가 요청한 개발 초안에는 `diagnostic=true`를 사용한다. 참고 영상·사진과 프롬프트 계산을 준비했지만 모델 가중치를 학습한 것은 아니다. 2초부터 확인하며 4초는 품질 실패 이력이 있고, 8초 이상은 현장 품질을 검증하지 않았다.

## 긴 영상(최대 15초)과 장시간 전용 모델

`continuous=true`에서 `duration_seconds`는 2, 4, 8, 10, 15를 받는다. 어느 길이든 한 번의 생성으로 이어서 만들며, 짧은 클립을 이어 붙이지 않는다. 모델은 컴퓨터마다 자동으로 고른다.

| 컴퓨터 | 모델 | 15초 |
|---|---|---|
| NVIDIA GPU(VRAM 8GB 이상) + 장시간 모델 설치 | SkyReels-V2 DF 1.3B (Diffusion Forcing, Wan 계열) | 377프레임, 24fps. 해상도와 블록 길이는 VRAM에 맞춘다(8GB 672×384 … 20GB 이상 960×544) |
| 그 외(그램 등 Intel 내장 GPU, AMD, CPU) | Neo | 361프레임, 24fps, 384×256 |

SkyReels는 블록마다 앞 블록의 끝 17프레임을 겹쳐 보며 생성하는 모델이라 긴 영상이 한 번에 이어진다. 설치기는 NVIDIA GPU가 감지된 PC에만 이 모델을 받는다. 원본은 FP32 27GB인데, 파일마다 해시를 확인하고 바로 BF16으로 바꿔 최종 약 15GB만 남긴다(받는 중 추가 여유 약 5GB 필요, 중단되면 이어서 받음). UMT5 텍스트 인코더는 VRAM이 28GB 이상일 때만 GPU에 두고, 그 밖에는 프롬프트당 한 번 CPU에서 실행한다. SkyReels가 실패하면(예: VRAM 부족) 같은 작업을 Neo로 다시 만들고, 1시간 동안 SkyReels를 쓰지 않는다. `video_doctor`의 `accelerator`에 `video_model`, `model_tier_reason`, `maximum_duration_seconds`가 나온다. 라이선스는 [Skywork Community License](docs/licenses/SkyReels-V2-Skywork-LICENSE.txt)이며 그 조건 안에서 상업적 사용도 허용된다.

SkyReels 경로는 이 개발 PC(Intel 내장 GPU)에서 실제로 실행하지 못했고 단위 테스트로만 검증했다. NVIDIA PC에서 `update.cmd` → `video_doctor` → 15초 작업 한 번으로 확인이 필요하다.

**Neo는 8초까지만 정식 생성한다.** 그램(Iris Xe, XPU)에서 15초(361프레임)를 실측한 결과 영상 추론 1,237초·해독 70초였고, 장면이 약 5초까지 유지되다가 7.5초 무렵부터 어두워져 끝은 파란 잡음과 검은 화면이 되었다. 안전 검사기도 이 프레임들을 거부했다. 그래서 Neo만 쓰는 컴퓨터는 10/15초를 `diagnostic=true` 시험으로만 받고, 정식 요청은 제출 단계에서 사유와 함께 거절한다. `video_doctor`의 `maximum_duration_seconds`가 Neo 컴퓨터에서 8, 장시간 모델 컴퓨터에서 15로 표시된다. 라이선스가 있는 참고 화면을 쓰면 결과의 `ATTRIBUTION.md`를 함께 유지한다. 안전 검사에서 거부된 영상은 정상 완료로 내보내지 않는다.

현재 모델은 저메모리 실행을 구현한 Neo다. 완벽한 최적 모델로 확정하지 않았다. LTX-Video 2B distilled/OpenVINO는 품질 비교 후보이며 아직 통합·그램 검증을 하지 않았다. Wan 1.3B Q4는 별도 진단 경로이고 큰 텍스트 인코더와 해독 RAM 때문에 그램 기본에서 제외한다. [후보와 이용자 보고](docs/LOW_RESOURCE_GPU.md), [기존 비교 조사](docs/MODEL_SELECTION.md)를 참고한다.

`video.config.json`의 `backend`는 `auto`, `gpu`, `intel-gpu`, `cpu`를 지원하며 Wan 전용 기존 `intel-vulkan`도 유지한다. 참고 이미지·캐시·임시 파일·모델·출력은 선택한 전용 실행 폴더 안에 둔다. 기존 D 환경과 원본 USB 아카이브는 보존했다. [포터블 구성](docs/PORTABLE.md).

같은 입력의 완료 영상은 해시와 완전 해독 검사를 거쳐 재사용한다. 대본·시작 사진의 조건 계산도 로컬에서 재사용한다. GPU·드라이버·모델·계산 코드가 바뀌면 해당 캐시를 구분한다. 생성 중에는 `video_wait`로 최대 60초 기다려 Cursor의 잦은 상태 조회를 줄인다.

프로젝트 코드와 문서는 [개인 학습 무료·기관 교육 유료 라이선스(변경 금지) 1.1](LICENSE)을 따른다. 개인의 학습 목적 사용만 무료다. 국가·공공 지원을 받거나 수강료를 받고 수업(비전 AI 등)을 진행하는 회사·학원·대학·교육기관은 라이선스 구입 또는 사전 서면 허락이 필요하다. 상업적 사용·개조·변형·재배포도 사전 서면 허락이 필요하며, 위반 시 관계 법령이 허용하는 최대 범위의 손해배상을 청구할 예정이다. 이 파일이 포함되기 전 MIT로 공개된 버전은 당시 받은 사람에게 MIT가 유지된다. 기본 공개 가중치는 BSD-3-Clause-Clear와 Qualcomm Responsible AI License, 시작 이미지 모델은 Open RAIL-M 조건을 따른다. 무조건적인 MIT/Apache 번들로 표시하지 않는다. [이용·재배포 조건](docs/MODEL_LICENSES.md).
