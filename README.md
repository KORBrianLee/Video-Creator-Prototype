# Cursor 로컬 영상 생성기 0.7.3

**설계 기준은 LG 그램: RAM 약 16GB, Intel Iris 내장 GPU, 평소 RAM 여유가 적은 Windows 노트북이다.** 더 좋은 컴퓨터에서는 아래 자원 한계가 자동으로 넓어지고, 그램에서는 검증된 값으로 동작한다.

## 설치와 업데이트

1. 저장소를 원하는 폴더에 clone한다. 실행 폴더(`C:\CursorVideoRuntime`) 안에 clone하지 않는다.
2. 처음 한 번 `setup.cmd`를 실행한다. Python·엔진·모델 약 9GB를 해시 검증하며 받고, 현장 참고 자료를 준비하고, 실행 폴더 `C:\CursorVideoRuntime\app`에 배포한다. D 드라이브에 설치하려면 `setup.cmd -RuntimeDir D:\CursorVideoLocal`(그램은 `setup-gram.cmd`).
3. 이후에는 `update.cmd`만 실행한다. `git pull` 후 코드만 다시 배포하고, lock 파일이 바뀐 경우에만 모델을 받아 검증한다. 실행 폴더의 `video.config.json` 설정은 유지된다.
4. Cursor에서 `C:\CursorVideoRuntime\app` 폴더를 열고 `local-video` MCP를 다시 시작한 뒤 `video_doctor`로 확인한다.

Cursor에서 설치된 `app` 폴더를 열고 `local-video` MCP를 사용한다. 대본은 Cursor가 한 번 짧은 장면 지시로 바꾸고, 모델 추론은 컴퓨터에서 진행한다. 외부 제작 GUI·클라우드 생성 API는 사용하지 않는다.

기본 `auto`는 감지한 외장 GPU를 우선 선택하고, 없으면 내장 GPU를 사용한다. Neo 영상 모델의 Linear 연산은 OpenCL GPU에서 계산하고, 텍스트·attention·VAE·안전 검사는 CPU에서 순서대로 실행한다. 전체 모델을 GPU에 올리지 않고 SSD에 매핑한 BF16 행렬 하나씩 전달한다. 드라이버 추가 메모리는 GPU 버퍼 수치에 포함되지 않으며 시스템 여유 RAM을 별도로 감시한다.

## 적응형 자원 한계

`video.config.json`의 자원 값은 기본 `"auto"`이며, 각 단계를 시작할 때마다 지금 컴퓨터 상태로 다시 계산한다. 숫자를 넣으면 그 값을 고정 한계로 쓴다. `video_doctor`의 `resource_plan`에서 계산 결과를 확인할 수 있다.

| 항목 | `auto` 계산 | 16GB 그램 예 |
|---|---|---|
| `threads` | 성능 코어 수(하이브리드 CPU의 효율 코어 제외), 최대 8 | 4 |
| `reserve_ram_gib` | 전체 RAM의 9%, 1~4GiB | 약 1.4GiB |
| `minimum_free_ram_gib` | 모델 요구량(Neo 3GiB)과 보호 RAM+1GiB 중 큰 값 | 3GiB |
| `maximum_working_set_gib` | 지금 여유 RAM−보호 RAM, 5GiB~전체 RAM의 50% | 5GiB |
| `gpu_buffer_mib` | 내장 GPU: (여유 RAM−보호 RAM)/24, 32~256MiB. 외장 GPU: VRAM의 1/4, 최대 1GiB | 64~100MiB |

실행 중에도 대응한다. 내장 GPU 버퍼는 여유 RAM이 보호 RAM+1GiB 아래로 내려가면 절반씩 줄이고(최소 32MiB) 회복되면 다시 늘린다. 여유 RAM이 보호 기준 아래로 잠깐 내려가는 것은 15초까지 허용하고, 보호 기준의 75% 아래로 떨어지면 바로 중지한다. RAM 부족으로 중지된 작업은 RAM이 회복될 때까지 최대 5분 기다린 뒤 두 번까지 이어서 다시 시도하며, 완료된 장면은 캐시에서 재사용한다. 버퍼 크기는 계산 결과를 바꾸지 않으므로 캐시 키에 넣지 않는다.

기본 `preview`는 **384×256·49프레임·24fps, 약 2초**다. `quality`는 512×320이다. 증류 모델의 실제 생성 단계는 3단계를 유지한다. GPU 경로는 원본 BF16을 FP32로 계산하여 기존 INT8의 활성값 양자화 손실을 피한다. 이 변경이 작업자 형태나 Sora2 사실감을 보장하지 않는다. 정지 이미지 연결·프레임 보간으로 대체하지 않는다.

`video_doctor`로 준비·GPU 선택·여유 RAM·품질 기록을 확인한다. RAM이 부족하면 시작을 거절하거나 이 작업만 중지한다. 위 한계는 보호 정책이며 실제 최소 사양이 아니다. 가능하면 여유 RAM 5~6GiB를 확보한다. SSD가 RAM 부족을 없애지는 않는다. GPU를 강제 지정했는데 감지하지 못하면 CPU로 몰래 바꿔 실행하지 않는다.

현장 영상은 `video_site_presets`의 8개 장면 또는 `domain="construction"`의 직접 작성 장면을 사용한다. **중장비와 작업자의 자연스러운 협업·형태 유지 품질은 아직 제작 검증을 통과하지 않았다.** 사용자가 요청한 개발 초안에는 `diagnostic=true`를 사용한다. 참고 영상·사진과 프롬프트 계산을 준비했지만 모델 가중치를 학습한 것은 아니다. 2초부터 확인하며 4초는 품질 실패 이력, 8초는 미검증이다. 라이선스가 있는 참고 화면을 쓰면 결과의 `ATTRIBUTION.md`를 함께 유지한다. 안전 검사에서 거부된 영상은 정상 완료로 내보내지 않는다.

현재 모델은 저메모리 실행을 구현한 Neo다. 완벽한 최적 모델로 확정하지 않았다. LTX-Video 2B distilled/OpenVINO는 품질 비교 후보이며 아직 통합·그램 검증을 하지 않았다. Wan 1.3B Q4는 별도 진단 경로이고 큰 텍스트 인코더와 해독 RAM 때문에 그램 기본에서 제외한다. [후보와 이용자 보고](docs/LOW_RESOURCE_GPU.md), [기존 비교 조사](docs/MODEL_SELECTION.md)를 참고한다.

`video.config.json`의 `backend`는 `auto`, `gpu`, `intel-gpu`, `cpu`를 지원하며 Wan 전용 기존 `intel-vulkan`도 유지한다. 참고 이미지·캐시·임시 파일·모델·출력은 선택한 전용 실행 폴더 안에 둔다. 기존 D 환경과 원본 USB 아카이브는 보존했다. [포터블 구성](docs/PORTABLE.md).

같은 입력의 완료 영상은 해시와 완전 해독 검사를 거쳐 재사용한다. 대본·시작 사진의 조건 계산도 로컬에서 재사용한다. GPU·드라이버·모델·계산 코드가 바뀌면 해당 캐시를 구분한다. 생성 중에는 `video_wait`로 최대 60초 기다려 Cursor의 잦은 상태 조회를 줄인다.

프로젝트 코드는 MIT다. 기본 공개 가중치는 BSD-3-Clause-Clear와 Qualcomm Responsible AI License, 시작 이미지 모델은 Open RAIL-M 조건을 따른다. 무조건적인 MIT/Apache 번들로 표시하지 않는다. [이용·재배포 조건](docs/MODEL_LICENSES.md).
