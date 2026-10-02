# Cursor 로컬 영상 생성기 0.7.0

**설계 기준은 LG 그램: RAM 약 16GB, Intel Iris 내장 GPU, 평소 RAM 여유가 적은 Windows 노트북이다.** 이 컴퓨터의 RTX 성능을 기준으로 해상도·메모리 상한을 늘리지 않는다. 현재 호스트에서는 사용자가 허용한 C SSD를 활용한다. 그램의 USB 포터블 환경에서는 D SSD를 명시적으로 선택할 수 있다.

Cursor에서 설치된 `app` 폴더를 열고 `local-video` MCP를 사용한다. 현재 실행 폴더는 `C:\CursorVideoRuntime\app`이다. 대본은 Cursor가 한 번 짧은 장면 지시로 바꾸고, 모델 추론은 컴퓨터에서 진행한다. 외부 제작 GUI·클라우드 생성 API는 사용하지 않는다.

기본 `auto`는 감지한 외장 GPU를 우선 선택하고, 없으면 내장 GPU를 사용한다. Neo 영상 모델의 Linear 연산은 OpenCL GPU에서 계산하고, 텍스트·attention·VAE·안전 검사는 CPU에서 순서대로 실행한다. 전체 모델을 GPU에 올리지 않고 SSD에 매핑한 BF16 행렬 하나씩 전달한다. 명시 GPU 버퍼 상한은 **64MiB**, 작업 상주 RAM 상한은 **5GiB**, CPU는 최대 **4스레드**다. 드라이버 추가 메모리는 이 버퍼 수치에 포함되지 않으며 시스템 여유 RAM을 별도로 감시한다.

기본 `preview`는 **384×256·49프레임·24fps, 약 2초**다. `quality`는 512×320이다. 증류 모델의 실제 생성 단계는 3단계를 유지한다. GPU 경로는 원본 BF16을 FP32로 계산하여 기존 INT8의 활성값 양자화 손실을 피한다. 이 변경이 작업자 형태나 Sora2 사실감을 보장하지 않는다. 정지 이미지 연결·프레임 보간으로 대체하지 않는다.

`video_doctor`로 준비·GPU 선택·여유 RAM·품질 기록을 확인한다. RAM이 부족하면 시작을 거절하거나 이 작업만 중지한다. 시작 여유 3GiB, 실행 중 시스템 여유 1.5GiB는 보호 정책이며 실제 최소 사양이 아니다. 가능하면 여유 RAM 5~6GiB를 확보한다. SSD가 RAM 부족을 없애지는 않는다. GPU를 강제 지정했는데 감지하지 못하면 CPU로 몰래 바꿔 실행하지 않는다.

현장 영상은 `video_site_presets`의 8개 장면 또는 `domain="construction"`의 직접 작성 장면을 사용한다. **중장비와 작업자의 자연스러운 협업·형태 유지 품질은 아직 제작 검증을 통과하지 않았다.** 사용자가 요청한 개발 초안에는 `diagnostic=true`를 사용한다. 참고 영상·사진과 프롬프트 계산을 준비했지만 모델 가중치를 학습한 것은 아니다. 2초부터 확인하며 4초는 품질 실패 이력, 8초는 미검증이다. 라이선스가 있는 참고 화면을 쓰면 결과의 `ATTRIBUTION.md`를 함께 유지한다. 안전 검사에서 거부된 영상은 정상 완료로 내보내지 않는다.

현재 모델은 저메모리 실행을 구현한 Neo다. 완벽한 최적 모델로 확정하지 않았다. LTX-Video 2B distilled/OpenVINO는 품질 비교 후보이며 아직 통합·그램 검증을 하지 않았다. Wan 1.3B Q4는 별도 진단 경로이고 큰 텍스트 인코더와 해독 RAM 때문에 그램 기본에서 제외한다. [후보와 이용자 보고](docs/LOW_RESOURCE_GPU.md), [기존 비교 조사](docs/MODEL_SELECTION.md)를 참고한다.

새 설치는 `setup.cmd`로 C SSD에 준비한다. 그램의 D 저장은 `setup-gram.cmd` 또는 `setup.cmd -RuntimeDir D:\CursorVideoLocal`로 지정한다. `video.config.json`의 `backend`는 `auto`, `gpu`, `intel-gpu`, `cpu`를 지원하며 Wan 전용 기존 `intel-vulkan`도 유지한다. 참고 이미지·캐시·임시 파일·모델·출력은 선택한 전용 실행 폴더 안에 둔다. 기존 D 환경과 원본 USB 아카이브는 보존했다. [포터블 구성](docs/PORTABLE.md).

같은 입력의 완료 영상은 해시와 완전 해독 검사를 거쳐 재사용한다. 대본·시작 사진의 조건 계산도 로컬에서 재사용한다. GPU·드라이버·모델·계산 코드가 바뀌면 해당 캐시를 구분한다. 생성 중에는 `video_wait`로 최대 60초 기다려 Cursor의 잦은 상태 조회를 줄인다.

프로젝트 코드는 MIT다. 기본 공개 가중치는 BSD-3-Clause-Clear와 Qualcomm Responsible AI License, 시작 이미지 모델은 Open RAIL-M 조건을 따른다. 무조건적인 MIT/Apache 번들로 표시하지 않는다. [이용·재배포 조건](docs/MODEL_LICENSES.md).
