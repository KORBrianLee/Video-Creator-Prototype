> 0.7.0 현재 기준: LG 그램 16GB·Intel 내장 GPU를 기준으로 한 저메모리 OpenCL 스트리밍 경로와 C/D SSD 선택을 추가했다. 기본 preview는 384×256이며 GPU 버퍼 64MiB·작업 RAM 5GiB 상한이다. 아래는 이전 버전의 조사·측정 이력이다. RTX의 시간은 그램 성능이 아니며 현장 사실감과 Iris 실행은 미검증이다. 현재 변경과 실측은 [LOW_RESOURCE_GPU.md](LOW_RESOURCE_GPU.md)를 우선한다.

## 0.7.0의 그램 전달

`runtime/python`, `runtime/neodragon-python`, `engines/cpu`, `engines/experimental-neodragon`, 기본 Neo 모델과 SD 1.5 Q4, `models/experimental-safety-checker`, `app`, `datasets/construction-v1`, `cache/conditioning`을 선택한 D SSD의 같은 구조로 복사한다. GPU 스트리밍에는 `models/experimental-neodragon/derived`의 INT8 사본이 필수가 아니다. CPU INT8도 쓰려면 사본과 provenance를 함께 가져간다. OpenCL은 그램에 설치된 Intel GPU 드라이버가 제공하며 드라이버를 이 묶음에 재배포하지 않는다.

복사한 그램의 Python으로 `app/installer.py deploy --runtime-dir D:\CursorVideoLocal`을 실행하면 저장 경로와 `.cursor/mcp.json`을 D용으로 다시 연결한다. 스레드·메모리 보호 등 사용자의 자원 설정은 유지한다. `video.cmd doctor`에서 Intel GPU 감지와 여유 RAM을 확인한다. `runtime/neodragon-python/python.exe app/tools/check_gpu_math.py --backend intel-gpu --output D:\CursorVideoLocal\audits\iris-math.json`으로 해당 Intel 드라이버의 실제 수학을 점검할 수 있다. 이것이 영상 속도·현장 사실감 검증을 대신하지 않는다.

# LG 그램용 포터블 구성

이미 준비한 D SSD를 그램에서 같은 드라이브 이름으로 연결하면 설치를 반복하지 않습니다. Cursor에서 `D:\CursorVideoLocal\app`을 열어 `video_doctor`를 확인합니다. 모델 경로와 임시 경로가 C로 우회하지 않습니다. 동일 D 경로를 권장합니다.

다른 D SSD로 복사할 때 필요한 구성은 다음과 같습니다.

- `app` 전체. 설명서, Cursor 설정, 소스, 라이선스, `examples/validated-beach.mp4` 기준 샘플을 포함합니다.
- `runtime/python`, `runtime/neodragon-python` 전체. 패키지 라이선스와 dist-info도 유지합니다.
- `engines/cpu`, `engines/experimental-neodragon` 전체. 원저자 코드·LICENSE·RAIL 사본과 출처 기록을 유지합니다.
- `neodragon.lock.json`의 `files[].filename`에 기록된 모델 29개. 합계 8.16GiB이며 여기에는 영상 원본 가중치·텍스트 처리·VAE·안전 검사·SD 첫 이미지 모델이 포함됩니다.
- `models/experimental-neodragon/derived/transformer-cpu-int8-v2.pt`와 같은 이름의 `.provenance.json`. 약 1.52GiB. 반드시 함께 복사합니다.
- `models/neodragon.lock.json`은 설치기에서 만든 출처 사본입니다. 앱 lock이 실제 검증 기준입니다.
- 현장용 `datasets/construction-v1` 전체. 3개 원본·9개 구간·참고 프레임·출처/라이선스·조건 계산 준비 기록을 함께 옮깁니다. 실사 템플릿은 SHA256으로 고정된 참고 화면이 없으면 실행하지 않습니다. 이는 학습된 가중치가 아닙니다.

`tmp`, `jobs`, `cache`는 실행할 때 D에서 만들어집니다. 기존 결과를 보존하려면 함께 복사합니다. `cache/first-frames`, `cache/shots`, `cache/conditioning`도 함께 옮기면 검증된 범위에서 계산을 재사용합니다. SSD의 모델 크기는 RAM 사용량과 다릅니다.

이 새 구성에는 기존 Wan·AnimateDiff 모델, SSD1B 첫 프레임 가중치, 이전 실패 실험·다운로드 사본·옛 INT8 v1 사본이 필요하지 않습니다. 현재 개발 D에서는 비교와 원본 보존을 위해 남겨 두었습니다. 이 목록은 필요한 구성만 복사할 때 쓰며 기존 파일을 지우라는 지시가 아닙니다.

새 SSD에 소스만 설치하면 `setup.cmd`가 필요한 모델과 CPU 라이브러리를 고정 버전으로 받고 최초 INT8을 준비합니다. 기준 샘플이 없는 새 설치에서는 진단 생성과 실제 결과 검토 후 검증 기록을 만들어야 합니다. 지금 준비된 `app`과 모델을 함께 옮기는 방식이 바로 사용할 수 있는 배포입니다. 최초 양자화는 7.5GiB 여유 RAM이 필요하며 준비된 변환본을 옮기면 다시 실행하지 않습니다.

이 구성 목록은 개인 작업 환경의 이동을 위한 것입니다. 프로그램 실행 파일이나 모델 번들을 제삼자에게 재배포할 때는 [라이선스 고지](MODEL_LICENSES.md)의 원문·변경 고지·용도 제한과 GPL 실행 파일의 해당 소스 제공 의무를 함께 이행해야 합니다.

그램용 새 설치는 `setup-gram.cmd`를 실행하면 D SSD와 GPU 자동 선택을 함께 적용한다. 기존 CPU 설정도 이 설치 명령의 명시적인 auto 선택으로 갱신하되 메모리·스레드 사용자 설정은 유지한다.
