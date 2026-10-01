# Video Creator Prototype (Cursor용 비공식 도구)

> 이 프로젝트는 개인이 만든 비공식 오픈소스 도구이며, Cursor 및 Anysphere, Inc.와 아무런 제휴·후원·보증 관계가 없습니다. "Cursor"는 해당 소유자의 상표이며, 이 도구가 Cursor 편집기에서 동작한다는 사실을 설명하기 위해서만 사용합니다.

Cursor 에이전트 채팅에 영상 내용을 쓰면, 에이전트가 장면 이미지를 만들고 MP4 파일까지 완성하는 도구입니다. 외부 영상 서비스 가입이나 별도 로그인이 없습니다. 이미지는 Cursor에 기본으로 있는 이미지 생성으로 만들고, 영상은 내 컴퓨터에서 렌더링합니다(GPU 영상 인코더가 있으면 사용).

구성은 다음과 같습니다.

- `.cursor/skills/cursor-video/SKILL.md`: 에이전트가 장면을 기획하고, 이미지를 만들고, 스토리보드를 쓰고, 렌더링까지 하게 하는 스킬.
- `.cursor/rules/video-storage.mdc`: 모든 영상 파일을 D 드라이브 저장 폴더에 두라는 규칙.
- `setup.py`, `vc_config.py`, `install.bat`, `install.sh`: 설치 확인과 첫 실행 환경 선택 창, 설정 저장.
- `video_render.py`: 에이전트가 터미널에서 실행하는 렌더러. 스토리보드를 ffmpeg로 MP4로 만듭니다. Python 표준 라이브러리만 씁니다.
- `keyframes_to_video.py`: 키프레임 사진 사이를 보간해 부드러운 영상으로 만드는 도구.
- `studio.html`: 브라우저 렌더러. 미리보기·직접 편집용이고, 에이전트가 터미널을 쓸 수 없을 때의 대체 수단입니다.

## 설치

필요한 것:

- Cursor(이미지 생성이 되는 플랜)
- 채팅만으로 MP4까지 받으려면: Python 3.8 이상, ffmpeg
  - Windows: `winget install --id Gyan.FFmpeg -e`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - 설치되어 있지 않으면 에이전트가 설치해도 되는지 먼저 묻습니다.
- 브라우저 렌더러를 쓸 때만: Chrome 또는 Edge 최신 버전

### 방법 1: 이 폴더를 Cursor로 열기 (가장 간단)

1. 이 폴더를 원하는 곳에 둡니다.
2. Cursor에서 **File → Open Folder**로 이 폴더를 엽니다. 폴더 안의 스킬이 자동으로 잡힙니다.
3. 에이전트 채팅에서 바로 요청합니다.

### 방법 2: 모든 프로젝트에서 쓰기

`.cursor/skills/cursor-video` 폴더를 사용자 스킬 폴더로 복사합니다.

- Windows: `%USERPROFILE%\.cursor\skills\cursor-video`
- macOS / Linux: `~/.cursor/skills/cursor-video`

Cursor를 다시 시작하면 어느 프로젝트에서든 스킬이 동작합니다. 이때 스토리보드와 이미지는 그 프로젝트 안(`video-projects/`, `assets/`)에 만들어집니다. 이 저장소 폴더는 지우지 말고 그대로 두세요. 에이전트가 `video_render.py` 위치를 물으면 이 폴더 경로를 알려 주면 됩니다.

## 처음 실행할 때 환경 선택

설치 후 한 번 실행합니다. 설정 창이 열려 실행 환경을 고를 수 있습니다.

```bash
python setup.py          # Windows에서는 install.bat 더블클릭도 같습니다
```

| 항목 | 선택지 |
| --- | --- |
| 렌더링 방식 | 자동(GPU 우선, 안 되면 CPU) / GPU만 / CPU만 |
| 인코더 | 이 PC에서 실제로 동작하는 것만 목록에 나옵니다(예: `h264_qsv`, `h264_nvenc`) |
| 저장 위치 | 기본 `D:\VideoCreator`. 이미지·스토리보드·결과 영상·임시 파일이 모두 이 폴더 아래에 저장됩니다 |
| 기본 화질 | 720p / 1080p / 1440p / 4k |
| 프레임 보간 | GPU 크로스페이드(빠름) / 움직임 보간(느리지만 자연스러움) |
| 스킬 설치 | 모든 프로젝트에서 쓰도록 Cursor 스킬을 복사 |

- 선택은 `video-creator.config.json`에 저장되고(Git에는 올라가지 않음), 이후에는 창이 뜨지 않습니다. 바꾸려면 `python setup.py`를 다시 실행하세요. 현재 설정은 `python vc_config.py --show`로 봅니다.
- 영상 파일은 용량이 커서 C 드라이브가 아닌 D 드라이브 사용을 기본으로 합니다. D 드라이브가 없는 PC(또는 macOS/Linux)에서는 홈 폴더 아래 `VideoCreator`가 기본입니다. C 드라이브를 고르면 경고가 뜹니다.
- 설정 전에 에이전트가 렌더러를 실행해도 같은 창이 사용자 화면에 열리고, 선택할 때까지 기다립니다. 화면이 없는 환경(서버 등)에서는 기본값(D 드라이브)만 쓰고 저장하지는 않으며, 나중에 `python setup.py`로 정할 수 있습니다.
- `.cursor/rules/video-storage.mdc`가 에이전트에게 같은 규칙을 알려 줍니다. 생성 이미지는 Cursor 도구가 작업 폴더에 저장하므로, 에이전트가 만든 직후 저장 위치로 옮깁니다.

## 채팅만으로 영상 만들기

1. 에이전트 채팅에 요청합니다. 예: "제주 바다 일출 20초 영상 만들어 줘, 16:9"
2. 에이전트가 장면 이미지를 만들어 `<저장 위치>/assets/`에 옮기고 `<저장 위치>/projects/<이름>/storyboard.json`을 쓴 뒤, `video_render.py`로 렌더링합니다.
3. 완성된 파일은 `<저장 위치>/renders/<이름>.mp4`입니다. 에이전트가 경로, 길이, 사용한 인코더를 알려 줍니다.

직접 렌더링할 수도 있습니다.

```bash
python video_render.py D:/VideoCreator/projects/<이름>/storyboard.json
python video_render.py D:/VideoCreator/projects/<이름>/storyboard.json --quality 4k --codec hevc
```

GPU 영상 인코더(NVIDIA NVENC, Intel Quick Sync, AMD AMF, Apple VideoToolbox)를 순서대로 시험해 보고, 동작하는 것이 없으면 소프트웨어 인코더(libx264)를 씁니다. 장면 합성(카메라 움직임, 자막, 색보정)은 ffmpeg가 CPU로 처리합니다. 자막 글꼴은 Windows는 맑은 고딕, macOS는 Apple SD Gothic Neo를 자동으로 씁니다. 다른 글꼴은 `--font`로 지정합니다.

### 사람·장비가 움직이는 장면

Cursor에는 영상 생성 모델이 없으므로, 피사체가 움직여야 하는 장면은 고정 카메라 키프레임 12~16장을 순서대로 만들고, 그 사이 프레임을 `keyframes_to_video.py`가 30fps로 채웁니다. 에이전트는 바로 앞 키프레임 한 장만 참조 이미지로 넘겨 카메라·배경·인물·장비를 같게 유지하고, 움직이는 대상만 바꿉니다(끝 상태까지 함께 넘기면 끝 자세가 앞쪽 사진에 섞여 들어갑니다).

```bash
python keyframes_to_video.py D:/VideoCreator/projects/<이름>/storyboard.json            # 설정의 보간 방식
python keyframes_to_video.py D:/VideoCreator/projects/<이름>/storyboard.json --mode gpu  # GPU 크로스페이드, 약 20초
python keyframes_to_video.py D:/VideoCreator/projects/<이름>/storyboard.json --mode mci  # 움직임 보간, 약 5분
```

- `gpu`: OpenCL로 GPU에서 크로스페이드를 하고 GPU 인코더로 저장합니다. 빠르지만 움직이는 물체가 전환 중에 약간 겹쳐 보입니다.
- `mci`: ffmpeg 움직임 보간(`minterpolate`)을 키프레임 사이 구간별로 나눠 모든 CPU 코어에서 동시에 계산합니다. 더 자연스럽지만 이 보간 필터는 GPU 버전이 없어 느립니다. 인코딩은 GPU로 합니다.
- 결과는 연속 동작이 아니라 키프레임 사이를 보간한 영상이고, 배경 세부가 약간 일렁일 수 있습니다. 세밀한 위치(발 위치 등)는 이미지 생성기가 정확히 따르지 않을 수 있습니다. 진짜 연속 동작이 필요하면 별도의 영상 생성 서비스가 필요합니다.

### 추천 모델

Claude Sonnet 5.5, 생각 단계 "보통"을 권합니다. 화질은 Cursor 내장 이미지 생성이 정하고, 채팅 모델은 프롬프트 정확도와 재생성 판단을 좌우합니다. 같은 형식을 대량으로 만들 때는 Composer 2.5 fast(낮음)로도 충분합니다. 사용량은 이미지 개수가 가장 많이 차지하므로 장면·키프레임 수를 정해서 요청하고, 영상 작업은 새 채팅에서 하는 것이 좋습니다.

## 브라우저 렌더러 (미리보기·직접 편집용)

### 처음 한 번 설정

1. `studio.html`을 Chrome이나 Edge로 엽니다.
2. **작업 폴더 연결**을 누르고 Cursor에서 연 폴더를 고릅니다. 이 아래의 `storyboard.json`과 이미지·음악을 모두 찾습니다. 브라우저가 폴더를 기억하므로 다음부터는 열기만 하면 됩니다(권한을 다시 물으면 **다시 연결** 한 번).
3. 노트북이라면 Windows 설정 → 시스템 → 디스플레이 → 그래픽에서 Chrome/Edge를 **고성능**으로 지정하세요. 외장 GPU를 쓰게 됩니다. 사용 중인 GPU 이름이 스튜디오 왼쪽 위에 나옵니다.
4. **데모** 버튼 → **영상 내보내기**로 이미지 없이 전체 흐름이 되는지 확인할 수 있습니다.

### 쓰는 법

1. 에이전트가 만든 `video-projects/<이름>/storyboard.json`이 있어야 합니다.
2. 스튜디오에서 **↻**을 누릅니다. 가장 최근 프로젝트가 자동으로 열립니다.
3. **영상 내보내기**(`Ctrl+Enter`)를 누르면 연결한 폴더의 `renders/<이름>.mp4`로 저장됩니다.

폴더를 연결하지 않았다면 `storyboard.json`과 이미지를 스튜디오 왼쪽 상자에 끌어다 놓아도 됩니다. 이때는 저장 위치를 물어봅니다.

단축키: `Space` 재생/정지, `Ctrl+Enter` 내보내기, `Ctrl+S` 편집한 스토리보드를 파일에 저장.

## 브라우저 렌더러 작동 방식

| 단계 | 처리 | 쓰는 자원 |
|---|---|---|
| 장면 이미지 생성 | Cursor 에이전트의 기본 이미지 생성 | 사용자의 Cursor 세션 |
| 합성 | WebGL2: 이미지 텍스처(밉맵·비등방 필터), 카메라 움직임, 전환, 자막 | GPU |
| 후처리 | 셰이더 한 패스: 색보정, 비네팅, 필름 그레인, 레터박스, 페이드 | GPU |
| 인코딩 | WebCodecs 하드웨어 가속: H.264 / HEVC / AV1 | GPU 영상 인코더(NVENC, Quick Sync, AMF) |
| 오디오 | 오프라인 믹싱(반복·볼륨·페이드아웃) 후 AAC 인코딩 | CPU(짧음) |
| 저장 | MP4를 디스크에 바로 기록 | 메모리에 쌓지 않음 |

프레임은 실시간이 아니라 GPU가 처리하는 속도대로 만듭니다. 그래서 짧은 영상은 영상 길이보다 빨리 끝나고, 탭이 뒤에 있어도 멈추지 않습니다. 진행 표시줄에 인코더 종류(하드웨어/소프트웨어), 초당 프레임, 실시간 대비 배속이 나옵니다.

아래쪽 선택 상자로 해상도(720p–4K)와 코덱을 바꿀 수 있습니다. H.264가 가장 호환성이 좋습니다. HEVC와 AV1은 파일이 더 작지만, 그 코덱의 하드웨어 인코더가 있는 GPU에서만 빠릅니다(AV1은 RTX 40 이상, Intel Arc, RX 7000 이상). 지원하지 않으면 H.264로 자동 전환됩니다.

## 스토리보드 형식

이미지와 음악은 파일 이름으로 찾습니다. `image`에 경로를 써도 마지막 파일 이름만 비교하므로, 프로젝트마다 `<이름>-01.png`처럼 겹치지 않는 이름을 쓰세요. 예시는 `video-projects/_template/storyboard.json`에 있습니다.

| 필드 | 기본값 | 설명 |
|---|---|---|
| `title` | `video` | 저장 파일 이름 |
| `aspect` | `16:9` | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` |
| `quality` | `1080p` | `720p`, `1080p`, `1440p`, `4k` (`width`/`height`를 직접 써도 됨) |
| `fps` | `30` | 12–60 |
| `bitrate` | 해상도·fps에 맞춰 자동 | Mbps |
| `transition` | `0.8` | 장면 전환 시간(초) |
| `transitionType` | `crossfade` | `crossfade`, `fade-black`, `cut` |
| `fadeIn` / `fadeOut` | `0.6` / `0.8` | 영상 처음과 끝의 암전(초) |
| `effects` | 비네팅 0.35, 그레인 0.04 | `vignette`, `grain`(숫자 또는 true/false), `letterbox`(true면 2.39:1 시네마 바) |
| `look` | 대비 1.05 | `contrast`, `saturation`, `warmth`(음수면 차갑게) |
| `music` | 없음 | 음악 파일 이름. 영상보다 짧으면 반복 |
| `musicVolume` | `0.8` | 0–1 |

장면(`scenes[]`):

| 필드 | 기본값 | 설명 |
|---|---|---|
| `image` | 없음 | 이미지 파일 이름 또는 경로 |
| `background` | `#000` | 이미지가 없을 때 색. 배열이면 그라데이션 |
| `duration` | `4` | 초 |
| `motion` | 장면마다 순환 | `zoom-in`, `zoom-out`, `pan-left`, `pan-right`, `pan-up`, `pan-down`, `static` |
| `intensity` | `0.15` | 움직임 크기, 0–0.6 |
| `focus` | 가운데 | 줌이 향하는 지점 `[x, y]`, 0–1 |
| `transition` / `transitionType` | 전역값 | 이 장면으로 들어올 때의 전환 |
| `title` | 없음 | 화면 가운데 큰 글씨 |
| `caption` | 없음 | 아래쪽 자막 |

## 요구 사항과 한계

- `video_render.py`: Python 3.8 이상과 ffmpeg(ffprobe 포함)가 필요합니다. 인터넷 연결은 필요 없습니다.
- 에이전트가 직접 렌더링하려면 Cursor 에이전트의 터미널이 동작해야 합니다. 터미널을 쓸 수 없으면 에이전트가 브라우저 렌더러 사용법을 안내합니다.
- `studio.html`: Chrome 또는 Edge 최신 버전(WebGL2, WebCodecs, 파일 시스템 접근). Firefox와 Safari는 폴더 연결을 지원하지 않습니다.
- `studio.html`은 MP4 저장 모듈(`mp4-muxer`)을 처음 내보낼 때 인터넷에서 불러옵니다. 오프라인이면 실시간 녹화 모드(무음, 영상 길이만큼 걸림)로 자동 전환됩니다.
- 움직임은 이미지 위의 카메라 이동과 전환입니다. 사람이 걷거나 물이 흐르는 식의 실제 동작은 생성하지 않습니다.
- 에이전트는 오디오 파일을 만들지 못합니다. 음악은 직접 넣은 파일을 씁니다.

## 생성물과 이용자 책임

- 이 도구는 이미지나 음악을 직접 제공하지 않습니다. 장면 이미지는 사용자의 Cursor 계정으로 생성되며, 그 사용 조건은 Cursor와 이미지 모델 제공사의 이용약관을 따릅니다.
- 실존 인물의 얼굴·이름, 타인의 캐릭터·로고·브랜드, 저작권이 있는 음악·이미지를 허락 없이 쓰지 마세요. 그런 결과물에 대한 법적 책임은 그것을 만들고 배포한 사람에게 있습니다.
- AI로 만든 영상을 공개할 때는 플랫폼 규정이나 관련 법에 따라 AI 생성물임을 표시해야 할 수 있습니다.
- 사람을 속이거나 해치는 용도(딥페이크, 사칭, 허위 정보, 불법 콘텐츠)로 쓰지 마세요.

## 개인정보

`video_render.py`는 내 컴퓨터에서 ffmpeg만 실행하며, 네트워크에 접속하지 않습니다.

`studio.html`은 브라우저 안에서만 동작합니다. 이미지·음악·영상을 어떤 서버로도 보내지 않으며, 분석·추적 코드가 없습니다. 외부와 통신하는 것은 처음 내보낼 때 MP4 저장 모듈을 CDN(jsDelivr)에서 내려받는 한 번뿐입니다. 연결한 폴더 정보는 이 브라우저의 로컬 저장소(IndexedDB)에만 남습니다.

## 라이선스

MIT. `LICENSE` 파일을 보세요. 외부 라이브러리의 라이선스는 `THIRD_PARTY_NOTICES.md`에 있습니다.

이 소프트웨어는 "있는 그대로" 제공되며, 어떤 보증도 하지 않습니다. 사용에 따른 결과는 사용자 책임입니다.
