> 0.7.0 현재 기준: LG 그램 16GB·Intel 내장 GPU를 기준으로 한 저메모리 OpenCL 스트리밍 경로와 C/D SSD 선택을 추가했다. 기본 preview는 384×256이며 GPU 버퍼 64MiB·작업 RAM 5GiB 상한이다. 아래는 이전 버전의 조사·측정 이력이다. RTX의 시간은 그램 성능이 아니며 현장 사실감과 Iris 실행은 미검증이다. 현재 변경과 실측은 [LOW_RESOURCE_GPU.md](LOW_RESOURCE_GPU.md)를 우선한다.

# 현재 실행 검증 — 2026-10-02

**현재 제어·연결 리빌드는 0.6.1이다.** 자동 검사 102개와 실제 D 설치의 MCP 연결 검사를 통과했다. 반복 요청 검색 및 응답만 변경했고, 아래 0.6.0 영상 계산 코드와 일반 검증·현장 품질 실패 상태는 유지했다. [실제 응답·검색 비교](REBUILD_OPTIMIZATION.md). 현재 실제 연결 기록은 `mcp-ready-validation.json`과 `mcp-quality-gate-validation.json`, 이전 0.6.0 기록은 `audits/saved-0.6.0`이다.

**현장용 영상 변경은 0.6.0에서 적용했다.** 중장비·작업자 장면의 실제 형태·동작 품질은 검증 기준을 통과하지 못했다. 실제 참고 영상·프롬프트 조건 준비와 모델 가중치 학습을 구분한다. 2초/4초 단일 샷 생성과 메모리 보호는 실행했지만, SORA2급 자연스러운 현장 영상과 LG 그램 실행은 미검증/미달이다. [현장 시험과 한계](SITE_ADAPTATION.md), [현장 품질 기록](quality/construction-cpu.json).

일반 해변 장면은 변경 코드로 **다시 신경망을 실행한 작업 `02df8410a909`**에서 117.593초, 최대 상주 약 3.95GiB였다. 첫 이미지 캐시를 사용했고 조건 계산은 다시 실행했다. 최종 MP4 SHA256은 0.5.0 기준과 동일했다. 전체 49프레임 해독과 표본 시각 검토를 마쳤다. 일반 기준과 현장 기준은 별도이며, 일반 검증 성공은 현장 제작 허용으로 이어지지 않는다.

**0.6.0 연결·제어 검증:** 자동 검사 94개 통과. 실제 D 설치의 MCP 서버에서 도구 6개, 현장 설정 8개, 일반 검증 클립 재사용, 동일 요청 재사용, 대기 중 취소를 확인했다. 현장 일반 제작 요청은 작업 파일을 만들지 않고 거절하며, 명시된 현장 초안 요청은 검토된 클립 재사용·49프레임 전체 해독·출처 표시·`production_realism_verified=false`를 확인했다. 근거는 `D:\CursorVideoLocal\audits\mcp-ready-validation.json`과 `mcp-quality-gate-validation.json`이다. 이 연결 검사는 새 영상 추론이나 실제 Cursor 화면 확인을 대신하지 않는다.

**이전 0.5.0 관측:** 첫 이미지 캐시·조건 계산 새 실행 119.703초, 조건 계산 재사용·새 영상 추론 87.922초, 두 영상 동일 SHA256. 작업 `7342c862602e`, `audits/rebuild-benchmark.json`, 당시 자동 검증 81개. [비교 조건](REBUILD_OPTIMIZATION.md). 아래 0.4.0과 더 이전 실패 기록은 비교 근거로 보존했다.

## 이전 0.4.0 검증 기록

**기본 Neodragon의 CPU 생성 경로를 연결했고 한 장면의 검증을 통과했다.** 한국어 원문을 보존한 텍스트 요청에서 로컬 첫 이미지와 시간축 영상 추론, 순차 복원, 필수 출력 검사, MP4 인코딩을 실행했다. 검증 범위는 짧은 해변·수면 장면이다. Sora 수준의 사실감이나 그램 속도·복잡한 인물 동작의 검증으로 확대하지 않는다.

## 실제 결과와 측정

| 항목 | 확인한 결과 |
|---|---|
| 최종 생성 작업 | `D:\CursorVideoLocal\jobs\22f99f0c7776` |
| 설정 | 512×320, 49프레임, 24fps, 2.0417초 |
| 장치 | CPU 4스레드. CPU-only Torch 2.8.0+cpu. CUDA 사용 없음 |
| 시험 호스트 | i9-13900HX, RAM 31.78GiB. LG 그램 실측 아님 |
| 최종 생성 시간 | 128.640초. 앞선 실제 생성 작업의 동일 첫 이미지를 캐시에서 재사용 |
| 첫 이미지 비용 | `85c15a63555a` 작업의 SD 1.5 Q4 생성 175.109초. 전체 초회 약 304초는 별도 관측값의 합산 추정 |
| 최대 추론 상주 | 3.947GiB. 단계를 분리한 하위 프로세스의 250ms 간격 관측 최대값이며 전체 PC 사용량 아님 |
| 최초 채널별 INT8 준비 | 약 6.300GiB 상주, 39.110초. 준비된 v2 사본을 이후 재사용하며 매번 변환하지 않음 |
| 출력 검사 | 샘플 프레임 3개의 unsafe=false. 검사 기능을 끄지 않음 |
| 파일 검사 | 49프레임 전체 해독, 해상도·FPS 일치, 48개 인접 쌍에 변화, MP4 SHA256 기록 |
| 시각 검토 | sampled frames에서 해변을 식별하고 수면·해안선의 변화와 세부 유지 확인. 짧고 절제된 움직임이며 질감·사실감의 한계가 남음 |

현재 기준 영상은 `D:\CursorVideoLocal\app\examples\validated-beach.mp4`, 프레임 비교 이미지는 같은 이름의 `.jpg`다. [검증 기록](quality/neodragon-cpu.json)은 실행 코드와 모델 lock의 해시를 함께 묶는다. 기준 영상 손상·누락, 생성 코드 변경, 모델 lock 변경은 제작 준비 상태를 무효화한다. 다른 새 장면은 별도 시각 검토가 필요하다.

## 수정과 비교 근거

- 전체 모델을 복제하던 첫 이미지 로딩이 메모리 제한을 넘었다. 메타 모델에 매핑 가중치를 연결한 뒤 필요한 계층만 변환했다. 연구용 SSD 첫 이미지의 해당 저해상도·계층 공통 INT8 설정은 주제 불일치가 있어 기본 첫 이미지로 선택하지 않았다.
- 실측 호스트에서 느렸던 BF16 합성 계산 대신 CPU의 FP32 계산과 INT8 행렬을 조합했다. Iris 그램에서 이 연산의 속도를 측정하지 않았으며 원저자의 GPU/NPU 성능 수치를 CPU 수치로 옮기지 않았다.
- 계층당 공통 스케일의 INT8 첫 영상은 생성에는 성공했으나 시간이 흐를수록 세부가 흐려졌다. 채널별 스케일의 v2에서는 같은 입력의 세부 유지가 개선됐다. `neodragon-cpu-probe-e0d922d1f9`에 49프레임 비교 자료를 보존했다.
- 정확한 BF16 가중치를 매핑하고 현재 행렬 계산만 FP32로 펼치는 대조 경로도 17프레임에서 실행했다. `neodragon-cpu-probe-c483d1a007`. 전체 49프레임의 기본 제작 모드로 선택하거나 그램 속도 우위를 주장하지 않는다.
- 공식 VAE의 병렬 복원은 RAM 제한을 넘었다. 공식 구현의 순차 인과 복원을 사용해 약 2.1GiB로 낮췄다. 소규모 동일 디코더 흐름의 병렬/순차 결과 최대 차이는 `8.940696716308594e-08`이었다. `audits/neodragon-decoder-equivalence.json`. 이 수치 검사는 영상 사실감 평가를 대신하지 않는다.
- 오래된 출력 검사 가중치의 position_ids 버퍼는 최신 코드에서 비영속 버퍼다. 값이 같은지 확인하고 나머지 가중치를 엄격히 로드했다. 원저자 코드·모델 원본은 보존했다.

이 과정의 실패 로그와 성공 기록을 모두 D에 남겼다. 기존 실패를 성공 결과로 덮어쓰지 않았다. [CPU 시험 이력](NEODRAGON_CPU_PROBE.md).

## Cursor 연결과 보호 기능

실제 D 설치의 표준 입출력 서버로 MCP 0.4.0을 실행해 도구 5개, 정상 제작, 기존 클립의 완전 해독·재사용, 동일 요청의 작업 ID 재사용, 대기 중 취소를 확인했다. 옛 초기화 방식과 최신 요청별 메타데이터 방식을 검사했다. 최신 구조의 근거는 [공식 MCP 버전 규칙](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)과 [discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)다. Cursor의 실제 화면과 승인·재생은 이 통신 검사에 포함되지 않았다.

증거: `D:\CursorVideoLocal\audits\mcp-ready-validation.json`. 정상 제작 검증 작업은 `de28fffc7875`이며 이 통신 검증에서는 검증된 최종 클립을 재사용했다. 새 신경망 계산을 또 실행한 것으로 표시하지 않는다. stdout의 JSON-RPC 외 출력과 stderr는 없었고 UTF-8을 엄격히 확인했다. 기존 실패 모델의 정상 제작 요청이 작업 파일을 변경하지 않고 차단되는 것도 `mcp-quality-gate-validation.json`으로 확인했다.

새 모델까지 포함한 68개 자동 검증은 2.143초에 통과했다. 모델 전체 영상 품질 시험과 별도로, D 경로·작업 큐·취소·메모리 초과·캐시 손상·같은 크기의 모델 및 영상 변조·참고 이미지 변경·안전 검사·프로토콜 오류를 검사했다. 로그: `audits/unit-tests-neodragon-final.txt`. Cursor 스킬의 구조 검사도 통과했다.

최종 전체 설치 재실행도 종료 코드 0으로 통과했고, 기존 모델·CPU 실행 환경·INT8 사본을 재사용했다. `audits/installer-neodragon-replay.log`. 재설치 후 모델 29개와 변환 사본의 SHA256을 다시 확인한 준비 점검은 `ready=true`, `ready_for_generation=true`였다. 준비된 현재 호스트의 재실행이며 새 그램의 최초 설치 성공을 의미하지 않는다.

## 그램에서 남은 확인

Iris GPU 가속은 구현하지 않았다. CPU 실행만 제공하며 실제 그램의 CPU·여유 RAM·속도는 미측정이다. 여유 RAM이 부족하면 시작을 거절하거나 이 작업의 하위 프로세스만 중지한다. SSD를 RAM이나 GPU의 대체재로 표시하지 않는다. 한 번에 한 작업을 수행하며 외부 앱·페이지 파일을 조작하지 않는다.

이하 기록은 새 모델을 연결하기 전의 기존 진단 이력이다.

---
# 검증 기록

확인일: 2026-10-01~02. **작업 제어와 MCP 통신은 검증했지만, 실제 영상 생성 품질은 실패했다.** 현재 결과로 사실적인 영상 생성기의 완성, Sora 수준의 결과, LG그램에서의 실행 성공을 주장할 수 없다. 정상적으로 저장되고 프레임이 변화하는 파일에서도 요청한 해안·파도를 알아볼 수 없는 줄무늬와 격자가 나타났다.

실행 호스트는 작업자가 확인한 **Core i9-13900HX, RAM 32GB**다. 아래 영상 진단은 Windows의 CPU 경로·4스레드를 사용했다. 사용자의 **LG그램, RAM 16GB, Intel Iris 내장 GPU**와 다른 장치다. LG그램의 가용 RAM·처리 시간·영상 품질과 Iris 가속은 실측하지 않았다. CPU 진단 수치를 해당 장치의 예상 시간이나 최소 사양으로 환산하지 않는다.

## 제어·연결 검증

| 검사 | 확인한 결과 | 범위와 근거 |
|---|---|---|
| 단위 검증 | 모델 재선정·품질 차단 변경 후 전체 실행에서 56개 통과: 제어·MCP 32개, 영상 처리·작업자 19개, 기반 파일 준비 5개. | 자원 준비가 정상이어도 품질 실패면 제작 작업을 제출하지 않는 검사를 추가했다. [실제 전체 실행 로그](<D:/CursorVideoLocal/audits/unit-tests-model-reselection.txt>), [제어 검사](<../tests/test_control.py>), [작업자 검사](<../tests/test_backend.py>), [파일 준비 검사](<../tests/test_lightning_prepare.py>) |
| 경로·자원 보호 | D 이외 경로와 저장 영역 밖으로 나가는 연결 폴더 거절, 부족한 RAM에서 시작 거절, 취소·실패 후 잠금과 소유 추론 프로세스 정리 검증. | 단위 검증에는 제어용 모형 작업자와 테스트용 하위 프로세스도 사용했다. 이는 실제 신경망 영상 품질 검증과 별개다. |
| 캐시 무결성 | 반복 요청·장면별 캐시, 설정 변경에 따른 키 변경, SHA256 검사, 같은 크기의 손상 파일 재사용 거절, 전체 영상 디코딩 검증. | 손상된 파일을 완료 결과로 재사용하는 경로를 검사했다. 프롬프트에 맞는 장면인지 자동 판정하는 검사는 아니다. |
| 실제 MCP stdio | 등록된 D의 Python·명령·인수로 서버 실행. MCP 2025-06-18 초기화, 응답 없는 초기화 통지, 도구 목록 5개, 장치 점검과 실패 작업 상태 호출 통과. | 도구 5개를 모두 실행한 것은 아니다. 영상 생성·취소를 이 통신 검사에서 호출하지 않았다. [실제 연결 설정](<D:/CursorVideoLocal/app/.cursor/mcp.json>), [독립 검증 기록](<D:/CursorVideoLocal/audits/mcp-stdio-validation.json>) |
| 한국어·응답 크기 | UTF-8 엄격 디코딩, 한국어 설명·실패 사유, JSON-RPC만 출력, 오류 출력 0바이트, 정상 종료. 장치 점검 721바이트·실패 상태 689바이트. | 장면 배열과 프레임별 결과를 노출하지 않았다. 완료 작업의 실제 응답 축약은 이 통신 검사에서 별도로 시험하지 않았다. |
| 재선정 후 실제 MCP 0.3.0 | 초기화·무응답 통지·도구 5개·점검·일반 제작 요청 차단 성공. 자원 `ready=true`와 품질 `ready_for_generation=false`를 별도로 확인. | 점검 응답 1,012바이트, 품질 실패 응답 280바이트. UTF-8 엄격 디코딩, 오류 출력 0바이트, 종료 0. 호출 전후 작업 JSON 파일이 동일해 제작 작업을 제출하지 않았음을 확인했다. [실제 연결과 차단 기록](<D:/CursorVideoLocal/audits/mcp-quality-gate-validation.json>) |
| 포터블 설치 재실행 | D의 내장 Python으로 전체 설치 경로 재실행이 9.75초에 성공했다고 작업자가 보고했다. PowerShell 문법과 UTF-8 BOM도 검사했다. | 준비된 이 호스트의 재실행이며, 최초 다운로드 시간·새 LG그램에서의 설치 성공·영상 품질을 뜻하지 않는다. [설치 명령](<D:/CursorVideoLocal/app/setup.cmd>), [설치 스크립트](<D:/CursorVideoLocal/app/setup.ps1>) |
| Cursor 화면 | 미검증. | 실제 Cursor 화면의 MCP 허용·연결·도구 선택·영상 재생 성공을 stdio 검사만으로 주장하지 않는다. |

단위 검증의 임시 파일은 D에 지정해야 한다. 앞선 전체 검사에서 임시 환경 변수 지정이 빠져 C에 만들어진 준비 시험 5개가 D 전용 규칙에 의해 거절된 기록은 경로 보호 실패가 아니다. 재선정 후 56개 검사는 D 임시 경로를 지정했고 실제 전체 출력 로그를 보존했다.

## 실제 생성 진단

아래 시간과 메모리는 **실패 출력을 생성한 구성의 진단 수치**다. 정상 영상의 속도나 요구 RAM을 입증하지 않는다. 메모리는 저장된 측정 기록의 추론 프로세스 최대 상주 메모리이며, 전체 시스템 RAM 사용량·공유 GPU 메모리·LG그램 최소 요구량과 다르다. 시간은 `benchmark.json`의 추론 경과 시간이며, 단일 이미지와 Wan 행은 엔진 로그가 보고한 생성 시간이다.

현재 엔진은 공식 929 배포의 `3f8527a`다. 교정 조합은 원래 학습 텐서 바이트를 보존하고 선형 확산 계수를 추가한 SD 1.5 Q4 GGUF 사본과 ByteDance 공식 4단계 ComfyUI 형식 동작 가중치다. ComfyUI 프로그램을 설치하거나 실행하지 않았다. TAE는 별도 진단용 작은 디코더이며 일반 VAE 결과와 구분한다. [준비 기록](<D:/CursorVideoLocal/models/stable-diffusion-v1-5-Q4_0-lightning-linear.provenance.json>), 원본·교정 근거(비공개 내부 기록)

| 진단 구성 | 해상도·프레임·단계 | 시간 / 최대 상주 메모리 | 실제 확인과 근거 |
|---|---|---|---|
| SD 1.5 기반 원본, 동작 모듈 없음 | 단일 이미지, 20단계 | 엔진 75.29초 / 미기록 | 모래·바다·하늘이 식별되는 정지 이미지. **영상 성공 아님.** [이미지](<D:/CursorVideoLocal/audits/sd-base-probe/base.png>), [로그](<D:/CursorVideoLocal/audits/sd-base-probe/engine.log>) |
| 선형 계수 추가 기반 사본, 동작 모듈 없음 | 단일 이미지, 20단계, TAE | 엔진 109.17초 / 미기록 | 해안이 식별되는 정지 이미지. 파일 준비 후 단일 이미지 경로가 작동함을 확인했으며 연속 동작은 증명하지 않는다. [이미지](<D:/CursorVideoLocal/audits/sd-base-probe/prepared-base.png>), [로그](<D:/CursorVideoLocal/audits/sd-base-probe/prepared-engine.log>) |
| 교정 Lightning, 일반 VAE | 256×256, 8프레임, 4단계 | 129.875초 / 2.830GiB | 황색·검정 줄무늬로 시각 실패. 8fps 파일 길이는 **1초**다. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-cpu-native/benchmark.json>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-cpu-native/contact-sheet.jpg>) |
| 교정 Lightning, TAE | 256×256, 8프레임, 4단계 | 84.500초 / 2.719GiB | 줄무늬·격자로 시각 실패. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-cpu-native-tae/benchmark.json>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-cpu-native-tae/contact-sheet.jpg>) |
| 교정 Lightning, TAE, 분할 계산 해제 | 256×256, 8프레임, 4단계 | 91.297초 / 2.719GiB | 같은 계열의 줄무늬·격자로 시각 실패. 분할 계산 해제만으로 해결되지 않았다. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-cpu-native-monolithic/benchmark.json>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-cpu-native-monolithic/contact-sheet.jpg>) |
| 공식 AnimateDiff v3 일반 동작 가중치, TAE | 256×256, **4프레임**, Euler 20단계·CFG 8 | 346.250초 / 3.216GiB | 색 격자·줄무늬로 시각 실패. 8fps 파일 길이는 **0.5초**다. Lightning 모델에 단계 수만 늘린 시험과 구분한다. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-stock/benchmark.json>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-stock/contact-sheet.jpg>) |
| 교정 Lightning, TAE, F16 옵션 요청 | 256×256, **4프레임**, 4단계 | 67.234초 / 3.014GiB | 색 격자·줄무늬로 시각 실패. 옵션은 F16을 요청했지만 확산 모델 통계는 F16 983개·Q4 291개를 기록해 GGUF의 Q4가 유지됐다. Q4 계산 경로를 제거한 시험이 아니며 Q4 문제를 배제할 수 없다. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-f16/benchmark.json>), [실제 가중치 로그](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-f16/inference.log>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-f16/contact-sheet.jpg>) |
| 이전 공식 CPU 841 `6b3edaa`, 교정 Lightning·TAE | 256×256, 8프레임, 4단계 | 75.469초 / 2.709GiB | 검정·황색 격자로 시각 실패. 이전 버전에서도 재현돼 최신 버전에만 생긴 문제라고 확인되지 않았다. 정확한 원인은 미확정이다. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-older/benchmark.json>), [로그](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-older/inference.log>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-older/contact-sheet.jpg>) |
| 정상 형식 Lightning, 작업 `4093d90d589b` | 256×256, **16프레임**, 4단계 요청 | 메모리 보호로 중단 | 5GiB 작업 상한 초과 오류로 실패 상태. 최후 저장된 표본은 4.956GiB이며 중단 순간의 최대값 전체를 뜻하지 않는다. 완성된 2초 영상 없음. [요청](<D:/CursorVideoLocal/jobs/4093d90d589b/request.json>), [상태](<D:/CursorVideoLocal/jobs/4093d90d589b/status.json>), [진단 로그](<D:/CursorVideoLocal/tmp/scene-fa0f3811f0a669dc-f2f2c39c/inference.log>) |
| 교정 Lightning, TAE, CPU 가중치·Flash Attention 끄기·분할 계산 해제 | 256×256, **16프레임**, 4단계 | 157.265초 / 3.126GiB | 이 호스트·이 구성은 5GiB 아래에서 실행해 8fps·2초 파일을 저장했다. 그러나 황색·녹색 세로 줄무늬로 시각 실패했다. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-context16/benchmark.json>), [구성 로그](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-context16/inference.log>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-context16/contact-sheet.jpg>) |
| 교정 Lightning, TAE, Haswell CPU 실행 라이브러리로 변경 | 256×256, **4프레임**, 4단계 | 41.609초 / 2.516GiB | 색 격자로 시각 실패. 실행 라이브러리를 바꾼 진단이며 Intel Iris 가속이나 LG그램 실측이 아니다. [측정](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-haswell/benchmark.json>), [구성 로그](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-haswell/inference.log>), [대표 프레임](<D:/CursorVideoLocal/audits/lightning-linear-alpha-probe-haswell/contact-sheet.jpg>) |
| Wan 2.1 T2V 1.3B 진단 | 256×144, 9프레임, 4단계 | 엔진 165.76초 / 작업자 보고 약 6.86GiB (7.36GB) | 0·4·8프레임 직접 검토에서 황색 가로 줄무늬로 시각 실패. 일반 Wan의 4단계 진단이며 증류 FastWan의 측정이 아니다. 저장 완료를 정상 장면 생성으로 기록하지 않는다. [로그](<D:/CursorVideoLocal/audits/wan-cpu-smoke/engine.log>), [첫 프레임](<D:/CursorVideoLocal/audits/wan-cpu-smoke/frame_000.png>), [중간 프레임](<D:/CursorVideoLocal/audits/wan-cpu-smoke/frame_004.png>), [마지막 프레임](<D:/CursorVideoLocal/audits/wan-cpu-smoke/frame_008.png>) |

이전 Diffusers 형식의 16프레임 결과 `e29d45342c41`도 회색 줄무늬·녹색 격자·색면으로 시각 실패했다. 해당 형식은 현재 Native 동작 모듈의 올바른 활성화가 확인된 구성과 다르므로 당시 작은 메모리 수치를 현재 정상 형식의 요구량으로 사용하지 않는다. 선형 계수만 바꾼 이전 Diffusers 형식의 8프레임 진단도 정상 구성 성능표에서 제외했다. [이전 요청·모델 형식](<D:/CursorVideoLocal/jobs/e29d45342c41/request.json>), [시각 실패 결과](<D:/CursorVideoLocal/jobs/e29d45342c41/output/contact-sheet.jpg>)

8프레임 시험과 4프레임 시험은 길이·동작 가중치·샘플링·디코더·가중치 읽기 형식이 다르다. 표의 시간 차이를 같은 영상의 속도 개선률로 비교하지 않는다. 정상 형식의 16프레임 시험 중 앞선 작업은 자원 보호로 중단됐고, TAE·CPU 가중치·Flash Attention 끄기·분할 계산 해제 조합은 실행을 마쳤지만 시각 품질은 실패했다. 따라서 16프레임을 항상 실행할 수 없다는 결론도, 정상적인 2초 영상을 완성했다는 결론도 현재 근거에 맞지 않는다.

## 현재 결론과 남은 확인

파일 디코딩·프레임 수·밝기 변화 검사만으로 실제 피사체와 자연스러운 동작을 확인할 수 없다는 것이 이번 실행에서 드러났다. 실제 시간축 생성 구현을 연결한 것과 사용 가능한 결과를 얻은 것은 다르다. 원본의 이미지 전환·보간 방식을 기본 생성 경로에서 바꿨지만 현재 영상 품질 목표는 충족하지 못했다.

모델을 고정하지 말라는 후속 요청에 따라 [LTX 2B·Neodragon·NOVA와 추가 후보 비교](MODEL_SELECTION.md)를 작성했다. Neodragon의 별도 CPU 환경과 공식 가중치는 D에 준비했고 라이브러리 불러오기·작은 INT8 연산을 확인했다. 실제 영상 시험은 시작 전 RAM 보호 기준으로 거절돼 **새 후보의 정상 영상과 전체 실행 단계는 아직 검증하지 못했다.** [새 후보 준비·차단 기록](NEODRAGON_CPU_PROBE.md). MCP 0.3.0은 파일·자원 준비와 영상 품질을 별도 필드로 전달하고 현재 제작 요청을 차단한다. 명시적인 진단 요청은 기존 실패를 재현할 수 있지만 제작 성공으로 표시하지 않는다.

결과가 알아볼 수 있는 장면인지와 동작이 요청에 맞는지를 먼저 확인해야 한다. 그 후 LG그램의 실제 여유 RAM과 Intel 장치에서 검증해야 한다. 사실적인 2초 영상이 나오기 전에는 설치 완료·MCP 준비 완료·단위 검증 통과를 영상 제작 성공으로 표시하지 않는다. 원본 자료는 그대로 보존하고, 추론 파일·임시 파일·로그·진단 결과·생성 결과는 D 드라이브에 둔다. 후속 품질 시험은 새 기록으로 추가하며 기존 실패 측정을 덮어쓰지 않는다.
