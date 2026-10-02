# Neodragon CPU 시험 — 2026-10-02 후속 완료

이전의 시작 전 RAM 거절 이후 여유가 생겨 실제 단계를 실행했다. **현재 기본 경로는 전체 영상 생성·검사·인코딩까지 완료했으며, CPU에서 단순 해변 장면을 확인했다.** [최신 실행과 한계](VALIDATION.md)가 현재 상태다. 아래 이전 준비 기록은 당시 미실행 상태의 보관 이력이다.

원저자 코드와 원본 가중치를 보존하고 다음 래퍼를 추가했다: D 전용 CPU 환경, 단계마다 종료하는 하위 프로세스, 매핑 로드, FP32 CPU 계산, 채널별 INT8 행렬, 공식 VAE의 순차 인과 복원, 필수 출력 검사. 수정·출처 고지는 v2 파생 파일에 기록했다. 원저자의 NPU 결과를 재현했다고 표시하지 않는다.

| 실제 시험 | 결과 |
|---|---|
| `ad42fc5e8b` | 텍스트 완료, 첫 이미지 로딩에서 상주 제한 초과 |
| `f3f5fed2dc` | 매핑 로드로 상주 감소. 양자화 계층 속성 호환 문제 수정 |
| `2984ade002`, `32815f52ec` | CPU BF16 계산이 느려 이 시험만 중지하고 계산 형식 변경. SSD 첫 이미지의 해당 설정은 요청 주제 불일치 |
| `87f9dd9909` | 첫 이미지 입력에서 영상 추론 완료. 병렬 복원 상주 제한 초과 |
| `d6206a2bb1` | 순차 복원 완료. 출력 검사 버퍼의 버전 호환 문제를 고친 뒤 MP4 검사 완료. 계층 공통 INT8 영상의 흐림 확인 |
| `e0d922d1f9` | 채널별 INT8, 49프레임 전체 생성·검사·인코딩 완료. 디테일 유지 개선 |
| `c483d1a007` | 원본 BF16 매핑을 유지하는 CPU 대조 경로, 17프레임 생성 완료 |
| 작업 `85c15a63555a` | 텍스트→SD 첫 이미지→Neo 영상, 304.125초. 계층 공통 INT8의 과도한 흐림 때문에 최종 기본 구성으로 선택하지 않음 |
| 작업 `22f99f0c7776` | SD 첫 이미지 캐시→최종 채널별 INT8 영상, 128.640초, 최대 3.947GiB. 기준 영상으로 보관 |

실험 폴더는 `D:\CursorVideoLocal\audits\neodragon-cpu-probe-식별자`, 제작 작업은 `D:\CursorVideoLocal\jobs\작업ID`에 있다. 단계별 로그·시간·상주 최대값을 남겼으며 실패 기록의 state를 성공으로 덮어쓰지 않았다. 파일 해독·프레임 변화 검사는 실제 동작·사실감의 독립적인 보증이 아니다. 현재 시각 검토는 한 장면에 한정된다.

기본 다운로드는 이제 `neodragon.lock.json`의 29개 파일 8.16GiB와 필요한 CPU 환경이다. 연구 당시의 SSD1B 포함 41개 파일과 사용하지 않는 옛 진단 모델은 개발 D에 비교 자료로 보존했다. 사용자가 다른 SSD로 옮길 구성은 [PORTABLE.md](PORTABLE.md)다.

---

## 이전 준비·자원 거절 기록
# Neodragon CPU 시험 준비와 현재 상태

확인일: 2026-10-02. **새 영상 추론은 실행하지 못했다.** 시작 전 RAM 보호 기준으로 거절됐으며 이 상태를 영상 품질 실패, CPU 추론 성공, LG 그램 최적 모델 확정으로 표시하지 않는다.

## 준비한 구성과 실제 확인

- 공식 코드 `Qualcomm-AI-research/neodragon`, revision `d2abbe99f46577c4e1db682ad3c26832ec0c23b9`. Python 소스 20개·LICENSE의 Git blob을 검증하고 D에 보존했다. 공식 원본 소스 파일은 그대로 보존했다.
- 공식 가중치 `Qualcomm-AI-Research/Neodragon`, revision `5bdc87a4895a4fd148ca14c03181345d278d1040`. 320p hybrid에 필요한 파일 41개의 합계 **10,267,976,667바이트**다. 큰 가중치는 LFS SHA256, 작은 설정은 Git blob과 취득 SHA256을 확인했다. 고해상도·monolithic·중복 FP32 변형은 받지 않았다. [설치 파일별 증거](<D:/CursorVideoLocal/models/experimental-neodragon/model-provenance.json>)
- 별도 포터블 Python 3.12.10, PyTorch **2.8.0+cpu**, torchvision **0.23.0+cpu**, diffusers 0.34.0, transformers 4.46.3. 실제 import에서 `torch.version.cuda`는 null이고 CPU 4스레드 설정을 확인했다. 공식 Neodragon import와 작은 INT8 선형 연산의 유한한 출력을 확인했다. **전체 모델 실행 확인은 아니다.** [성공한 예비 명령의 출력 전사](<D:/CursorVideoLocal/audits/neodragon-cpu-preflight.json>), [정확한 의존 파일 URL·해시](<D:/CursorVideoLocal/audits/neodragon-cpu-dependencies.json>), [설치 로그](<D:/CursorVideoLocal/audits/neodragon-cpu-setup.log>)
- 공식 코드가 일반 생성에 요구하는 출력 검사 부품도 별도로 준비했다. 외부 pickle 객체를 임의 실행하지 않도록 실험 코드는 `weights_only=True`로 읽는다. [출처·해시](<D:/CursorVideoLocal/models/experimental-safety-checker/model-provenance.json>)

## 작성한 실험 경로

`tools/setup_neodragon_probe.py`는 준비·다운로드 전용이다. `tools/probe_neodragon_cpu.py`는 네트워크를 끄고 D에서 실행하도록 작성했다. 텍스트 → 첫 프레임 추론 → 첫 프레임 디코딩 → 영상 텍스트·문맥 → 첫 프레임 인코딩 → 최초 INT8 준비 → 영상 추론 → BF16 영상 디코딩 → 출력 검사 순으로 각각 새 프로세스를 쓰고 종료한다. 변환 가중치는 별도 `derived` 하위 폴더에 보존하며 원본을 덮어쓰지 않는다.

이 경로는 **전체 실행을 확인하지 못한 실험 코드**다. 단계별 분리가 메모리를 얼마나 줄이는지, INT8 변환·재사용·역직렬화가 전체 모델에서 작동하는지, 출력 장면과 동작이 정상인지는 아직 확인해야 한다. 설치한 라이브러리와 작은 연산의 성공으로 이를 대체하지 않는다.

요청은 512×320·49프레임·24fps, 해안에 파도가 밀려오는 장면이었다. 공식 hybrid 예제의 첫 프레임은 1024×640이지만 이 저자원 실험 코드는 512×320으로 작성했다. 이 차이와 INT8 변경을 기록하고 원저자의 결과·속도를 그대로 재현했다고 주장하지 않는다.

## 자원 거절 증거

1. 최초 4.5GiB 시작 기준에서 거절됐다. [첫 기록](<D:/CursorVideoLocal/audits/neodragon-cpu-probe-a197dcb5a3/benchmark.json>)
2. 중복 복사 제거·공식 BF16 디코딩·모델 매핑을 반영한 분리 실험은 시작 기준 3.0GiB에서도 거절됐다. [두 번째 기록](<D:/CursorVideoLocal/audits/neodragon-cpu-probe-b897508e18/benchmark.json>)
3. 후속 실제 여유 RAM은 **744,247,296바이트, 약 0.693GiB**였다. 다른 `sd-cli` 프로세스의 상주 메모리도 약 16.14GiB로 확인했다. 이 실험이 시작한 프로세스가 아니므로 종료하지 않았다. 현재 호스트의 일시적 상태이며 사용자 LG 그램의 측정이 아니다. [메모리 확인](<D:/CursorVideoLocal/audits/neodragon-start-resource-check.json>)

두 기록의 `stages`가 비어 있어 추론 하위 프로세스가 시작되지 않았음을 확인할 수 있다. **영상 파일 없음, 모델 최대 상주 메모리 미측정, 생성 시간 미측정.** 모델의 필요 RAM이 3.0GiB라는 뜻도 아니다.

실험 정책은 시작 전 여유 3.0GiB, 실행 중 시스템 여유 1.0GiB, 추론 프로세스 상주 5.0GiB, 최초 변환 상주 6.5GiB다. 이 값은 실측 최소 사양이 아니며 실제로 실행 중 보호가 작동하는지 새 모델에서 확인해야 한다. 기존 MCP의 4.5GiB 정책과 제작용 품질 차단은 별도로 유지한다.

## 경로와 이용 조건

런타임 `D:\CursorVideoLocal\runtime\neodragon-python`, 소스 `D:\CursorVideoLocal\engines\experimental-neodragon`, 모델 `D:\CursorVideoLocal\models\experimental-neodragon`, 로그·결과 `D:\CursorVideoLocal\audits`. 모델·캐시·임시 파일을 C나 사용자 홈으로 우회하지 않는다. PyTorch와 영상 계산은 외부 GUI·클라우드 서비스 없이 프로젝트 내부에서 호출한다.

Neodragon은 [공식 모델 고지](https://huggingface.co/Qualcomm-AI-Research/Neodragon#licenseterms-of-use)의 **BSD-3-Clause-Clear와 Qualcomm Responsible AI License**를 함께 따른다. 코드 파일에는 Apache-2.0 등 개별 고지도 있으므로 함께 보존한다. 공식 LICENSE·RAIL 원문 웹페이지는 D의 소스 폴더에 있다. 설치된 의존 패키지의 라이선스·고지는 각 dist-info 폴더에 유지했다. INT8 변경본을 만들 때도 원래 모델 조건과 변경·출처 고지가 필요하다. 이번 준비는 완전한 대외 배포물 권리 감사가 아니며 원래 프로젝트를 일괄 MIT 모델 번들로 변경하지 않는다.
