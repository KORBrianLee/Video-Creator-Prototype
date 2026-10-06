# 저사양 기준 GPU 개선 0.7.0

설계 기준은 LG 그램, 약 16GB RAM, Intel Iris 내장 GPU, 평소 RAM 여유가 적은 상태다. 정확한 Iris 세대와 드라이버는 아직 확인되지 않았다. 현재 RTX/i9 호스트는 기능·수학·영상 호환성을 시험한 장비다. 그램에서 실행 시간이나 품질을 측정한 것은 아니다. C SSD 허용은 저장 경로에만 반영했고 저사양 기본 해상도·RAM·GPU 버퍼 상한은 외장이 있더라도 동일하다.

## 구현한 경로

기본 Neo preview는 384×256, 49 실제 생성 프레임, 24fps, 약 2.04초다. 외장 GPU 우선, 없으면 감지한 내장 GPU를 선택한다. Windows OpenCL 드라이버를 통해 영상 transformer Linear 연산을 계산한다. attention·텍스트·VAE·안전 검사는 CPU에서 별도 프로세스로 순서대로 처리한다. GPU만으로 모든 연산을 수행한다고 표시하지 않는다.

원본 BF16 가중치를 SSD에 매핑하고 현재 행렬만 GPU로 보낸다. 커널에서 BF16 비트를 FP32로 복원하므로 Intel GPU의 BF16 전용 명령 지원이 필수는 아니다. 가중치 전체를 FP32 또는 GPU 모델로 복제하지 않는다. 행렬 행을 나눠 GPU 버퍼를 최대 64MiB로 제한한다. 드라이버가 별도로 할당하는 메모리까지 64MiB라고 보장하지 않는다. 작업 상주 RAM 상한 5GiB·CPU 4스레드·시스템 여유 1.5GiB 보호를 유지한다. RAM 압박 시 Windows의 [EmptyWorkingSet](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset)으로 이 추론 프로세스의 페이지를 회수 가능하게 한다. 다른 앱이나 페이지 파일 설정은 변경하지 않는다. [OpenCL API](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html).

GPU 스트리밍에는 기존의 무거운 INT8 변환이 필요 없다. GPU가 감지된 새 설치는 그 변환을 생략한다. CPU INT8 폴백은 기존 준비 파일을 옮기거나 별도 준비가 필요하다. GPU 강제 지정이 실패하면 오류를 반환하며 CPU 연산을 GPU 성공으로 표시하지 않는다. 자동 CPU 폴백은 GPU 사용이 없다고 명시한다.

## 실제 검사

자동 검사 **115개**와 실제 드라이버 행렬 검사를 통과했다. 행렬 검사는 bias 유무, 16의 배수가 아닌 크기, 작은 버퍼로 나눈 행을 CPU FP32 기준과 비교한다. `audits/gpu-math.json`에 각 오차와 실제 GPU 커널 횟수가 있다. Cursor stdio 연결도 6개 도구·UTF-8·완료 클립 재사용·중복 요청·대기 취소·현장 제작 거절을 실제 설치 환경에서 확인했다.

최종 일반 영상 작업 `14e3474c6f1e`은 다음 결과를 기록했다.

| 측정 | 실제 기록 |
|---|---:|
| 전체 생성 시간 | 130.078초 |
| 최대 단계 상주 RAM | 3.988GiB |
| 실제 GPU kernel 실행 | 4,590회 |
| 최대 명시 GPU 버퍼 | 46.148MiB |
| 버퍼 상한 | 64MiB |
| 실제 출력 | 384×256, 49프레임, 24fps |
| 실제 시험 장치 | RTX 4070 Laptop GPU / driver 610.74 |
| 안전 검사·전체 프레임 해독 | 통과 |
| 그램·Iris 실측 | 없음 |

조건 임베딩과 시작 화면은 캐시/참고 화면을 썼고 시간축 영상 추론은 새로 실행했다. 클립 재사용을 새 추론 시간으로 표시하지 않았다. 이전 CPU 영상과 해상도·정밀도·캐시 조건이 다르므로 속도 개선률을 주장하지 않는다. 저해상도 해변과 흰 거품 움직임의 기본 검토만 통과했다. 모든 대본·사람·장비 사실감을 인증한 것은 아니다. `docs/quality/neodragon-opencl.json`은 계산 코드·모델·시험 GPU/드라이버·실제 영상 SHA256을 묶는다. 다른 Iris 장치에는 검증을 전용하지 않는다.

현장 작업 `bab8b40bb560`도 GPU 4,590회 연산과 프레임 해독 단계까지 실행했으나 필수 안전 검사에서 거부되어 정상 영상으로 내보내지 않았다. 그 이전 `cc871b711a9e`와 Wan 시험 2건은 RAM 보호로 중지했다. 이 기록을 보존했고 보호 기준이나 안전 검사기를 꺼서 성공시킨 것은 아니다. 현장 협업·형태 유지·Sora2 사실감은 여전히 실패/미검증이며 `diagnostic=true` 개발 초안만 허용한다.

## 품질을 위한 현실적 대안과 이용자 보고

- **원본 BF16 + 허가된 시작 사진 + 짧은 단일 동작**: 현재 구현한 방법. INT8 활성값 양자화 손실을 피하고 장비·작업자의 시작 배치를 고정한다. 여러 사람과 복잡한 물리 동작을 학습한 새 모델이 되는 것은 아니다. 8개 현장 프롬프트 계산을 로컬에서 준비하며 가중치 재학습과 구분한다.
- **LTX-Video 2B distilled + OpenVINO INT4/I2V**: 다음 비교 가치가 있는 소형 후보다. [공식 OpenVINO 영상 경로](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/video-generation/)는 LTX의 텍스트·이미지 조건을 지원하고 [지원 하드웨어](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html)는 Iris Xe를 포함한다. 다만 이것이 해당 그램의 RAM·속도·결과를 검증하지는 않는다. 사전 변환된 공식 2B 묶음을 실제 저장소 검색에서 확인하지 못했다. 현재 설치·통합된 기능으로 표시하지 않는다. 정확한 가중치·라이선스·T5와 VAE까지 포함한 피크 RAM 검사가 필요하다.
- **사용자 평가도 반영**: [OpenVINO 사용자 보고 #2830](https://github.com/openvinotoolkit/openvino_notebooks/issues/2830)은 Core Ultra7 155H에서 GPU 강제 시 노이즈, CPU에서는 정상 결과를 보고했다. 이슈는 닫혀 있으며 2025년 당시의 자가 보고라 현재 모든 Iris의 실패 근거로 일반화하지 않는다. [#4223](https://github.com/openvinotoolkit/openvino.genai/issues/4223)에도 LTX 영상 품질·긴 처리 시간 보고가 있다. 공식 지원 표시나 생성 종료 코드만으로 모델을 최적으로 확정하지 않는 이유다. 이전 후기와 후보 비교는 MODEL_SELECTION.md에 보존했다.
- **Wan 1.3B Q4**: Apache-2.0 공개 모델의 별도 Vulkan 진단 경로를 개선했다. 기존 4단계 시험 대신 20/30/50단계 설정과 flow-shift 3을 사용하며 내장 GPU에서는 SSD 가중치 스트리밍을 적용한다. 텍스트 인코더 약 3.26GiB와 해독 RAM 부담이 커서 그램 기본으로 채택하지 않았다. 실제 현재 호스트 시험에서도 RAM 보호가 작동했다. 참고 이미지 모델이 아니므로 image_path를 거부한다. [공식 Wan](https://github.com/Wan-Video/Wan2.1), [엔진 설정](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/wan.md).

LTX-2의 19B/22B급과 Wan 14B를 RTX에서 실행할 수 있다는 이유로 기본에 넣지 않는다. 큰 모델명보다 전체 인코더·VAE·중간값과 성공 영상의 비용을 판단한다. CPU용 작은 DistilT5를 가진 Neo는 저메모리 실행을 구현한 현재 기본이며, 원하는 현장 품질을 충족한 최적 모델이라고 확정하지 않았다.

## 저장과 전달

기본 실행 폴더는 `D:\VideoCreator\CursorVideoRuntime`이다. C는 운영체제 전용으로 두고, D 볼륨이 없는 PC에서만 `C:\CursorVideoRuntime`을 쓴다. 다른 D 폴더는 `setup.cmd -RuntimeDir <D 경로>`로 지정한다. 경로 탈출·다른 볼륨 junction·드라이브 루트·네트워크 경로를 거부한다. 자료·모델·캐시·임시 파일·결과는 선택한 폴더 안에 저장한다. Windows가 관리하는 GPU 드라이버 캐시와 시스템 페이지 파일까지 앱 폴더에 강제할 수 있는 것은 아니다.

C 실행 폴더의 `.cursor/mcp.json`은 준비된 Python과 이 앱을 직접 연결한다. 새 리빌드 ZIP은 소스와 검토 샘플 묶음이고 모델/실행 라이브러리는 별도다. 라이선스·출처는 MODEL_LICENSES.md, 기존 현장 사진/영상의 CC BY 고지는 결과 ATTRIBUTION.md에 유지한다.

그램용 새 설치는 `setup-gram.cmd`를 실행하면 D SSD와 GPU 자동 선택을 함께 적용한다. 기존 CPU 설정도 이 설치 명령의 명시적인 auto 선택으로 갱신하되 메모리·스레드 사용자 설정은 유지한다.
