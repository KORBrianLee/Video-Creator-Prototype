# AnimateDiff-Lightning 기반 모델 변경 고지

작성일: 2026-10-01. 이 파일은 프로젝트의 변경 고지이며 원저자의 라이선스 원문이 아니다. 원본과 변경본 모두 동봉한 CreativeML Open RAIL-M 조건을 따른다.

원본: `gpustack/stable-diffusion-v1-5-GGUF`, revision `dcac270609fffc0ce7c7d41a3c0e752721859f7c`, `stable-diffusion-v1-5-Q4_0.gguf`. SHA-256 `c2f6e92f9d08d69cc673a1003528ac8199274b3c0eaec88d5fbefe5af67bd42b`, 크기 1,747,190,784 바이트. 이 원본은 제삼자 SD 1.5 양자화다.

로컬 변경본: `stable-diffusion-v1-5-Q4_0-lightning-linear.gguf`. SHA-256 `b4a990b50d0700a9b80687adde0ba3cf545137bbe230b2605bbf0bb366d0c61f`, 크기 1,747,194,848 바이트. 준비 버전: `lightning-linear-alphas-gguf-v1`.

변경: AnimateDiff-Lightning의 공식 EulerDiscreteScheduler 설정(beta_start=0.00085, beta_end=0.012, beta_schedule=linear)을 적용하도록 `alphas_cumprod` 텐서(F32, 길이 1,000)를 추가했다. 텐서 SHA-256 `b78dc5b23009b9e29b4fc81b614838efd3e06651df4203fcda4fbe1c1f471ce3`.

보존: 원본 파일은 유지한다. 원본의 텐서 1,130개의 데이터는 바이트 단위로 동일하며 학습 가중치를 바꾸지 않았다. 원본과 변경본의 공통 텐서 payload SHA-256은 `c34518cc07686f8e371574c66d81b46e487bb473af9cba1462c749aea5474920`이다. 새로운 학습이나 파인튜닝을 수행한 모델이 아니다.

함께 사용하는 공식 동작 가중치: ByteDance/AnimateDiff-Lightning revision `027c893eec01df7330f5d4b733bc9485ee02e8b2`, `animatediff_lightning_4step_comfyui.safetensors`, SHA-256 `aeb66ae8ff4a868d31379c3bde3e5e7e510a4d4b565a06ee4f1297e93a561dc5`, 크기 908,929,664 바이트. 이 파일은 원본 그대로 사용한다. ComfyUI 프로그램을 실행하지 않는다.

원문과 근거:

- [ByteDance 라이선스](https://huggingface.co/ByteDance/AnimateDiff-Lightning/blob/027c893eec01df7330f5d4b733bc9485ee02e8b2/LICENSE.md)
- [SD 모델의 CreativeML Open RAIL-M](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)
- [공식 Lightning 추론 설정](https://huggingface.co/ByteDance/AnimateDiff-Lightning/blob/027c893eec01df7330f5d4b733bc9485ee02e8b2/README.md)
- [고정 엔진의 alphas_cumprod 로드 경로](https://github.com/leejet/stable-diffusion.cpp/blob/3f8527a/src/pipeline/diffusion_engine.cpp)

원본 라이선스·출처 고지와 이 변경 고지를 함께 유지한다. 이 고지는 성능·비침해·생성 결과물의 모든 권리를 보증하지 않는다.
