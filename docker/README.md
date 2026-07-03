# Docker Containers for 3D Reconstruction

Korean Painting 3D 파이프라인의 3D 재구성 모듈을 Docker 컨테이너로 실행합니다.

## 지원 모델

| 모델 | 설명 | VRAM | 속도 |
|------|------|------|------|
| **Wonder3D** | Multi-view diffusion 기반 고품질 재구성 | ~12GB | ~30초/객체 |
| **TripoSR** | Stability AI의 빠른 재구성 모델 | ~8GB | ~10초/객체 |

## 빠른 시작

### 1. Docker 이미지 빌드

```bash
cd korean-painting-3d/docker

# Wonder3D 빌드 (권장)
docker-compose build wonder3d

# 또는 TripoSR 빌드 (더 가벼움)
docker-compose --profile triposr build triposr
```

### 2. 단일 이미지 재구성

```bash
# Wonder3D
docker run --rm --gpus all \
    -v $(pwd)/outputs:/output \
    -v $(pwd)/input:/input \
    kp3d-wonder3d:latest \
    --input /input/object.png \
    --output /output/object.obj

# TripoSR
docker run --rm --gpus all \
    -v $(pwd)/outputs:/output \
    -v $(pwd)/input:/input \
    kp3d-triposr:latest \
    --input /input/object.png \
    --output /output/object.obj
```

### 3. 배치 처리

```bash
docker run --rm --gpus all \
    -v $(pwd)/outputs:/output \
    -v $(pwd)/extracted_objects:/input \
    kp3d-wonder3d:latest \
    --input-dir /input \
    --output-dir /output
```

## Python에서 사용

```python
from kp3d.modules.reconstruction.docker_wrapper import DockerReconstructionWrapper

# 초기화
wrapper = DockerReconstructionWrapper(model="wonder3d")

# 단일 이미지 재구성
result = wrapper.reconstruct(
    input_path="extracted/object_1_rgba.png",
    output_path="meshes/object_1.obj"
)

if result["success"]:
    print(f"Mesh saved: {result['output_path']}")

# 배치 처리
result = wrapper.reconstruct_batch(
    input_dir="extracted/",
    output_dir="meshes/"
)
```

## 통합 파이프라인에서 사용

```python
from kp3d.pipelines import IntegratedPipeline, PipelineConfig

config = PipelineConfig(
    use_reconstruction=True,
    reconstruction_model="wonder3d",  # Docker 자동 사용
    reconstruction_backend="docker",  # Docker 백엔드 지정
)

pipeline = IntegratedPipeline(config)
result = pipeline.process(
    image_path="image.png",
    annotation_path="annotation.json"
)
```

## GPU 설정

### 특정 GPU 사용
```bash
docker run --rm --gpus "device=0" ...
```

### NVIDIA Container Toolkit 설치 (필요시)
```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 모델 가중치 캐싱

첫 실행 시 모델 가중치가 다운로드됩니다. 캐시를 유지하려면:

```bash
# Named volume 사용 (docker-compose 기본 설정)
docker-compose up wonder3d

# 또는 호스트 디렉토리 마운트
docker run --rm --gpus all \
    -v $(pwd)/cache:/cache \
    -v $(pwd)/outputs:/output \
    kp3d-wonder3d:latest ...
```

## 문제 해결

### CUDA Out of Memory
- TripoSR 사용 (더 적은 VRAM)
- 이미지 크기 축소
- `--chunk-size` 옵션 조정

### Docker 빌드 실패
```bash
# 캐시 없이 재빌드
docker-compose build --no-cache wonder3d
```

### GPU 접근 불가
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Docker GPU 지원 확인
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

## 디렉토리 구조

```
docker/
├── docker-compose.yml          # Docker Compose 설정
├── README.md                   # 이 문서
└── triposr/
    ├── Dockerfile              # Wonder3D 이미지
    ├── Dockerfile.triposr      # TripoSR 이미지
    ├── inference_wonder3d.py   # Wonder3D 추론 스크립트
    └── inference_triposr.py    # TripoSR 추론 스크립트
```
