###############################################################################
# Wonder3D++ inference container
#
# Builds on the validated smoke-test base (torch 2.7 + cu128 + pytorch3d +
# nvdiffrast on Blackwell sm_120). Adds the rest of Wonder3D++'s inference
# stack and copies in the source code from the Wonder3D_Plus branch.
#
# Checkpoints are downloaded on first run into a mounted HF cache volume,
# so the image itself stays around 5-6 GB.
###############################################################################
FROM wonder3d-smoketest:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface/transformers
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"

# Wonder3D++ inference deps. Pins relaxed vs upstream requirements.txt
# to avoid conflicts with our torch 2.7 / numpy 2.x / cu128 base.
# Skipped: bitsandbytes (training quant), gradio (demo UI), comet_ml/tensorboard
# (training logging), mosaicml-streaming (training data), pytorch-lightning
# (training only), torch_efficient_distloss (training only).
RUN pip install --no-cache-dir \
    "diffusers==0.30.0" \
    "transformers>=4.30.0" \
    accelerate \
    omegaconf \
    "trimesh>=3.9,<4.0" \
    "nerfacc==0.3.3" \
    rembg \
    pyhocon \
    icecream \
    einops \
    ftfy \
    piq \
    matplotlib \
    opencv-python \
    imageio \
    "imageio-ffmpeg" \
    scipy \
    pyransac3d \
    onnxruntime-gpu \
    pyrender \
    jaxtyping \
    pymeshlab \
    fire \
    safetensors \
    huggingface_hub \
    modelcards \
    decord || true

# Some inference paths rely on segment_anything (foreground mask). Optional.
RUN pip install --no-cache-dir segment-anything || true

# open3d brings heavy GUI deps; install separately so its failure is non-fatal.
RUN pip install --no-cache-dir open3d || echo "open3d install failed (non-fatal)"

# torch-scatter is needed by some pytorch3d/nerfacc paths. Build from source
# against the installed torch 2.7 (no prebuilt wheels for cu128 may exist).
RUN pip install --no-cache-dir --no-build-isolation torch-scatter || \
    echo "torch-scatter install failed (non-fatal, only used by some paths)"

# Copy Wonder3D++ source code (Wonder3D_Plus branch checkout) into image.
WORKDIR /app
COPY wonder3d_plus_src /app/wonder3d
# Wonder3D's MVMeshRecon expects to import as top-level modules; expose them.
ENV PYTHONPATH="/app/wonder3d:/app/wonder3d/MVMeshRecon:/app/wonder3d/MVMeshRecon/utils"

# Our inference wrapper script (matches TripoSR's CLI shape for drop-in use).
COPY inference_wonder3d.py /app/inference_wonder3d.py

# I/O and cache mountpoints.
RUN mkdir -p /input /output /cache/huggingface

WORKDIR /app/wonder3d
ENTRYPOINT ["python", "/app/inference_wonder3d.py"]
CMD ["--help"]
