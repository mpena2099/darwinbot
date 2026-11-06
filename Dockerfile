# CPU image for training, replay to video, and export with STL parts.
#   docker build -t darwinbot .
#   docker run --rm -v "$PWD/checkpoints:/app/checkpoints" darwinbot make train
FROM python:3.10-slim

# OpenGL for MuJoCo's offscreen renderer (EGL) and for build123d, ffmpeg for video
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglib2.0-0 ffmpeg make \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=egl

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[video,cad]" pytest

CMD ["make", "help"]
