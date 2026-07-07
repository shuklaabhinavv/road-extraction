FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir segmentation-models-pytorch albumentations numpy pillow \
      pyyaml fastapi uvicorn python-multipart rasterio shapely pyproj requests

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY web ./web
COPY runs ./runs
RUN pip install --no-cache-dir -e . --no-deps

ENV ROADX_HOST=0.0.0.0 \
    PORT=7860 \
    ROADX_MAX_TILES=36

EXPOSE 7860
CMD ["python", "-m", "roadx.app"]
