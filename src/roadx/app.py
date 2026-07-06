"""Web demo: upload an aerial image, pick a model, get extracted roads.

    python -m roadx.app          # serves http://localhost:8501

If the upload is a georeferenced GeoTIFF, the response includes WGS84 bounds
and GeoJSON so the frontend can draw the roads on an interactive map.
"""

import base64
import io
import json
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from roadx.models import build_model, pick_device
from roadx.predict import overlay, predict_image

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "comparison.csv"
import os

MAX_SIDE = 3000  # guardrail for huge uploads
# smaller anywhere-mode regions on CPU deployments (free hosting)
MAX_TILES_ENV = int(os.environ.get("ROADX_MAX_TILES", "0"))

# friendly demo filenames -> dataset stems (for ground-truth lookup)
STEM_ALIASES = {
    "roadx-test-newton": "22078975_15",
    "roadx-test-highway": "10378780_15",
    "roadx-test-suburb": "26878690_15",
}

app = FastAPI(title="roadx demo")
device = pick_device()
_models: dict[str, torch.nn.Module] = {}


def get_model(name: str) -> torch.nn.Module:
    if name not in _models:
        ckpt = torch.load(RUNS / name / "best.pt", map_location="cpu")
        m = build_model(ckpt["model"], ckpt["encoder"], encoder_weights=None)
        m.load_state_dict(ckpt["state_dict"])
        m.to(device).eval()
        _models[name] = m
    return _models[name]


def png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def try_georef(raw: bytes, mask: np.ndarray):
    """If the upload is a GeoTIFF, return (bounds_wgs84, geojson)."""
    try:
        import rasterio
        from roadx.georef import mask_to_geojson
        from pyproj import Transformer

        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
            tmp.write(raw)
            tmp.flush()
            with rasterio.open(tmp.name) as src:
                if src.crs is None or src.transform.is_identity:
                    return None, None
                t = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                w, s = t.transform(src.bounds.left, src.bounds.bottom)
                e, n = t.transform(src.bounds.right, src.bounds.top)
                gj = mask_to_geojson(mask, src.transform, src.crs)
                return [w, s, e, n], gj
    except Exception:
        return None, None


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


def find_ground_truth(filename: str) -> Path | None:
    stem = Path(filename).stem
    stem = STEM_ALIASES.get(stem, stem)
    for split in ("test", "valid", "train"):
        for ext in (".tif", ".tiff", ".png"):
            cand = DATA / split / "map" / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


def gt_metrics(mask: np.ndarray, gt_path: Path) -> dict | None:
    gt = np.asarray(Image.open(gt_path).convert("L")) > 127
    if gt.shape != mask.shape:
        return None
    tp = float((mask & gt).sum())
    fp = float((mask & ~gt).sum())
    fn = float((~mask & gt).sum())
    eps = 1e-7
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
    }


@app.get("/api/models")
def models():
    available = sorted(p.parent.name for p in RUNS.glob("*/best.pt"))
    metrics = {}
    if RESULTS.exists():
        import csv
        with open(RESULTS) as f:
            for row in csv.DictReader(f):
                metrics[row["model"]] = {
                    "test_iou": round(float(row["iou"]), 3),
                    "test_f1": round(float(row["f1"]), 3),
                }
    return {"models": available, "device": device.type, "metrics": metrics}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form("unetpp"),
                  threshold: float = Form(0.5)):
    raw = await file.read()
    try:
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        return JSONResponse({"error": "could not read image"}, status_code=400)
    if max(img.shape[:2]) > MAX_SIDE:
        return JSONResponse(
            {"error": f"image too large (max {MAX_SIDE}px per side)"}, status_code=400)

    prob = predict_image(get_model(model), img, device)
    mask = prob > threshold
    bounds, gj = try_georef(raw, mask)

    gt_path = find_ground_truth(file.filename or "")
    accuracy = gt_metrics(mask, gt_path) if gt_path else None

    return {
        "model": model,
        "road_fraction": float(mask.mean()),
        "overlay_png": png_b64(overlay(img, mask)),
        "mask_png": png_b64((mask * 255).astype(np.uint8)),
        "bounds_wgs84": bounds,
        "geojson": gj,
        "accuracy": accuracy,
    }


# --- "anywhere on Earth" mode: fetch Esri World Imagery tiles for a region ----

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TILE = 256
MERC = 20037508.342789244
MAX_TILES = MAX_TILES_ENV or 144  # 12x12 tiles = ~3072 px per side


class Region(BaseModel):
    west: float
    south: float
    east: float
    north: float
    model: str = "unetpp"
    threshold: float = 0.5


def lonlat_to_tilef(lon: float, lat: float, z: int) -> tuple[float, float]:
    lat = max(-85.05, min(85.05, lat))
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def pick_zoom(lat: float) -> int:
    """Zoom whose ground resolution is closest to the 1 m/px training GSD."""
    best, best_z = None, 17
    for z in range(15, 19):
        res = 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)
        if best is None or abs(res - 1.0) < best:
            best, best_z = abs(res - 1.0), z
    return best_z


def fetch_tile(z: int, x: int, y: int) -> Image.Image:
    r = requests.get(TILE_URL.format(z=z, x=x, y=y), timeout=20,
                     headers={"User-Agent": "roadx-demo/0.1"})
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


@app.post("/api/extract_region")
def extract_region(req: Region):
    lat_c = (req.south + req.north) / 2
    z = pick_zoom(lat_c)
    x0f, y1f = lonlat_to_tilef(req.west, req.south, z)
    x1f, y0f = lonlat_to_tilef(req.east, req.north, z)
    tx0, ty0, tx1, ty1 = int(x0f), int(y0f), int(x1f), int(y1f)
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    if nx * ny > MAX_TILES:
        return JSONResponse(
            {"error": f"region too large ({nx}x{ny} tiles) — zoom in a bit"},
            status_code=400)

    coords = [(z, tx0 + i, ty0 + j) for j in range(ny) for i in range(nx)]
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            tiles = list(ex.map(lambda c: fetch_tile(*c), coords))
    except Exception:
        return JSONResponse({"error": "could not fetch satellite imagery"},
                            status_code=502)

    canvas = Image.new("RGB", (nx * TILE, ny * TILE))
    for (zz, x, y), tile in zip(coords, tiles):
        canvas.paste(tile, ((x - tx0) * TILE, (y - ty0) * TILE))
    img = np.asarray(canvas)

    prob = predict_image(get_model(req.model), img, device)
    mask = prob > req.threshold

    # affine transform of the stitched image in web mercator (EPSG:3857)
    from rasterio.transform import Affine
    from roadx.georef import mask_to_geojson
    tile_m = 2 * MERC / (2 ** z)
    res = tile_m / TILE
    minx_m = tx0 * tile_m - MERC
    maxy_m = MERC - ty0 * tile_m
    transform = Affine(res, 0, minx_m, 0, -res, maxy_m)
    gj = mask_to_geojson(mask, transform, "EPSG:3857")

    return {
        "model": req.model,
        "zoom": z,
        "resolution_m": round(res, 2),
        "size_px": [img.shape[1], img.shape[0]],
        "road_fraction": float(mask.mean()),
        "overlay_png": png_b64(overlay(img, mask)),
        "geojson": gj,
    }


def main() -> None:
    uvicorn.run(app, host=os.environ.get("ROADX_HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8501")))


if __name__ == "__main__":
    main()
