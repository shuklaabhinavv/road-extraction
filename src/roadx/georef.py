"""Geo-reference a predicted road mask: convert road pixels to real-world
coordinates, export GeoJSON, render an interactive map over OpenStreetMap, and
quantitatively validate against OSM's road network.

    python -m roadx.georef --checkpoint runs/unetpp/best.pt \
        --image data/raw/test/sat/10378780_15.tiff --out results/georef

The Massachusetts Roads tiles are georeferenced GeoTIFFs (EPSG:26986,
1 m/pixel), so pixel->coordinate conversion uses the embedded affine transform.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import requests
import torch
from pyproj import Transformer
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

from roadx.models import build_model, pick_device
from roadx.predict import predict_image

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_BUFFER_M = 6.0  # tolerance buffer around OSM centerlines, meters


def mask_to_geojson(mask: np.ndarray, transform, crs) -> dict:
    """Vectorize road pixels into polygons in WGS84 lon/lat."""
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    feats = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        poly = shp_transform(to_wgs84, shape(geom))
        feats.append({"type": "Feature", "properties": {}, "geometry": mapping(poly)})
    return {"type": "FeatureCollection", "features": feats}


def fetch_osm_roads(bounds_wgs84) -> list:
    """Fetch OSM highway ways within (west, south, east, north) via Overpass."""
    w, s, e, n = bounds_wgs84
    query = f"""
    [out:json][timeout:60];
    way["highway"]({s},{w},{n},{e});
    out geom;
    """
    r = requests.post(
        OVERPASS_URL, data={"data": query}, timeout=90,
        headers={"User-Agent": "roadx-research/0.1 (road extraction OSM validation)"},
    )
    r.raise_for_status()
    ways = []
    for el in r.json().get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            ways.append([(pt["lon"], pt["lat"]) for pt in el["geometry"]])
    return ways


def rasterize_osm(ways, transform, crs, out_shape) -> np.ndarray:
    """Rasterize buffered OSM centerlines into the image grid."""
    from shapely.geometry import LineString

    to_native = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    shapes = []
    for coords in ways:
        if len(coords) < 2:
            continue
        line = shp_transform(to_native, LineString(coords))
        shapes.append(line.buffer(OSM_BUFFER_M))
    if not shapes:
        return np.zeros(out_shape, dtype=bool)
    ras = features.rasterize(
        ((mapping(g), 1) for g in shapes), out_shape=out_shape,
        transform=transform, fill=0, dtype="uint8",
    )
    return ras.astype(bool)


def osm_agreement(pred: np.ndarray, osm: np.ndarray) -> dict:
    eps = 1e-7
    correctness = (pred & osm).sum() / (pred.sum() + eps)   # predicted road near an OSM road
    completeness = (osm & pred).sum() / (osm.sum() + eps)   # OSM road area recovered
    return {
        "correctness_vs_osm": float(correctness),
        "completeness_vs_osm": float(completeness),
        "pred_road_px": int(pred.sum()),
        "osm_road_px": int(osm.sum()),
    }


def folium_map(geojson: dict, bounds_wgs84, dest: Path) -> None:
    import folium

    w, s, e, n = bounds_wgs84
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=14, tiles="OpenStreetMap")
    folium.GeoJson(
        geojson, name="predicted roads",
        style_function=lambda _: {"color": "#d62728", "weight": 1, "fillColor": "#d62728", "fillOpacity": 0.55},
    ).add_to(m)
    folium.Rectangle([(s, w), (n, e)], color="#1f77b4", weight=2, fill=False,
                     tooltip="image extent").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/georef"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--skip-osm", action="store_true", help="skip Overpass validation")
    args = p.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["model"], ckpt["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    with rasterio.open(args.image) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform, crs = src.transform, src.crs
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        w, s = to_wgs84.transform(src.bounds.left, src.bounds.bottom)
        e, n = to_wgs84.transform(src.bounds.right, src.bounds.top)
    bounds = (w, s, e, n)
    print(f"image CRS {crs}, WGS84 bounds: W{w:.5f} S{s:.5f} E{e:.5f} N{n:.5f}")

    prob = predict_image(model, img, device)
    mask = prob > args.threshold
    print(f"predicted road pixels: {mask.mean():.2%}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.image.stem}_{ckpt['model']}"

    gj = mask_to_geojson(mask, transform, crs)
    gj_path = args.out / f"{stem}.geojson"
    gj_path.write_text(json.dumps(gj))
    print(f"wrote {gj_path} ({len(gj['features'])} road polygons)")

    map_path = args.out / f"{stem}_map.html"
    folium_map(gj, bounds, map_path)
    print(f"wrote {map_path}")

    if not args.skip_osm:
        print("fetching OSM roads via Overpass...")
        ways = fetch_osm_roads(bounds)
        osm = rasterize_osm(ways, transform, crs, mask.shape)
        stats = osm_agreement(mask, osm)
        stats_path = args.out / f"{stem}_osm_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"OSM ways: {len(ways)} | correctness {stats['correctness_vs_osm']:.3f} "
              f"| completeness {stats['completeness_vs_osm']:.3f}")
        print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
