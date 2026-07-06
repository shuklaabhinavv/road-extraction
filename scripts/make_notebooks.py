"""Generate the Kaggle notebooks by embedding the repo's own source files, so the
code that trains on Kaggle is byte-identical to the code in src/roadx.

    python scripts/make_kaggle_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "roadx"
OUT = ROOT / "notebooks"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def writefile_cell(rel: str, path: Path) -> dict:
    return code(f"%%writefile {rel}\n{path.read_text()}")


def notebook(cells: list) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


# One plain-python setup cell: %%writefile needs a non-empty body and must be the
# first line of a cell, so package dirs and empty __init__ files are made with os.
SETUP = (
    "import os; os.makedirs('roadx/data', exist_ok=True); "
    "os.makedirs('configs', exist_ok=True); "
    "open('roadx/__init__.py','w').close(); "
    "open('roadx/data/__init__.py','w').close()\n"
)


def prepare_notebook() -> dict:
    cells = [
        md(
            "# Massachusetts Roads — prepare 512x512 tiles\n\n"
            "**Setup (once):**\n"
            "1. *Add Input* -> search `massachusetts-roads-dataset` (by balraj98) -> attach.\n"
            "2. No GPU needed. Internet not needed if the dataset is attached.\n"
            "3. *Run All*, then *Save Version*. The `tiles/` output of this notebook is the\n"
            "   input dataset for the training notebook (*Add Input -> Your Work*).\n\n"
            "Fallback: if the Kaggle dataset is unavailable, enable Internet and use the\n"
            "download cell near the end to fetch from the original UofT mirror instead."
        ),
        code(SETUP),
        writefile_cell("roadx/data/tile.py", SRC / "data" / "tile.py"),
        writefile_cell("roadx/data/download.py", SRC / "data" / "download.py"),
        code(
            "# Adapt whatever structure the attached dataset has into raw/{split}/{sat,map}.\n"
            "# Pairs sat images with label masks by filename stem; maps val -> valid.\n"
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "INPUT = Path('/kaggle/input')\n"
            "EXTS = {'.tif', '.tiff', '.png'}\n"
            "SPLIT_ALIAS = {'train': 'train', 'val': 'valid', 'valid': 'valid', 'test': 'test'}\n"
            "\n"
            "def classify(d: Path):\n"
            "    parts = [p.lower() for p in d.parts]\n"
            "    split = next((SPLIT_ALIAS[k] for p in parts for k in SPLIT_ALIAS\n"
            "                  if p == k or p.startswith(k + '_') or p.endswith('_' + k)), None)\n"
            "    is_label = any('label' in p or p == 'map' for p in parts)\n"
            "    return split, is_label\n"
            "\n"
            "dirs = {}\n"
            "for dirpath, _, filenames in os.walk(INPUT):\n"
            "    d = Path(dirpath)\n"
            "    if any(Path(f).suffix.lower() in EXTS for f in filenames):\n"
            "        split, is_label = classify(d.relative_to(INPUT))\n"
            "        if split:\n"
            "            dirs.setdefault((split, is_label), d)\n"
            "\n"
            "raw = Path('raw')\n"
            "pairs_total = 0\n"
            "for split in ('train', 'valid', 'test'):\n"
            "    sat_dir, map_dir = dirs.get((split, False)), dirs.get((split, True))\n"
            "    if not (sat_dir and map_dir):\n"
            "        print(f'!! missing {split}: sat={sat_dir} map={map_dir}')\n"
            "        continue\n"
            "    sats = {p.stem: p for p in sat_dir.iterdir() if p.suffix.lower() in EXTS}\n"
            "    maps = {p.stem: p for p in map_dir.iterdir() if p.suffix.lower() in EXTS}\n"
            "    stems = sorted(set(sats) & set(maps))\n"
            "    (raw / split / 'sat').mkdir(parents=True, exist_ok=True)\n"
            "    (raw / split / 'map').mkdir(parents=True, exist_ok=True)\n"
            "    for s in stems:\n"
            "        for src, sub in ((sats[s], 'sat'), (maps[s], 'map')):\n"
            "            link = raw / split / sub / src.name\n"
            "            if not link.exists():\n"
            "                link.symlink_to(src)\n"
            "    pairs_total += len(stems)\n"
            "    print(f'{split}: {len(stems)} pairs  (sat={sat_dir}, map={map_dir})')\n"
            "print('total pairs:', pairs_total)\n"
            "assert pairs_total > 0, 'No pairs found — attach the dataset or use the download cell below.'\n"
        ),
        code(
            "# Fallback ONLY (needs Internet enabled): download from the UofT mirror (~8 GB).\n"
            "# !python -m roadx.data.download --out raw_dl --all\n"
            "# Then point --raw at raw_dl in the next cell instead of raw.\n"
        ),
        code("!python -m roadx.data.tile --raw raw --out /kaggle/working/tiles\n"),
        code(
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "from PIL import Image\n"
            "\n"
            "tiles = Path('/kaggle/working/tiles')\n"
            "for split in ('train', 'valid', 'test'):\n"
            "    n = len(list((tiles / split / 'images').glob('*.png')))\n"
            "    print(f'{split}: {n} tiles')\n"
            "\n"
            "sample = sorted((tiles / 'train' / 'images').glob('*.png'))[0]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10, 5))\n"
            "ax[0].imshow(Image.open(sample)); ax[0].set_title('image'); ax[0].axis('off')\n"
            "ax[1].imshow(Image.open(tiles / 'train' / 'masks' / sample.name), cmap='gray')\n"
            "ax[1].set_title('mask'); ax[1].axis('off')\n"
            "plt.show()\n"
        ),
    ]
    return notebook(cells)


def train_notebook() -> dict:
    config_cells = [
        writefile_cell(f"configs/{name}.yaml", ROOT / "configs" / f"{name}.yaml")
        for name in ("unet", "unetpp", "deeplabv3plus", "linknet")
    ]
    cells = [
        md(
            "# Road extraction — train the 4 models (identical recipe)\n\n"
            "**Setup (once):**\n"
            "1. *Settings -> Accelerator -> GPU T4 x2* (NOT P100 — Kaggle's current PyTorch\n"
            "   no longer supports its sm_60 architecture).\n"
            "2. *Settings -> Internet -> ON* (pip + ImageNet encoder weights).\n"
            "3. *Add Input -> Your Work* -> the prepare-data notebook's output (the `tiles/` folder).\n"
            "4. Set `MODELS` below. The two models train in parallel, one per T4 (~4h wall);\n"
            "   run the notebook twice:\n"
            "   session A `['unet', 'unetpp']`, session B `['deeplabv3plus', 'linknet']`.\n"
            "5. *Save Version -> Save & Run All*. Download `runs/` from the output when done.\n"
        ),
        code("%pip install -q segmentation-models-pytorch albumentations\n"),
        code(SETUP),
        writefile_cell("roadx/data/dataset.py", SRC / "data" / "dataset.py"),
        writefile_cell("roadx/models.py", SRC / "models.py"),
        writefile_cell("roadx/losses.py", SRC / "losses.py"),
        writefile_cell("roadx/metrics.py", SRC / "metrics.py"),
        writefile_cell("roadx/train.py", SRC / "train.py"),
        *config_cells,
        code(
            "import os\n"
            "\n"
            "MODELS = ['unet', 'unetpp']  # session B: ['deeplabv3plus', 'linknet']\n"
            "BATCH_SIZE = 8  # identical for all models; 16 OOMs unetpp on a 16 GB T4\n"
            "\n"
            "# Kaggle mounts inputs at nested paths (/kaggle/input/datasets/<owner>/<slug>,\n"
            "# notebook outputs similarly), so search recursively for the tiles folder.\n"
            "hits = [d for d, _, _ in os.walk('/kaggle/input') if os.path.isdir(os.path.join(d, 'train', 'images'))]\n"
            "assert hits, 'tiles dataset not attached — add the prepare notebook output as input'\n"
            "DATA_DIR = hits[0]\n"
            "print('DATA_DIR =', DATA_DIR)\n"
        ),
        code(
            "# Train the models in parallel, one per GPU. A model failing (e.g. OOM) must\n"
            "# never abort the notebook — finished checkpoints in runs/ are saved regardless.\n"
            "import os, subprocess, sys, time\n"
            "\n"
            "procs = []\n"
            "for i, m in enumerate(MODELS):\n"
            "    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 2))\n"
            "    logf = open(f'{m}.log', 'w')\n"
            "    p = subprocess.Popen(\n"
            "        [sys.executable, '-m', 'roadx.train', '--model', m,\n"
            "         '--data-dir', DATA_DIR, '--batch-size', str(BATCH_SIZE), '--out', 'runs'],\n"
            "        stdout=logf, stderr=subprocess.STDOUT, env=env,\n"
            "    )\n"
            "    procs.append((m, p, logf))\n"
            "    print(f'launched {m} on GPU {i % 2}')\n"
            "\n"
            "def tail(fn):\n"
            "    try:\n"
            "        lines = [l.strip() for l in open(fn, errors='ignore') if 'epoch ' in l or 'done.' in l]\n"
            "        return lines[-1] if lines else '(warming up...)'\n"
            "    except FileNotFoundError:\n"
            "        return '(no log yet)'\n"
            "\n"
            "while any(p.poll() is None for _, p, _ in procs):\n"
            "    time.sleep(300)\n"
            "    for m, p, _ in procs:\n"
            "        state = 'running' if p.poll() is None else f'exited {p.returncode}'\n"
            "        print(f'[{m} | {state}] {tail(f\"{m}.log\")}', flush=True)\n"
            "\n"
            "print('=' * 70)\n"
            "for m, p, logf in procs:\n"
            "    logf.close()\n"
            "    print(f'{m}: exit code {p.returncode}')\n"
            "    log = open(f'{m}.log', errors='ignore').read()\n"
            "    print(log[-2500:])\n"
        ),
        code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "\n"
            "for run in sorted(Path('runs').glob('*/log.csv')):\n"
            "    df = pd.read_csv(run)\n"
            "    plt.plot(df['epoch'], df['val_iou'], label=run.parent.name)\n"
            "    print(run.parent.name, 'best val IoU:', df['val_iou'].max())\n"
            "plt.xlabel('epoch'); plt.ylabel('val IoU'); plt.legend(); plt.grid(alpha=0.3)\n"
            "plt.title('Validation IoU'); plt.show()\n"
        ),
    ]
    return notebook(cells)


DRIVE = "/content/drive/MyDrive/roadx"


def colab_prepare_notebook() -> dict:
    cells = [
        md(
            "# Massachusetts Roads — prepare 512x512 tiles (Colab)\n\n"
            "Run once. No GPU needed. Downloads the full dataset (~8 GB) from the original\n"
            "UofT mirror, tiles it, and stores `tiles.zip` in your Google Drive under\n"
            "`MyDrive/roadx/` (~4 GB) so training sessions can reuse it.\n\n"
            "*Runtime -> Run all*, approve the Drive permission popup, then wait (~30-45 min)."
        ),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            f"!mkdir -p {DRIVE}\n"
        ),
        code(SETUP),
        writefile_cell("roadx/data/download.py", SRC / "data" / "download.py"),
        writefile_cell("roadx/data/tile.py", SRC / "data" / "tile.py"),
        code("!python -m roadx.data.download --out /content/raw --all\n"),
        code("!python -m roadx.data.tile --raw /content/raw --out /content/tiles\n"),
        code(
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "from PIL import Image\n"
            "\n"
            "tiles = Path('/content/tiles')\n"
            "for split in ('train', 'valid', 'test'):\n"
            "    n = len(list((tiles / split / 'images').glob('*.png')))\n"
            "    print(f'{split}: {n} tiles')\n"
            "\n"
            "sample = sorted((tiles / 'train' / 'images').glob('*.png'))[0]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10, 5))\n"
            "ax[0].imshow(Image.open(sample)); ax[0].set_title('image'); ax[0].axis('off')\n"
            "ax[1].imshow(Image.open(tiles / 'train' / 'masks' / sample.name), cmap='gray')\n"
            "ax[1].set_title('mask'); ax[1].axis('off')\n"
            "plt.show()\n"
        ),
        code(
            "!cd /content && zip -qr tiles.zip tiles\n"
            f"!cp /content/tiles.zip {DRIVE}/tiles.zip\n"
            f"!ls -lh {DRIVE}/\n"
            "print('done — tiles.zip is in Drive, you can close this session')\n"
        ),
    ]
    return notebook(cells)


def colab_train_notebook() -> dict:
    config_cells = [
        writefile_cell(f"configs/{name}.yaml", ROOT / "configs" / f"{name}.yaml")
        for name in ("unet", "unetpp", "deeplabv3plus", "linknet")
    ]
    cells = [
        md(
            "# Road extraction — train one model per session (Colab)\n\n"
            "**Every session:**\n"
            "1. *Runtime -> Change runtime type -> T4 GPU*.\n"
            "2. Set `MODEL` in the parameters cell: run this notebook 4 times, once each with\n"
            "   `unet`, `unetpp`, `deeplabv3plus`, `linknet` (~2.5-3 h per model).\n"
            "3. *Runtime -> Run all*, approve the Drive popup.\n\n"
            "Checkpoints and logs stream straight to `MyDrive/roadx/runs/<model>/` during\n"
            "training, so a disconnect never loses a completed epoch's best checkpoint.\n"
            "Requires `tiles.zip` in `MyDrive/roadx/` (from the prepare notebook)."
        ),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
        ),
        code("%pip install -q segmentation-models-pytorch albumentations\n"),
        code(SETUP),
        writefile_cell("roadx/data/dataset.py", SRC / "data" / "dataset.py"),
        writefile_cell("roadx/models.py", SRC / "models.py"),
        writefile_cell("roadx/losses.py", SRC / "losses.py"),
        writefile_cell("roadx/metrics.py", SRC / "metrics.py"),
        writefile_cell("roadx/train.py", SRC / "train.py"),
        *config_cells,
        code(
            "MODEL = 'unet'  # <-- change per session: unet | unetpp | deeplabv3plus | linknet\n"
            "BATCH_SIZE = 16\n"
            "\n"
            "import os\n"
            f"assert os.path.exists('{DRIVE}/tiles.zip'), 'run the prepare notebook first'\n"
            "if not os.path.isdir('/content/tiles'):\n"
            f"    !unzip -q {DRIVE}/tiles.zip -d /content\n"
            "!ls /content/tiles\n"
        ),
        code(
            "import torch\n"
            "assert torch.cuda.is_available(), 'enable the GPU runtime first'\n"
            "print(torch.cuda.get_device_name(0))\n"
            "\n"
            "!python -m roadx.train --model {MODEL} --data-dir /content/tiles "
            f"--batch-size {{BATCH_SIZE}} --out {DRIVE}/runs\n"
        ),
        code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "\n"
            f"for run in sorted(Path('{DRIVE}/runs').glob('*/log.csv')):\n"
            "    df = pd.read_csv(run)\n"
            "    plt.plot(df['epoch'], df['val_iou'], label=run.parent.name)\n"
            "    print(run.parent.name, 'best val IoU:', df['val_iou'].max())\n"
            "plt.xlabel('epoch'); plt.ylabel('val IoU'); plt.legend(); plt.grid(alpha=0.3)\n"
            "plt.title('Validation IoU'); plt.show()\n"
        ),
    ]
    return notebook(cells)


HF_SPACE = "shuklaabhinavv/roadx"
GITHUB = "https://github.com/shuklaabhinavv/road-extraction"
DEMO = "https://huggingface.co/spaces/shuklaabhinavv/roadx"


def showcase_notebook() -> dict:
    """Public Kaggle showcase: results summary + zero-setup inference demo."""
    cells = [
        md(
            "# Road Extraction from Aerial Imagery — a 4-Model Comparative Study\n\n"
            "Which segmentation architecture is best for extracting road networks from\n"
            "aerial imagery? This project trained **U-Net, U-Net++, DeepLabV3+ and\n"
            "LinkNet** under a *strictly identical* protocol (same ResNet-34 encoder,\n"
            "loss, optimizer, schedule, augmentation, batch size and data) on the\n"
            "Massachusetts Roads Dataset, so measured differences come from the\n"
            "architecture alone.\n\n"
            "## Test-set results (441 held-out tiles)\n\n"
            "| Model | IoU | F1 | Precision | Recall | Params (M) | ms/tile |\n"
            "|---|---|---|---|---|---|---|\n"
            "| **U-Net++** | **0.654** | **0.791** | 0.808 | 0.774 | 26.1 | 199.8 |\n"
            "| U-Net | 0.650 | 0.788 | 0.807 | 0.770 | 24.4 | 110.5 |\n"
            "| LinkNet | 0.649 | 0.787 | 0.803 | 0.771 | **21.8** | **105.0** |\n"
            "| DeepLabV3+ | 0.642 | 0.782 | 0.798 | 0.766 | 22.4 | 117.5 |\n"
            "| Ensemble (x4) | 0.656 | 0.792 | 0.814 | 0.772 | 94.7 | 488.3 |\n\n"
            "**Highlights**\n"
            "- U-Net++ wins on accuracy; LinkNet delivers 99% of that accuracy at half\n"
            "  the inference cost.\n"
            "- Zero-shot transfer to DeepGlobe (India/Indonesia/Thailand) *inverts* the\n"
            "  ranking: LinkNet generalizes best (IoU 0.159), U-Net++ worst (0.125).\n"
            "- Geo-referenced output agrees with OpenStreetMap as well as the human\n"
            "  ground truth does (correctness 0.985 vs 0.979).\n\n"
            f"🔗 [Source code]({GITHUB}) · [Live demo — extract roads anywhere on Earth]({DEMO})\n\n"
            "---\n"
            "Below: load the trained weights and run road extraction on test images,\n"
            "end-to-end, right here. *(Attach the `massachusetts-roads-dataset` by\n"
            "balraj98 as input, enable Internet, then Run All — no GPU needed.)*"
        ),
        code("%pip install -q segmentation-models-pytorch huggingface_hub\n"),
        code(SETUP),
        writefile_cell("roadx/data/dataset.py", SRC / "data" / "dataset.py"),
        writefile_cell("roadx/models.py", SRC / "models.py"),
        writefile_cell("roadx/predict.py", SRC / "predict.py"),
        code(
            "# trained checkpoints are hosted in the project's Hugging Face Space\n"
            "from huggingface_hub import hf_hub_download\n"
            "\n"
            "CKPTS = {}\n"
            "for name in ('unetpp', 'linknet'):\n"
            f"    CKPTS[name] = hf_hub_download('{HF_SPACE}', f'runs/{{name}}/best.pt', repo_type='space')\n"
            "    print('downloaded', name)\n"
        ),
        code(
            "import os\n"
            "import numpy as np\n"
            "import torch\n"
            "from PIL import Image\n"
            "from roadx.models import build_model, pick_device\n"
            "from roadx.predict import predict_image, overlay\n"
            "\n"
            "device = pick_device()\n"
            "models = {}\n"
            "for name, path in CKPTS.items():\n"
            "    ckpt = torch.load(path, map_location='cpu', weights_only=False)\n"
            "    m = build_model(ckpt['model'], ckpt['encoder'], encoder_weights=None)\n"
            "    m.load_state_dict(ckpt['state_dict'])\n"
            "    m.to(device).eval()\n"
            "    models[name] = m\n"
            "print('loaded', list(models), 'on', device.type)\n"
        ),
        code(
            "# find test images in the attached public dataset (any folder layout)\n"
            "test_imgs = []\n"
            "for dirpath, _, files in os.walk('/kaggle/input'):\n"
            "    if dirpath.endswith(('test', 'test_sat')) or '/test' in dirpath:\n"
            "        test_imgs += [os.path.join(dirpath, f) for f in files\n"
            "                      if f.endswith(('.tiff', '.tif', '.png', '.jpg')) and 'label' not in dirpath]\n"
            "test_imgs = sorted(set(test_imgs))[:3]\n"
            "assert test_imgs, 'attach the massachusetts-roads-dataset (balraj98) as input'\n"
            "print(len(test_imgs), 'test images selected')\n"
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "for path in test_imgs:\n"
            "    img = np.asarray(Image.open(path).convert('RGB'))\n"
            "    fig, axes = plt.subplots(1, 1 + len(models), figsize=(5.5 * (1 + len(models)), 5.5))\n"
            "    axes[0].imshow(img); axes[0].set_title('input'); axes[0].axis('off')\n"
            "    for ax, (name, m) in zip(axes[1:], models.items()):\n"
            "        mask = predict_image(m, img, device) > 0.5\n"
            "        ax.imshow(overlay(img, mask))\n"
            "        ax.set_title(f'{name} — {mask.mean():.1%} road'); ax.axis('off')\n"
            "    plt.tight_layout(); plt.show()\n"
        ),
        md(
            "## How this was built\n"
            "The full pipeline (data prep, identical-protocol training on Kaggle T4s,\n"
            "evaluation, geo-referencing with OpenStreetMap validation, and the web app)\n"
            f"is open source: **[{GITHUB.split('//')[1]}]({GITHUB})**.\n\n"
            f"Try the interactive demo — extract roads over your own city: **[{DEMO.split('//')[1]}]({DEMO})**"
        ),
    ]
    return notebook(cells)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, nb in (
        ("colab_1_prepare_data.ipynb", colab_prepare_notebook()),
        ("colab_2_train.ipynb", colab_train_notebook()),
        ("kaggle_1_prepare_data.ipynb", prepare_notebook()),
        ("kaggle_2_train.ipynb", train_notebook()),
        ("kaggle_showcase.ipynb", showcase_notebook()),
    ):
        (OUT / name).write_text(json.dumps(nb, indent=1))
        print(f"wrote notebooks/{name}")


if __name__ == "__main__":
    main()
