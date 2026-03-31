# Day 2 Training Handoff

**Written:** 2026-03-30 (end of Day 1)  
**Purpose:** Resume `police_full_v1` training on Day 2 — either on WSL2 RTX 4060 or Google Colab.

---

## What Day 1 Completed

| Step | Result |
|------|--------|
| pip + ultralytics + PyTorch 2.11 (CUDA 13) installed | `/usr/bin/python3` |
| VisDrone downloaded + remapped to 6 police classes | `data/visdrone_police/` — 8629 images |
| **Baseline trained** YOLO11n, 30 epochs, VisDrone only | `best.pt` — **mAP50=0.480** |
| ONNX export of baseline | `best.onnx` — 10.1MB |
| D-Fire (fire/smoke) — 21K images via Kaggle | `data/supplementary/fire_dfire/` |
| ShanghaiTech crowd — 482 images via Kaggle | `data/supplementary/crowd_shanghaitech/` |
| Synthetic weapon_person — re-labeled VisDrone persons | `data/supplementary/weapon_synthetic/` |
| All supplementary merged + merged into visdrone_police | `data/visdrone_police/` — **31,538 total images** |
| **Full model training started** YOLO11s, target 100 epochs | Completed **1/100 epochs** |

**Epoch 1 result:** P=0.658, R=0.391, mAP50=0.406, mAP50-95=0.205

---

## Checkpoint to Resume From

```
Path:  runs/detect/runs/detect/police_full_v1/weights/last.pt
Size:  55 MB (YOLO11s — 9.4M params)
MD5:   3a5c3c4dd6406dafbd8a3c2f09921d08
State: epoch 1/100 complete
```

---

## Final Dataset State (what is on disk)

```
data/visdrone_police/
  images/train/   21,693 images  (VisDrone 6,471 + supplementary 15,222)
  images/val/      8,235 images
  labels/...      matched label files
  
Class distribution — train (408,545 instances):
  0  person          122,027  (29.9%)
  1  weapon_person     1,176  (0.3%)  ← synthetic
  2  vehicle         267,610  (65.5%)
  3  fire             17,432  (4.3%)  ← D-Fire
  4  explosive_device      0  ← DEFERRED (no dataset)
  5  crowd               300  (0.1%)  ← ShanghaiTech

config/training/visdrone_police.yaml   ← training config
```

---

## Option A — Resume on WSL2 RTX 4060 (same machine)

Dataset and checkpoint are already on disk.

```bash
cd /mnt/d/Sanjay_MK2

# Quick sanity check
ls -lh runs/detect/runs/detect/police_full_v1/weights/last.pt

# Resume — Ultralytics auto-detects epoch number and continues
/usr/bin/python3 -c "
from ultralytics import YOLO
model = YOLO('runs/detect/runs/detect/police_full_v1/weights/last.pt')
model.train(resume=True)
" 2>&1 | tee reports/day2/police_full_v1_resumed.log

# Watch progress
tail -f reports/day2/police_full_v1_resumed.log
```

If `resume=True` fails (run config missing), use explicit args:

```bash
/usr/bin/python3 scripts/train_yolo.py \
  --train \
  --model runs/detect/runs/detect/police_full_v1/weights/last.pt \
  --epochs 100 \
  --device 0 \
  --name police_full_v1_day2
```

---

## Option B — Google Colab (Recommended — A100 is ~10x faster)

### B1. Upload checkpoint to Google Drive

Upload this file to your Drive before opening Colab:
```
Local:  runs/detect/runs/detect/police_full_v1/weights/last.pt  (55MB)
Drive:  My Drive/SanjayMK2/checkpoints/police_full_v1_epoch1.pt
```

### B2. Rebuild dataset on Colab (do NOT upload the 15GB dataset)

Run these cells in order in a new Colab notebook:

```python
# CELL 1 — Install + clone
!pip install ultralytics kagglehub scipy pillow -q
!git clone https://github.com/YOUR_ORG/Sanjay_MK2 /content/Sanjay_MK2
%cd /content/Sanjay_MK2
```

```python
# CELL 2 — VisDrone download + remap (~2GB, ~12 min)
!python scripts/train_yolo.py --setup-visdrone
```

```python
# CELL 3 — D-Fire from Kaggle (~2.84GB, ~5 min)
import kagglehub, shutil
from pathlib import Path

path = kagglehub.dataset_download('sayedgamal99/smoke-fire-detection-yolo')
FIRE_SRC = Path(path) / 'data'
FIRE_DST = Path('data/supplementary/fire_dfire')

for ss, sd in [('train','train'), ('val','val'), ('test','val')]:
    sl, si = FIRE_SRC/ss/'labels', FIRE_SRC/ss/'images'
    dl, di = FIRE_DST/'labels'/sd, FIRE_DST/'images'/sd
    dl.mkdir(parents=True, exist_ok=True)
    di.mkdir(parents=True, exist_ok=True)
    if not sl.exists(): continue
    for lp in sorted(sl.glob('*.txt')):
        lines = []
        with open(lp) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    p[0] = '3'  # smoke=0 and fire=1 both → class 3 (fire)
                    lines.append(' '.join(p))
        with open(dl/lp.name, 'w') as f:
            f.write('\n'.join(lines) + '\n' if lines else '')
        stem = lp.stem
        for ext in ['.jpg','.jpeg','.png']:
            isrc = si/(stem+ext)
            if isrc.exists():
                idst = di/(stem+ext)
                if not idst.exists(): shutil.copy2(isrc, idst)
                break
print('D-Fire done — train:', len(list((FIRE_DST/'images'/'train').glob('*'))),
      'val:', len(list((FIRE_DST/'images'/'val').glob('*'))))
```

```python
# CELL 4 — ShanghaiTech crowd from Kaggle (~333MB, ~2 min)
import kagglehub, shutil, numpy as np
from pathlib import Path
from scipy.io import loadmat
from PIL import Image

path = kagglehub.dataset_download('tthien/shanghaitech')
CS = Path(path) / 'ShanghaiTech' / 'part_A'
CD = Path('data/supplementary/crowd_shanghaitech')

for ss, sd in [('train_data','train'), ('test_data','val')]:
    ig = CS/ss/'images'; gt = CS/ss/'ground-truth'
    di = CD/'images'/sd; dl = CD/'labels'/sd
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    if not ig.exists(): continue
    c = 0
    for ip in sorted(ig.glob('*.jpg')):
        gp = gt/f'GT_{ip.stem}.mat'
        if not gp.exists(): continue
        try:
            mat = loadmat(str(gp))
            pts = mat['image_info'][0][0][0][0][0]
            if len(pts) < 5: continue
            im = Image.open(ip); w, h = im.size
            xn = max(0, pts[:,0].min()-20)/w; yn = max(0, pts[:,1].min()-20)/h
            xx = min(w, pts[:,0].max()+20)/w; yx = min(h, pts[:,1].max()+20)/h
            cx=(xn+xx)/2; cy=(yn+yx)/2; bw=xx-xn; bh=yx-yn
            shutil.copy2(ip, di/ip.name)
            with open(dl/f'{ip.stem}.txt','w') as f:
                f.write(f'5 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            c += 1
        except: continue
    print(f'ShanghaiTech {ss}->{sd}: {c} images')
```

```python
# CELL 5 — Synthetic weapon_person from VisDrone persons
import shutil, random
from pathlib import Path
random.seed(42)
VS = Path('data/visdrone_police')
WD = Path('data/supplementary/weapon_synthetic')

for split, max_n in [('train',800), ('val',100)]:
    sl = VS/'labels'/split; si = VS/'images'/split
    dl = WD/'labels'/split; di = WD/'images'/split
    dl.mkdir(parents=True, exist_ok=True); di.mkdir(parents=True, exist_ok=True)
    cands = []
    for lp in sorted(sl.glob('*.txt')):
        boxes = [line.strip().split() for line in open(lp) if len(line.strip().split())>=5]
        persons = [i for i,b in enumerate(boxes) if b[0]=='0']
        if persons: cands.append((lp, boxes, persons))
    random.shuffle(cands)
    c = 0
    for lp, boxes, persons in cands[:max_n]:
        stem = lp.stem
        n_wpn = min(random.randint(1,2), len(persons))
        wpn_idx = set(random.sample(persons, n_wpn))
        new_lines = [' '.join(['1']+b[1:]) if i in wpn_idx else ' '.join(b)
                     for i,b in enumerate(boxes)]
        img = next((si/(stem+ext) for ext in ['.jpg','.jpeg','.png']
                    if (si/(stem+ext)).exists()), None)
        if not img: continue
        shutil.copy2(img, di/f'wpn_{stem}{img.suffix}')
        with open(dl/f'wpn_{stem}.txt','w') as f: f.write('\n'.join(new_lines)+'\n')
        c += 1
    print(f'weapon_person {split}: {c}')
```

```python
# CELL 6 — Merge all supplementary into visdrone_police
!python scripts/prepare_supplementary_data.py --merge-all
!python scripts/train_yolo.py --merge data/supplementary_merged --auto-prefix
!python scripts/audit_dataset.py data/visdrone_police
# Expected: ~31,538 total images, 5 classes populated
```

```python
# CELL 7 — Mount Drive and copy checkpoint
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy(
    '/content/drive/MyDrive/SanjayMK2/checkpoints/police_full_v1_epoch1.pt',
    'police_full_v1_epoch1.pt'
)
```

```python
# CELL 8 — Resume training (saves results back to Drive)
from ultralytics import YOLO

model = YOLO('police_full_v1_epoch1.pt')
results = model.train(
    data='config/training/visdrone_police.yaml',
    epochs=100,
    imgsz=640,
    device=0,
    batch=-1,           # auto batch
    patience=20,
    name='police_full_v1',
    project='/content/drive/MyDrive/SanjayMK2/runs',
    resume=True,
    exist_ok=True,
)
print('Best mAP50:', results.results_dict.get('metrics/mAP50(B)'))
```

```python
# CELL 9 — Download best weights when done
from google.colab import files
best_path = '/content/drive/MyDrive/SanjayMK2/runs/police_full_v1/weights/best.pt'
files.download(best_path)
```

---

## B3. After Colab — Validate on WSL2

Once you have `best.pt` back on the RTX machine:

```bash
cd /mnt/d/Sanjay_MK2
mkdir -p reports/day2

# Copy best.pt from wherever you downloaded it
cp ~/Downloads/best.pt runs/detect/runs/detect/police_full_v1/weights/best_day2.pt

# Validate against all 50 police scenarios
/usr/bin/python3 scripts/validate_model.py \
  --yolo runs/detect/runs/detect/police_full_v1/weights/best_day2.pt \
  --all --compare \
  2>&1 | tee reports/day2/police_full_v1_validation.log

# Audit final dataset state
/usr/bin/python3 scripts/audit_dataset.py data/visdrone_police \
  | tee reports/day2/dataset_audit.log
```

---

## Day 2 Success Bar

| Metric | Baseline (Day 1) | Target (Day 2) |
|--------|-----------------|----------------|
| mAP50 (all) | 0.480 | **> 0.55** |
| mAP50-95 (all) | 0.247 | **> 0.28** |
| person mAP50 | 0.329 | > 0.35 |
| vehicle mAP50 | 0.631 | > 0.65 |
| fire mAP50 | 0.000 | **> 0.40** |
| weapon_person mAP50 | 0.000 | > 0.10 |
| crowd mAP50 | 0.000 | > 0.15 |
| explosive_device mAP50 | 0.000 | 0.000 (deferred) |

---

## Known Issues to Watch

- **Out-of-bounds D-Fire labels:** A few D-Fire images have coordinates slightly > 1.0. YOLO skips them automatically — non-fatal, ~20 images affected.
- **weapon_person is synthetic:** Expect low precision at first. Real weapon data (Roboflow API key required) will improve it in a future run.
- **crowd is ground-level images:** ShanghaiTech is not aerial. Expect moderate performance until DroneCrowd (manual download required) is added.
- **explosive_device:** Zero instances, stays zero until a real dataset decision is made.
- **`resume=True` requires the run `.yaml` file** to exist alongside the checkpoint. If the Colab runtime resets between cells 7 and 8, re-run cell 7 first. If the yaml is missing, remove `resume=True` and add `epochs=99` manually.

---

## Quick Reference — File Paths

| Item | Path |
|------|------|
| **Checkpoint (resume from)** | `runs/detect/runs/detect/police_full_v1/weights/last.pt` |
| Baseline best (YOLO11n) | `runs/detect/runs/detect/visdrone_baseline_day1/weights/best.pt` |
| Baseline ONNX | `runs/detect/runs/detect/visdrone_baseline_day1/weights/best.onnx` |
| Training config | `config/training/visdrone_police.yaml` |
| Merged dataset | `data/visdrone_police/` |
| Fire supplementary | `data/supplementary/fire_dfire/` |
| Crowd supplementary | `data/supplementary/crowd_shanghaitech/` |
| Weapon supplementary | `data/supplementary/weapon_synthetic/` |
| Day 1 logs | `reports/day1/` |
| Day 1 status | `reports/day1/STATUS.md` |
