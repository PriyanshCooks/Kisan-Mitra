"""
Real-world evaluation of the Kisan Mitra plant disease model.

The model reached 99.2% validation accuracy and 33/33 on the test folder,
but both came from the same PlantVillage source as the training data.
This script checks how it does on leaf photos taken outside that dataset.

Setup
-----
Run from the repo root (the folder that contains `app/`).

Put your photos in `real_images/`, one subfolder per TRUE label, using the
exact class names from DISEASE_CLASSES below. For example:

    real_images/
        Tomato___healthy/            leaf1.jpg, leaf2.jpg
        Tomato___Early_blight/       blight1.jpg
        Potato___Late_blight/        img_0042.jpg

If you're not sure what disease a leaf has, put it in `real_images/unknown/`.
Those images are still scored for confidence, just not for correctness.

Run
---
    python real_world_eval.py

It prints a summary and writes `real_world_results.csv`.

It runs every image through two preprocessing paths:
  - "app":    exactly what app/app.py does today (Resize(256), aspect ratio kept)
  - "square": Resize to 256x256, matching the square images the model was trained on
so you can also see whether the app's preprocessing costs accuracy.
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path("app").resolve()))
from utils.model import ResNet9  # noqa: E402

DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

MODEL_PATH = Path("app/models/plant_disease_model.pth")
IMAGE_DIR = Path("real_images")
OUT_CSV = Path("real_world_results.csv")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PREPROCESS = {
    "app": transforms.Compose([transforms.Resize(256), transforms.ToTensor()]),
    "square": transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
}


def crop_of(label: str) -> str:
    return label.split("___")[0]


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run this from the repo root.")
    if not IMAGE_DIR.exists():
        sys.exit(f"Create {IMAGE_DIR}/ with one subfolder per true label (see the top of this file).")

    model = ResNet9(3, len(DISEASE_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    rows = []
    for folder in sorted(p for p in IMAGE_DIR.iterdir() if p.is_dir()):
        true_label = folder.name
        known = true_label in DISEASE_CLASSES
        if not known and true_label != "unknown":
            print(f"Skipping folder '{true_label}': not a class name and not 'unknown'.")
            continue

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            # The app does not call .convert("RGB"), so a PNG with transparency
            # (4 channels) would crash it. Record that, then convert so we can score it.
            raw = Image.open(img_path)
            app_would_crash = raw.mode != "RGB"
            image = raw.convert("RGB")

            for mode, tf in PREPROCESS.items():
                try:
                    with torch.no_grad():
                        probs = F.softmax(model(tf(image).unsqueeze(0)), dim=1)[0]
                except RuntimeError:
                    # ResNet9's classifier (MaxPool2d(4) + Linear(512)) only works on
                    # inputs that end up 4x4 before pooling, i.e. square ~256x256 images.
                    # Non-square images crash the model; record that instead of aborting.
                    rows.append({
                        "image": str(img_path), "preprocess": mode, "true_label": true_label,
                        "predicted": "CRASH", "confidence": "", "correct": "", "correct_crop": "",
                        "top3": "", "app_would_crash_non_rgb": app_would_crash,
                        "model_crashed": True,
                    })
                    continue
                top3 = torch.topk(probs, 3)
                pred = DISEASE_CLASSES[top3.indices[0].item()]
                rows.append({
                    "image": str(img_path),
                    "preprocess": mode,
                    "true_label": true_label,
                    "predicted": pred,
                    "confidence": round(top3.values[0].item(), 4),
                    "correct": (pred == true_label) if known else "",
                    "correct_crop": (crop_of(pred) == crop_of(true_label)) if known else "",
                    "top3": " | ".join(
                        f"{DISEASE_CLASSES[i.item()]} ({v.item():.2f})"
                        for v, i in zip(top3.values, top3.indices)
                    ),
                    "app_would_crash_non_rgb": app_would_crash,
                    "model_crashed": False,
                })

    if not rows:
        sys.exit(f"No images found under {IMAGE_DIR}/.")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nScored {len({r['image'] for r in rows})} images. Results: {OUT_CSV}\n")
    for mode in PREPROCESS:
        all_sub = [r for r in rows if r["preprocess"] == mode]
        crashed = sum(r["model_crashed"] for r in all_sub)
        sub = [r for r in all_sub if not r["model_crashed"]]
        labelled = [r for r in sub if r["correct"] != ""]
        wrong = [r for r in labelled if not r["correct"]]
        print(f"[{mode}]  ({crashed}/{len(all_sub)} images crashed the model; stats below exclude them)")
        if labelled:
            n = len(labelled)
            acc = sum(r["correct"] for r in labelled) / n
            crop_acc = sum(r["correct_crop"] for r in labelled) / n
            print(f"  exact label correct: {n - len(wrong)}/{n} ({acc:.0%})")
            print(f"  right crop:          {crop_acc:.0%}")
            if wrong:
                avg_wrong_conf = sum(r["confidence"] for r in wrong) / len(wrong)
                confident_wrong = sum(r["confidence"] >= 0.9 for r in wrong)
                print(f"  avg confidence when wrong: {avg_wrong_conf:.0%}")
                print(f"  wrong with >=90% confidence: {confident_wrong}/{len(wrong)}")
        unk = [r for r in sub if r["correct"] == ""]
        if unk:
            print(f"  unlabelled images: {len(unk)}, avg confidence {sum(r['confidence'] for r in unk) / len(unk):.0%}")
        print()

    crashes = len({r["image"] for r in rows if r["app_would_crash_non_rgb"]})
    if crashes:
        print(f"{crashes} image(s) are not plain RGB; the deployed app would fail on these as-is.")


if __name__ == "__main__":
    main()
