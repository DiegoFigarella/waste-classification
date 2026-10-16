"""Self-check for TACODataset on a tiny synthetic COCO file. Run: python data/test_taco_dataset.py"""
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, "./")
from PIL import Image
from torchvision import transforms
from data.taco_dataset import TACODataset


def make_fake_taco(root, n_images=20):
    (root / "batch_1").mkdir(parents=True)
    categories = [
        {"id": 0, "name": "Clear plastic bottle", "supercategory": "Bottle"},
        {"id": 1, "name": "Glass bottle", "supercategory": "Bottle"},
        {"id": 2, "name": "Drink can", "supercategory": "Can"},
    ]
    images, annotations = [], []
    for i in range(n_images):
        Image.new("RGB", (64, 48)).save(root / f"batch_1/{i:06d}.jpg")
        images.append({"id": i, "file_name": f"batch_1/{i:06d}.jpg", "width": 64, "height": 48})
        for j in range(2):  # two objects per photo
            annotations.append({"id": 2 * i + j, "image_id": i, "category_id": (i + j) % 3,
                                "bbox": [5.0, 5.0, 20.5, 10.0]})
    with open(root / "annotations.json", "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)


def test_taco_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_taco(root)
        tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

        splits = {s: TACODataset(root, split=s, transform=tf) for s in ("train", "val", "test")}
        assert sum(len(d) for d in splits.values()) == len(TACODataset(root, split="all", transform=tf)) == 40

        # no photo leaks across splits
        paths = [{p for p, _, _ in d.samples} for d in splits.values()]
        assert not (paths[0] & paths[1]) and not (paths[0] & paths[2]) and not (paths[1] & paths[2])

        # split is deterministic
        assert TACODataset(root, split="val", transform=tf).samples == splits["val"].samples

        assert splits["train"].classes == ["Bottle", "Can"]
        fine = TACODataset(root, split="all", transform=tf, label_level="category")
        assert fine.classes == ["Clear plastic bottle", "Drink can", "Glass bottle"]

        img, label = splits["train"][0]
        assert img.shape == (3, 32, 32) and label.item() in (0, 1)


if __name__ == "__main__":
    test_taco_dataset()
    print("ok")
