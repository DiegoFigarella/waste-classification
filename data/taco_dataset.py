import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

import sys
sys.path.insert(0, "./")  # add repo root to path
from data.constants import TACO_ROOT, SEED, TACO_SPLIT_FRACTIONS

LABEL_LEVELS = ("supercategory", "category")


class TACODataset(Dataset):
    """TACO (http://tacodataset.org) as a crop-level classification dataset.

    TACO ships COCO-style detection annotations, so every annotated object becomes one
    sample: its bounding box is cropped out of the photo and labelled with its
    supercategory (28 classes, default) or fine-grained category (60 classes).
    """

    def __init__(
        self,
        root_dir=TACO_ROOT,
        split="train",
        transform=None,
        label_level="supercategory",
        annotations_file="annotations.json",
    ):
        if split not in (*TACO_SPLIT_FRACTIONS, "all"):
            raise ValueError(f"Unknown split {split}. Choose from {(*TACO_SPLIT_FRACTIONS, 'all')}")
        if label_level not in LABEL_LEVELS:
            raise ValueError(f"Unknown label_level {label_level}. Choose from {LABEL_LEVELS}")
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_level = label_level
        self.annotations_file = annotations_file
        self.load_metadata()
        if not self.samples:
            raise ValueError(f"No samples found in {self.root_dir} for {self.split} split")

    def __len__(self):
        return len(self.samples)

    def load_metadata(self):
        """Populate self.samples with (path, bbox, label) and self.classes with label names."""
        root = Path(self.root_dir)
        with open(root / self.annotations_file) as f:
            coco = json.load(f)

        # Map category id -> label name, then label name -> contiguous index
        key = "name" if self.label_level == "category" else "supercategory"
        cat_to_name = {c["id"]: c[key] for c in coco["categories"]}
        self.classes = sorted(set(cat_to_name.values()))
        name_to_idx = {name: i for i, name in enumerate(self.classes)}

        image_ids = sorted(img["id"] for img in coco["images"])
        keep_ids = set(image_ids) if self.split == "all" else self._split_ids(image_ids)
        id_to_path = {img["id"]: str(root / img["file_name"]) for img in coco["images"]}

        self.samples = [
            (id_to_path[ann["image_id"]], ann["bbox"], name_to_idx[cat_to_name[ann["category_id"]]])
            for ann in coco["annotations"]
            if ann["image_id"] in keep_ids
        ]

    def _split_ids(self, image_ids):
        """Deterministic split over image ids so crops of one photo stay in one split."""
        ids = image_ids[:]
        random.Random(SEED).shuffle(ids)
        start = 0
        for name, frac in TACO_SPLIT_FRACTIONS.items():
            # last split takes the remainder so no image is dropped by rounding
            end = len(ids) if name == list(TACO_SPLIT_FRACTIONS)[-1] else start + round(frac * len(ids))
            if name == self.split:
                return set(ids[start:end])
            start = end

    def __getitem__(self, idx):
        path, (x, y, w, h), label = self.samples[idx]
        assert self.transform is not None, "Transform must be provided"

        # TACO bboxes are in EXIF-corrected orientation, so rotate before cropping
        # ponytail: reopens the full photo per crop; cache decoded images if loading is the bottleneck
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = img.crop((x, y, x + w, y + h))
        img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label
