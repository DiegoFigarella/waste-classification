# Dataset Summary: TACO (Trash Annotations in Context)

## Quick Stats

| Attribute | Details |
| :--- | :--- |
| **Total Size** | 1,500 images / 4,784 annotated objects (official set). Disk size TBD after download |
| **License(s)** | Annotations CC BY 4.0, code MIT, images under their individual Flickr licenses |
| **Source(s)** | [TACO website](http://tacodataset.org), [GitHub](https://github.com/pedropro/TACO) |
| **Data Format(s)** | .jpg images + COCO-format `annotations.json` |
| **Classes** | 60 categories grouped into 28 supercategories (highly imbalanced, see `eda.ipynb`) |
| **Resolution** | Variable, mostly full-resolution phone photos |

## Metadata Attributes (Column Mapping)
How the COCO JSON fields map to `TACODataset` (`data/taco_dataset.py`):

| Original Variable | Our TACODataset Mapping | Description |
| :--- | :--- | :--- |
| `images[].file_name` | sample path | Path relative to `data/`, e.g. `batch_1/000006.jpg` |
| `images[].width/height` | not used | Size in the EXIF-rotated (upright) frame |
| `annotations[].bbox` | crop box | `[x, y, w, h]` in pixels, upright frame |
| `annotations[].category_id` | label (via categories) | Links an object to its category |
| `categories[].supercategory` | label (`label_level="supercategory"`, default) | 28 coarse classes, e.g. `Bottle`, `Can`, `Cigarette` |
| `categories[].name` | label (`label_level="category"`) | 60 fine classes, e.g. `Clear plastic bottle` |
| `annotations[].segmentation` | not used | Polygon masks |

## How to Access the Dataset?
```shell
git clone https://github.com/pedropro/TACO.git
cd TACO
pip install -r requirements.txt
python download.py   # pulls images from Flickr into data/batch_*/
```
Then set `TACO_ROOT` in `data/constants.py` to `TACO/data`.

```python
from data.taco_dataset import TACODataset
from data.constants import DEFAULT_TRAIN_TRANSFORMS

ds = TACODataset(split="train", transform=DEFAULT_TRAIN_TRANSFORMS)
```

## Dataset Structure

```
TACO/data/
├── annotations.json              # official COCO annotations
├── annotations_unofficial.json   # extra community annotations (not used)
├── batch_1/
│   ├── 000000.jpg
│   └── ...
└── batch_15/
```

## Potential Issues
- **No official split**: `TACODataset` makes a seeded 80/10/10 split by image id so crops of one photo stay in one split. Rare classes may be missing from val/test.
- **Class imbalance**: a few supercategories (cigarettes, plastic film, bottles) dominate; many have only a handful of objects.
- **Small objects**: many boxes are tiny relative to the photo, so crops are upsampled heavily when resized.
- **EXIF orientation**: annotations are in the upright frame, so images must go through `ImageOps.exif_transpose` before cropping (the loader does this).
- **Flickr availability**: images are fetched from Flickr URLs, so some may fail to download.
- **Detection data used for classification**: crops lose scene context and can contain overlapping objects.

## Overlap With Other Datasets
| Source | Reference |
|:--|:--|
| Flickr | Original image host |

## Miscellaneous Details
- **Loading cost**: each crop reopens the full-resolution photo. Cache crops to disk if data loading becomes the bottleneck.
