from torchvision import transforms

DATA_ROOT = "/data"  # set for your machine!
TACO_ROOT = f"{DATA_ROOT}/TACO/data"  # folder holding annotations.json and batch_*/
SEED = 42
IMG_DIM = 224
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# TACO has no official split, so we split by image id (crops from one photo never cross splits)
TACO_SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Training: includes augmentation
DEFAULT_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_DIM, IMG_DIM)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
])

# Evaluation: no augmentation
DEFAULT_EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_DIM, IMG_DIM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
])
