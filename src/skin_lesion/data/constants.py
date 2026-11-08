"""HAM10000 label constants."""

# Order defines the integer label (index) used everywhere.
CLASSES: list[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
MELANOMA_IDX: int = CLASS_TO_IDX["mel"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
