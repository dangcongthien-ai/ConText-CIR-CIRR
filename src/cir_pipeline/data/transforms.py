"""Image transforms for CLIP-style inputs."""

from __future__ import annotations


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_BACKBONE_IMAGE_SIZES: list[tuple[str, int]] = [
    ("clip-rn50x64", 448),
    ("clip-rn50x16", 384),
    ("clip-rn50x4", 288),
    ("patch14-336", 336),
]


def infer_clip_image_size(backbone: str | None = None, default: int = 224) -> int:
    """Infer the CLIP image size from a known backbone name."""

    normalized = str(backbone or "").lower()
    for key, size in _BACKBONE_IMAGE_SIZES:
        if key in normalized:
            return size
    return default


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """Square-pad an input PIL image with zeros."""

    def __call__(self, image):
        from torchvision.transforms import functional as tf

        width, height = image.size
        max_wh = max(width, height)
        pad_w = int((max_wh - width) / 2)
        pad_h = int((max_wh - height) / 2)
        padding = [pad_w, pad_h, pad_w, pad_h]
        return tf.pad(image, padding, 0, "constant")


class TargetPad:
    """Pad an image only when its aspect ratio exceeds the target ratio."""

    def __init__(self, target_ratio: float) -> None:
        self.target_ratio = target_ratio

    def __call__(self, image):
        from torchvision.transforms import functional as tf

        width, height = image.size
        actual_ratio = max(width, height) / max(1, min(width, height))
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(width, height) / self.target_ratio
        pad_w = max(int((scaled_max_wh - width) / 2), 0)
        pad_h = max(int((scaled_max_wh - height) / 2), 0)
        padding = [pad_w, pad_h, pad_w, pad_h]
        return tf.pad(image, padding, 0, "constant")


def build_clip_transform(
    image_size: int | None = None,
    train: bool = False,
    backbone: str | None = None,
    transform_name: str = "clip",
    target_pad_ratio: float = 1.25,
):
    """Return a PIL-to-tensor transform matching CLIP normalization."""

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    resolved_image_size = int(image_size or infer_clip_image_size(backbone))
    transform_key = str(transform_name or "clip").lower()

    if transform_key == "clip":
        resize = (
            transforms.RandomResizedCrop(
                resolved_image_size,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            if train
            else transforms.Compose(
                [
                    transforms.Resize(
                        resolved_image_size,
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.CenterCrop(resolved_image_size),
                ]
            )
        )
        image_ops = [resize]
    elif transform_key == "squarepad":
        image_ops = [
            SquarePad(),
            transforms.Resize(
                resolved_image_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(resolved_image_size),
        ]
    elif transform_key == "targetpad":
        image_ops = [
            TargetPad(float(target_pad_ratio)),
            transforms.Resize(
                resolved_image_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(resolved_image_size),
        ]
    else:
        raise ValueError("transform_name must be one of: clip, squarepad, targetpad.")

    return transforms.Compose(
        [
            *image_ops,
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
