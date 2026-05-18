"""Image transforms for CLIP-style inputs."""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_clip_transform(image_size: int = 224, train: bool = False):
    """Return a PIL-to-tensor transform matching CLIP normalization."""

    if train:
        resize = transforms.RandomResizedCrop(
            image_size,
            scale=(0.9, 1.0),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    else:
        resize = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(image_size),
            ]
        )
    return transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
