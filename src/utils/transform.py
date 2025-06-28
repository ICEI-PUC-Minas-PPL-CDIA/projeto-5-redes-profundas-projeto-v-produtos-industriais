import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

class AlbumentationsTransform:
    def __init__(self, mode="train"):
        if mode == "train":
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.MotionBlur(blur_limit=(3, 7), p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.HueSaturationValue(p=0.3),
                ], p=0.7),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                ToTensorV2()
            ])

    def __call__(self, image):
        # Garante que a imagem está em RGB
        image_np = np.array(image.convert("RGB"))

        # Aplica as transformações
        augmented = self.transform(image=image_np)
        tensor = augmented['image']

        # Garante que a saída está em float32 no intervalo [0, 1]
        if tensor.dtype != torch.float32:
            tensor = tensor.float() / 255.0

        return tensor
