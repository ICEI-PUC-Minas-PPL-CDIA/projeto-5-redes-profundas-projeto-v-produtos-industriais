import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

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
                A.Normalize(
                    mean=(0.5071, 0.4865, 0.4409),
                    std=(0.2675, 0.2565, 0.2761)
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(
                    mean=(0.5071, 0.4865, 0.4409),
                    std=(0.2675, 0.2565, 0.2761)
                ),
                ToTensorV2()
            ])

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        return augmented['image']
