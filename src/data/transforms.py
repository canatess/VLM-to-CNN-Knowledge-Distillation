import torchvision.transforms as T


def build_transforms(image_size: int = 224, train: bool = True):
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return T.Compose([
            T.Resize(int(image_size * 1.15)),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(10),
            T.ToTensor(),
            normalize
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.15)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            normalize
        ])


def build_clip_transforms(image_size: int = 224):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
