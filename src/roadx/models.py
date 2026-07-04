import segmentation_models_pytorch as smp
import torch

ARCHS = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "linknet": smp.Linknet,
}


def build_model(
    name: str, encoder: str = "resnet34", encoder_weights: str | None = "imagenet"
) -> torch.nn.Module:
    if name not in ARCHS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(ARCHS)}")
    return ARCHS[name](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
