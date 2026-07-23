from torchvision import models as torchvision_models

from .mobilevit import ExMobileViT, MobileViT

MODEL_FACTORY = {
    "resnet18": torchvision_models.resnet18,
    "resnet34": torchvision_models.resnet34,
    "resnet50": torchvision_models.resnet50,
    "resnet101": torchvision_models.resnet101,
    "resnet152": torchvision_models.resnet152,
    "densenet121": torchvision_models.densenet121,
    "densenet169": torchvision_models.densenet169,
    "densenet201": torchvision_models.densenet201,
    "mobilenet_v2": torchvision_models.mobilenet_v2,
    "mobilenetv2": torchvision_models.mobilenet_v2,
    "mobilenet": torchvision_models.mobilenet_v2,
    "mobilevit": MobileViT,
    "exmobilevit": ExMobileViT,
}


def create_model(opts, num_classes: int):
    model_name = str(getattr(opts, "model", "resnet18")).lower().replace("_", "")
    if model_name not in MODEL_FACTORY:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {', '.join(sorted(MODEL_FACTORY.keys()))}"
        )

    model_fn = MODEL_FACTORY[model_name]
    if model_name in ["mobilevit", "exmobilevit"]:
        return model_fn(num_classes=num_classes)

    return model_fn(pretrained=False, num_classes=num_classes)
