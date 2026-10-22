"""Architectures that torchvision already implements well. Trained from scratch (``weights=None``)."""
from functools import partial

from torchvision import models as tvm

from .registry import register_builder


def _tv_builder(fn, **fixed_kwargs):
    def build(num_classes=1000, img_size=224):
        return fn(weights=None, num_classes=num_classes, **fixed_kwargs)

    build.__name__ = fn.__name__
    return build


def _tv_builder_with_size(fn, size_kwarg, as_tuple=False):
    def build(num_classes=1000, img_size=224):
        size = (img_size, img_size) if as_tuple else img_size
        return fn(weights=None, num_classes=num_classes, **{size_kwarg: size})

    build.__name__ = fn.__name__
    return build


def _register_family(names, category, family, paper, builder_factory=_tv_builder, aliases=None, **spec_kwargs):
    aliases = aliases or {}
    for name in names:
        register_builder(
            name,
            builder_factory(getattr(tvm, name)),
            category=category,
            family=family,
            paper=paper,
            aliases=aliases.get(name, ()),
            **spec_kwargs,
        )


cnn = partial(_register_family, category="cnn")
lightweight = partial(_register_family, category="lightweight")
transformer = partial(_register_family, category="transformer")

# --- CNN & Residual ------------------------------------------------------------------------
cnn(["alexnet"], family="AlexNet", paper="Krizhevsky et al., 2012", min_input_size=64)
cnn(
    ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "vgg16"],
    family="VGG",
    paper="Simonyan & Zisserman, 2014",
)
# Auxiliary classifiers would make forward() return a tuple during training, so they are disabled.
_tv_builder_no_aux = partial(_tv_builder, aux_logits=False, init_weights=True)
cnn(["googlenet"], family="GoogLeNet (Inception v1)", paper="Szegedy et al., 2015", builder_factory=_tv_builder_no_aux)
cnn(
    ["inception_v3"],
    family="Inception v3",
    paper="Szegedy et al., 2016",
    builder_factory=_tv_builder_no_aux,
    min_input_size=75,
)
cnn(
    ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    family="ResNet",
    paper="He et al., 2016",
)
cnn(["resnext50_32x4d", "resnext101_32x8d"], family="ResNeXt", paper="Xie et al., 2017")
cnn(["wide_resnet50_2", "wide_resnet101_2"], family="Wide ResNet", paper="Zagoruyko & Komodakis, 2016")
cnn(
    ["regnet_x_400mf", "regnet_x_800mf", "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf"],
    family="RegNet",
    paper="Radosavovic et al., 2020",
)
cnn(["convnext_tiny", "convnext_small"], family="ConvNeXt", paper="Liu et al., 2022")
cnn(["densenet121", "densenet169", "densenet201"], family="DenseNet", paper="Huang et al., 2017")

# --- Lightweight & Channel Attention -------------------------------------------------------
lightweight(
    ["mobilenet_v2"],
    family="MobileNetV2",
    paper="Sandler et al., 2018",
    aliases={"mobilenet_v2": ("mobilenet",)},
)
lightweight(["mobilenet_v3_small", "mobilenet_v3_large"], family="MobileNetV3", paper="Howard et al., 2019")
lightweight(
    ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x2_0"],
    family="ShuffleNetV2",
    paper="Ma et al., 2018",
)
lightweight(["squeezenet1_0", "squeezenet1_1"], family="SqueezeNet", paper="Iandola et al., 2016")
lightweight(["mnasnet0_5", "mnasnet1_0"], family="MnasNet", paper="Tan et al., 2019")
lightweight(
    ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"],
    family="EfficientNet",
    paper="Tan & Le, 2019",
)
lightweight(["efficientnet_v2_s"], family="EfficientNetV2", paper="Tan & Le, 2021")

# --- Transformer ---------------------------------------------------------------------------
transformer(
    ["vit_b_16", "vit_b_32"],
    family="ViT",
    paper="Dosovitskiy et al., 2021",
    builder_factory=lambda fn: _tv_builder_with_size(fn, "image_size"),
    min_input_size=32,
    input_size_divisor=32,
)
transformer(["swin_t", "swin_s"], family="Swin Transformer", paper="Liu et al., 2021")
transformer(["swin_v2_t", "swin_v2_s"], family="Swin Transformer V2", paper="Liu et al., 2022")
transformer(
    ["maxvit_t"],
    family="MaxViT",
    paper="Tu et al., 2022",
    builder_factory=lambda fn: _tv_builder_with_size(fn, "input_size", as_tuple=True),
    min_input_size=224,
    input_size_divisor=224,
)
