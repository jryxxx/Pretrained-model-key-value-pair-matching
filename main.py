import torch
from models.convnextv2 import ConvNeXtV2


def pretrain_model_changed(
    pretrained_model="weights/convnextv2_tiny_1k_224_ema.pt",
    save_dir="weights/convnextv2_tiny_1k_224_ema_changed.pt",
):
    """
    Args:
        pretrained_model: Pretrained model for training.
        save_dir: Directory for saving.
    """
    # 模型网络
    model = ConvNeXtV2()
    model_state_dict = model.state_dict()
    model_keys = list(model_state_dict.keys())
    # 预训练权重
    weights = torch.load(pretrained_model)
    weights_state_dict = weights["model"]
    weights_keys = list(weights_state_dict.keys())

    assert len(weights_keys) == len(model_keys)
    for i in range(len(model_keys)):
        # 网络模型 keys
        l1 = model_keys[i]
        # 网络模型 weights
        w1 = model_state_dict[l1]
        w2 = weights_state_dict[l1]
        # 导入权重
        w1 = w2
    torch.save(model_state_dict, save_dir)


input = torch.randn(1, 3, 256, 256)
model = ConvNeXtV2()
model.load_state_dict(torch.load("weights/convnextv2_tiny_1k_224_ema_changed.pt"))
pred = model(input)
print(pred.shape)
