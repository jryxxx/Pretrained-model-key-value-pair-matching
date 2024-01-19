import torch
from mmdet.models.backbones.van import VAN
from collections import OrderedDict

pretrained_model = "pretrained_model/van/tiny.pth.tar"
save_dir = "pretrained_model/van/tiny_changed.pth.tar"

# 模型网络
model = VAN()
model_state_dict = model.state_dict()
# 预训练权重
weights = torch.load(pretrained_model)
weights_state_dict = weights["state_dict"]
new_ckpt = OrderedDict()
# 键值对替换
for k, v in weights_state_dict.items():
    # print(k)
    if "block" in k:
        new_k = k.replace("block", "blocks")
        if "conv0h" in k:
            new_k = new_k.replace("conv0h", "DW_conv_h")
            new_v = v
        elif "conv0v" in k:
            new_k = new_k.replace("conv0v", "DW_conv_v")
            new_v = v
        elif "conv_spatial_h" in k:
            new_k = new_k.replace("conv_spatial_h", "DW_D_conv_h")
            new_v = v
        elif "conv_spatial_v" in k:
            new_k = new_k.replace("conv_spatial_v", "DW_D_conv_v")
            new_v = v
        elif ".dwconv.dwconv.weight" in k:
            new_k = new_k.replace(".dwconv.dwconv.weight", ".dwconv.weight")
            new_v = v
        elif ".dwconv.dwconv.bias" in k:
            new_k = new_k.replace(".dwconv.dwconv.bias", ".dwconv.bias")
            new_v = v
        else:
            new_k = new_k
            new_v = v
    else:
        if "proj" in k:
            new_k = k.replace("proj", "projection")
            new_v = v
        else:
            new_k = k
            new_v = v
    # print(new_k)
    new_ckpt[new_k] = new_v

# model_keys = list(model_state_dict.keys())
# weights_keys = list(new_ckpt.keys())
# print(len(model_keys), len(weights_keys))
# print("*" * 40)
# for i in range(min(len(model_keys), len(weights_keys))):
#     if model_keys[i] != weights_keys[i]:
#         print(model_keys[i], "****", weights_keys[i])
del new_ckpt["head.weight"]
del new_ckpt["head.bias"]
model.load_state_dict(new_ckpt)
torch.save("pretrained_model/van/tiny_changed.pth.tar", new_ckpt)
