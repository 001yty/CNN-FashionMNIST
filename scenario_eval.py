import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallCNN
from utils import set_seed


CONFIGS = {
    "Baseline (Standard Conv + BN + ReLU)":
        dict(conv_type="standard", use_se=False, act_type="relu", norm_type="bn"),
    "Depthwise Separable":
        dict(conv_type="dw", use_se=False, act_type="relu", norm_type="bn"),
    "SE Attention":
        dict(conv_type="standard", use_se=True, act_type="relu", norm_type="bn"),
    "GELU Activation":
        dict(conv_type="standard", use_se=False, act_type="gelu", norm_type="bn"),
    "GroupNorm":
        dict(conv_type="standard", use_se=False, act_type="relu", norm_type="gn"),
}


def ckpt_name(model_name):
    return "checkpoints/" + (model_name.replace(" ", "_")
                             .replace("(", "").replace(")", "")
                             .replace("+", "p")) + ".pt"


def load_checkpoint(model, path, device):
    """
    加载 checkpoint，兼容：
    1. thop.profile 注入的 total_ops / total_params 字段；
    2. PyTorch 2.6+ 的 weights_only 默认值变化。
    """
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # 旧版本 PyTorch 没有 weights_only 参数
        state = torch.load(path, map_location=device)

    # 过滤 thop 注入的无关键
    state = {k: v for k, v in state.items()
             if not (k.endswith("total_ops") or k.endswith("total_params"))}

    model.load_state_dict(state)
    return model


def perturb(x, kind, level):
    """对已 Normalize 的 tensor 施加扰动。"""
    if kind == "clean":
        return x
    if kind == "gaussian":
        return x + torch.randn_like(x) * level
    if kind == "blur":
        k = level
        kernel = torch.ones(1, 1, k, k, device=x.device) / (k * k)
        kernel = kernel.expand(x.size(1), 1, k, k)
        return F.conv2d(x, kernel, padding=k // 2, groups=x.size(1))
    if kind == "occlusion":
        # level 表示被遮挡的概率，0~1
        mask = torch.rand_like(x) > level
        return x * mask.float()
    if kind == "brightness":
        return x * level
    raise ValueError(kind)


@torch.no_grad()
def eval_perturbed(model, loader, device, kind, level):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = perturb(x, kind, level)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]))
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=0)

    scenarios = [
        ("clean", 0),
        ("gaussian", 0.1), ("gaussian", 0.3), ("gaussian", 0.5),
        ("blur", 3), ("blur", 5),
        ("occlusion", 0.3), ("occlusion", 0.5),
        ("brightness", 0.5), ("brightness", 1.5),
    ]

    rows = []
    for name, cfg in CONFIGS.items():
        model = SmallCNN(**cfg).to(device)
        load_checkpoint(model, ckpt_name(name), device)

        row = {"model": name}
        for kind, level in scenarios:
            acc = eval_perturbed(model, test_loader, device, kind, level)
            col = f"{kind}_{level}"
            row[col] = round(acc, 4)
            print(f"{name:45s} {col:18s} acc={acc:.4f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("scenario_results.csv", index=False)
    print("\nSaved to scenario_results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()