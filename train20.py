import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallCNN
from utils import set_seed, count_parameters, get_flops


def get_loaders(batch_size=128, val_ratio=0.1):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    full_train = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=test_transform)

    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    # 固定切分，保证可复现
    train_sub, val_sub = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    # 验证集不增强：用同一批索引，但用 test_transform 重新取图
    val_indices = val_sub.indices
    val_set = torch.utils.data.Subset(
        datasets.FashionMNIST(root="./data", train=True,
                              download=False, transform=test_transform),
        val_indices)

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def ckpt_name(model_name):
    return "checkpoints/" + (model_name.replace(" ", "_")
                             .replace("(", "").replace(")", "")
                             .replace("+", "p")) + ".pt"


def train_one_config(config, train_loader, val_loader, test_loader,
                     device, epochs=20):
    set_seed(42)
    model_kwargs = {k: v for k, v in config.items() if k != "name"}
    model = SmallCNN(**model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    params = count_parameters(model)
    macs = get_flops(model, (1, 1, 28, 28))

    best_val_acc = 0.0
    best_state = None
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)   # 用验证集选模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:02d}  val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
    elapsed = time.time() - start

    # 用验证集选出的 best 权重，在 test 上只评估一次
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, ckpt_name(config["name"]))

    return {
        "model": config["name"],
        "conv_type": config["conv_type"],
        "use_se": config["use_se"],
        "act_type": config["act_type"],
        "norm_type": config["norm_type"],
        "params_M": round(params / 1e6, 4),
        "macs_M": round(macs / 1e6, 4),
        "best_val_acc": round(best_val_acc, 4),
        "test_acc_at_best_val": round(test_acc, 4),
        "train_time_s": round(elapsed, 1),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader, test_loader = get_loaders()

    configs = [
        dict(name="Baseline (Standard Conv + BN + ReLU)",
             conv_type="standard", use_se=False, act_type="relu", norm_type="bn"),
        dict(name="Depthwise Separable",
             conv_type="dw", use_se=False, act_type="relu", norm_type="bn"),
        dict(name="SE Attention",
             conv_type="standard", use_se=True, act_type="relu", norm_type="bn"),
        dict(name="GELU Activation",
             conv_type="standard", use_se=False, act_type="gelu", norm_type="bn"),
        dict(name="GroupNorm",
             conv_type="standard", use_se=False, act_type="relu", norm_type="gn"),
    ]

    results = []
    for cfg in configs:
        print(f"\n===== Training: {cfg['name']} =====")
        res = train_one_config(cfg, train_loader, val_loader, test_loader,
                               device, epochs=20)
        results.append(res)
        print(res)

    df = pd.DataFrame(results)
    df.to_csv("results20.csv", index=False)
    print("\nResults saved to results20.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()