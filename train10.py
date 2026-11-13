import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallCNN
from utils import set_seed, count_parameters, get_flops


def get_loaders(batch_size=128):
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

    train_set = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def train_one_config(config, train_loader, test_loader, device, epochs=10):
    set_seed(42)

    # 关键修复：config 中的 "name" 只用于记录，不传给模型构造函数
    model_kwargs = {k: v for k, v in config.items() if k != "name"}
    model = SmallCNN(**model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    params = count_parameters(model)
    macs = get_flops(model, (1, 1, 28, 28))

    best_acc = 0.0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
        print(f"  epoch {epoch:02d}  test_acc={acc:.4f}")
    elapsed = time.time() - start

    return {
        "model": config["name"],
        "conv_type": config["conv_type"],
        "use_se": config["use_se"],
        "act_type": config["act_type"],
        "norm_type": config["norm_type"],
        "params_M": round(params / 1e6, 4),
        "macs_M": round(macs / 1e6, 4),
        "best_test_acc": round(best_acc, 4),
        "train_time_s": round(elapsed, 1),
    }


def main():
    device = torch.device("cpu")
    train_loader, test_loader = get_loaders()

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
        res = train_one_config(cfg, train_loader, test_loader, device, epochs=10)
        results.append(res)
        print(res)

    df = pd.DataFrame(results)
    df.to_csv("results10.csv", index=False)
    print("\nResults saved to results10.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()