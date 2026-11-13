import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 中文字体（Windows 用 SimHei，缺失时自动回退） ----------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("figures", exist_ok=True)

# ---------- 读取结果 ----------
df_main = pd.read_csv("results20.csv")
df_scen = pd.read_csv("scenario_results.csv")

SHORT = {
    "Baseline (Standard Conv + BN + ReLU)": "Baseline",
    "Depthwise Separable": "DW",
    "SE Attention": "SE",
    "GELU Activation": "GELU",
    "GroupNorm": "GN",
}
df_main["short"] = df_main["model"].map(SHORT)
df_scen["short"] = df_scen["model"].map(SHORT)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
model_order = ["Baseline", "DW", "SE", "GELU", "GN"]
df_main = df_main.set_index("short").loc[model_order].reset_index()
df_scen = df_scen.set_index("short").loc[model_order].reset_index()


# =====================================================================
# 图 1：主表对比（Params / MACs / Test Acc 三联柱状图）
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1a Params
ax = axes[0]
vals = df_main["params_M"].values
bars = ax.bar(df_main["short"], vals, color=COLORS)
ax.set_title("参数量 Params (M)")
ax.set_ylabel("Params (M)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(vals) * 1.18)

# 1b MACs
ax = axes[1]
vals = df_main["macs_M"].values
bars = ax.bar(df_main["short"], vals, color=COLORS)
ax.set_title("计算量 MACs (M)")
ax.set_ylabel("MACs (M)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(vals) * 1.18)

# 1c Test Acc
ax = axes[2]
vals = df_main["test_acc_at_best_val"].values * 100
bars = ax.bar(df_main["short"], vals, color=COLORS)
ax.set_title("测试准确率 Test Acc (%)")
ax.set_ylabel("Accuracy (%)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(min(vals) - 3, max(vals) + 1.5)

fig.suptitle("5 组消融实验：参数量 / 计算量 / 测试准确率", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig1_main_comparison.png", dpi=200)
plt.close(fig)


# =====================================================================
# 图 2：场景鲁棒性——绝对准确率分组柱状图
# =====================================================================
scen_cols = [c for c in df_scen.columns if c not in ("model", "short")]
n_models = len(df_scen)
n_scen = len(scen_cols)
x = np.arange(n_scen)
width = 0.8 / n_models

fig, ax = plt.subplots(figsize=(16, 6))
for i, (_, row) in enumerate(df_scen.iterrows()):
    offset = (i - (n_models - 1) / 2) * width
    ax.bar(x + offset, row[scen_cols].values * 100, width,
           label=row["short"], color=COLORS[i])

ax.set_xticks(x)
ax.set_xticklabels(scen_cols, rotation=30, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_title("退化场景下的绝对准确率（10 种场景 × 5 个模型）")
ax.legend(ncol=5, loc="upper right")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig2_scenario_absolute.png", dpi=200)
plt.close(fig)


# =====================================================================
# 图 3：场景鲁棒性——相对 clean 的掉点热力图
# =====================================================================
# ΔAcc = 各场景 acc - clean acc，单位：百分点（负值表示掉点）
clean_col = "clean_0"
perturb_cols = [c for c in scen_cols if c != clean_col]
delta = df_scen[perturb_cols].sub(df_scen[clean_col], axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
data = delta.values
im = ax.imshow(data, cmap="RdYlGn", vmin=-80, vmax=0, aspect="auto")

ax.set_xticks(range(len(perturb_cols)))
ax.set_xticklabels(perturb_cols, rotation=30, ha="right")
ax.set_yticks(range(len(df_scen)))
ax.set_yticklabels(df_scen["short"])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                fontsize=8, color="black")

fig.colorbar(im, ax=ax, label="ΔAcc vs clean（百分点，越接近 0 越鲁棒）")
ax.set_title("相对 clean 的准确率掉点（绿色=鲁棒，红色=掉点严重）")
fig.tight_layout()
fig.savefig("figures/fig3_scenario_delta_heatmap.png", dpi=200)
plt.close(fig)


# =====================================================================
# 图 4：10 epoch vs 20 epoch 的测试准确率对比
#    10 epoch 数字来自旧 results_10ep.csv（代码首次运行结果）
# =====================================================================
acc_10ep = {
    "Baseline": 0.8841,
    "DW": 0.8350,
    "SE": 0.8846,
    "GELU": 0.8899,
    "GN": 0.8701,
}
acc_20ep = dict(zip(df_main["short"], df_main["test_acc_at_best_val"]))

x = np.arange(len(model_order))
w = 0.35
v10 = [acc_10ep[m] * 100 for m in model_order]
v20 = [acc_20ep[m] * 100 for m in model_order]

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w / 2, v10, w, label="10 epoch", color="#B0B0B0")
b2 = ax.bar(x + w / 2, v20, w, label="20 epoch + CosineLR + wd", color="#4C72B0")

for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(model_order)
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("延长训练 + 学习率调度 + 权重衰减带来的准确率变化")
ax.legend()
ax.set_ylim(min(v10 + v20) - 2, max(v10 + v20) + 2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig4_10ep_vs_20ep.png", dpi=200)
plt.close(fig)


print("所有图已保存到 figures/ 目录：")
for f in sorted(os.listdir("figures")):
    print("  -", f)