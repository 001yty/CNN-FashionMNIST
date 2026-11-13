import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flops(model, input_size=(1, 1, 28, 28)):
    """
    用 thop 统计 MACs（乘加次数）。
    MACs ≈ FLOPs / 2，报告中统一标注为 MACs 即可。
    如果 thop 不可用，返回 0 并在结果里标注。
    """
    try:
        from thop import profile
        model.eval()
        dummy = torch.randn(*input_size)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        return macs
    except Exception as e:
        print(f"[warn] thop profile failed: {e}")
        return 0