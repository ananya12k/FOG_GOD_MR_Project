"""Print the --lr that makes the training script's fast-forwarded LR equal TARGET at epoch E+1.
Replays exactly what train_fogprompt_exp.py does on resume. Usage: ft_lr.py TARGET E N WARMUP"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_train_inference', 'utils'))
from warmup_scheduler import GradualWarmupScheduler
target, E, N, W = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
lr_min = 1e-7
def after_replay(lr):
    o = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    c = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(1, E + N - W), eta_min=lr_min)
    s = GradualWarmupScheduler(o, multiplier=1, total_epoch=W, after_scheduler=c)
    for p in range(1, E + 1):
        if p >= W: s.step()
    return o.param_groups[0]['lr'], o, s
lo, hi = target, target * 1000
for _ in range(60):
    mid = (lo + hi) / 2
    if after_replay(mid)[0] < target: lo = mid
    else: hi = mid
lr = (lo + hi) / 2
if len(sys.argv) > 5:
    v, o, s = after_replay(lr); t = []
    for ep in range(E + 1, E + N + 1):
        t.append(f"{ep}:{o.param_groups[0]['lr']:.2e}")
        if ep >= W: s.step()
    print(' '.join(t), file=sys.stderr)
print(f'{lr:.6e}')
