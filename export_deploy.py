"""Slim the checkpoint for deployment: fp16 weights, drop training history.
162 MB -> ~85 MB, which keeps it under GitHub's 100 MB per-file limit (no Git LFS).
Precision loss is irrelevant for a demonstrator; weights are cast back to fp32 on load.
"""
import torch, os
ROOT = os.path.dirname(os.path.abspath(__file__))
src = torch.load(os.path.join(ROOT, "out", "model.pt"), map_location="cpu", weights_only=False)
slim = {
    "state": {k: (v.half() if v.is_floating_point() else v) for k, v in src["state"].items()},
    "scaler": src["scaler"], "terciles": src["terciles"],
    "val_acc": src["val_acc"], "test_acc": src["test_acc"],
    "baseline_val": src["baseline_val"], "baseline_test": src["baseline_test"],
    "embed_model": src["embed_model"], "epochs": src.get("stopped_at"),
}
dst = os.path.join(ROOT, "model_deploy.pt")
torch.save(slim, dst)
print(f"  {os.path.getsize(os.path.join(ROOT,'out','model.pt'))/1e6:6.1f} MB -> "
      f"{os.path.getsize(dst)/1e6:6.1f} MB   {dst}")
