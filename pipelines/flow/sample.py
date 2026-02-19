import torch

@torch.no_grad()
def euler_sample(
    model, cond: torch.Tensor, txt_cond: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None,
    adaln_conds: torch.Tensor | None = None, max_t: int = 250, model_pred: str = "v"
):
    x = torch.randn((cond.shape[0], 3, cond.shape[2], cond.shape[3]), device=cond.device)
    dt = 1 / max_t

    for i in range(max_t):
        t = torch.tensor([i * dt], dtype=torch.float32, device=cond.device).repeat(cond.shape[0])

        model_out = model(x, t, cond, txt_cond, key_padding_mask, adaln_conds)
        
        if model_pred == "x":
            if i == max_t - 1:
                return model_out
            
            denom = (torch.ones_like(t) - t.repeat(-1, 1, 1, 1)).clamp(min=0.05)
            vel = (model_out - x) / denom
        else:
            vel = model_out

        x = x + vel * dt

    return x