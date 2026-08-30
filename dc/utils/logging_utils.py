from typing import Optional

import torch


def make_wandb_image(image_tensor, tensor_to_pil_fn, resize_image_fn, caption: str, min_size: int = 256):
    """Convert a tensor image or mask into a compact WandB image."""
    import wandb

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.detach().float().cpu().clamp(0.0, 1.0)
    return wandb.Image(
        resize_image_fn(tensor_to_pil_fn(image_tensor), min_size=min_size),
        caption=caption,
    )


def summarize_mask(mask: torch.Tensor):
    mask = mask.detach().float()
    return {
        "mean": mask.mean().item(),
        "max": mask.max().item(),
        "coverage_0.5": (mask > 0.5).float().mean().item(),
    }


def batch_l2_norm_mean(tensor: torch.Tensor) -> float:
    """Return the mean L2 norm across the batch for arbitrary-shaped tensors."""
    return tensor.detach().float().flatten(1).norm(dim=1).mean().item()


def log_dc_debug_to_wandb(
    *,
    step: int,
    current_spot: int,
    t: torch.Tensor,
    t_normalized: float,
    eta_tag_current: float,
    current_stg_scale: float,
    current_edit_strength: Optional[float],
    w_dds: float,
    preserve_weight,
    eps_tgt: torch.Tensor,
    eps_src: torch.Tensor,
    eps_tgt_for_grad: torch.Tensor,
    grad: torch.Tensor,
    pred_x0_tgt: torch.Tensor,
    pred_x0_src: torch.Tensor,
    grad_mask: Optional[torch.Tensor],
    self_grad_mask: Optional[torch.Tensor],
    cross_attention_mask: Optional[torch.Tensor],
    cross_attention_mask_weight_current: Optional[float],
    tensor_to_pil_fn,
    resize_image_fn,
):
    """Log the most useful DC debugging images and scalars to WandB."""
    import wandb

    caption = (
        f"step={step} | spot={current_spot} | "
        f"t={t.item()} | t_norm={t_normalized:.3f}"
    )
    raw_eps_delta = batch_l2_norm_mean(eps_tgt - eps_src)
    effective_eps_delta = batch_l2_norm_mean(eps_tgt_for_grad - eps_src)
    grad_norm = batch_l2_norm_mean(grad)
    preserve_weight_mean = (
        preserve_weight.mean().item()
        if isinstance(preserve_weight, torch.Tensor)
        else float(preserve_weight)
    )

    log_payload = {
        "dc_debug/tgt_pred_x0": make_wandb_image(
            pred_x0_tgt,
            tensor_to_pil_fn,
            resize_image_fn,
            caption,
        ),
        "dc_debug/src_pred_x0": make_wandb_image(
            pred_x0_src,
            tensor_to_pil_fn,
            resize_image_fn,
            caption,
        ),
        "dc_debug/current_spot": int(current_spot),
        "dc_debug/timestep": float(t.item()),
        "dc_debug/timestep_normalized": float(t_normalized),
        "dc_debug/eta_tag_current": float(eta_tag_current),
        "dc_debug/stg_scale_current": float(current_stg_scale),
        "dc_debug/w_dds": float(w_dds),
        "dc_debug/preserve_weight_mean": preserve_weight_mean,
        "dc_debug/raw_eps_delta_norm": raw_eps_delta,
        "dc_debug/effective_eps_delta_norm": effective_eps_delta,
        "dc_debug/grad_norm": grad_norm,
    }
    if current_edit_strength is not None:
        log_payload["dc_debug/edit_strength"] = float(current_edit_strength)
    if cross_attention_mask_weight_current is not None:
        log_payload["dc_debug/cross_attention_mask_weight_current"] = float(
            cross_attention_mask_weight_current
        )

    if self_grad_mask is not None:
        self_mask_stats = summarize_mask(self_grad_mask)
        log_payload["dc_debug/self_mask"] = make_wandb_image(
            self_grad_mask,
            tensor_to_pil_fn,
            resize_image_fn,
            caption,
        )
        log_payload["dc_debug/self_mask_mean"] = self_mask_stats["mean"]
        log_payload["dc_debug/self_mask_max"] = self_mask_stats["max"]
        log_payload["dc_debug/self_mask_coverage_0.5"] = self_mask_stats["coverage_0.5"]

    if cross_attention_mask is not None:
        cross_mask_stats = summarize_mask(cross_attention_mask)
        log_payload["dc_debug/cross_attention_mask"] = make_wandb_image(
            cross_attention_mask,
            tensor_to_pil_fn,
            resize_image_fn,
            caption,
        )
        log_payload["dc_debug/cross_attention_mask_mean"] = cross_mask_stats["mean"]
        log_payload["dc_debug/cross_attention_mask_max"] = cross_mask_stats["max"]
        log_payload["dc_debug/cross_attention_mask_coverage_0.5"] = cross_mask_stats["coverage_0.5"]

    if grad_mask is not None:
        grad_mask_stats = summarize_mask(grad_mask)
        log_payload["dc_debug/final_mask"] = make_wandb_image(
            grad_mask,
            tensor_to_pil_fn,
            resize_image_fn,
            caption,
        )
        log_payload["dc_debug/final_mask_mean"] = grad_mask_stats["mean"]
        log_payload["dc_debug/final_mask_max"] = grad_mask_stats["max"]
        log_payload["dc_debug/final_mask_coverage_0.5"] = grad_mask_stats["coverage_0.5"]

    wandb.log(log_payload, step=step, commit=False)
