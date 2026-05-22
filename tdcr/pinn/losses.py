from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is optional for simulator-only use.
    torch = None
    F = None


def _require_torch():
    if torch is None or F is None:
        raise ImportError("tdcr.pinn losses require PyTorch.")


def supervised_shape_loss(prediction, target_p, target_u, *, position_weight: float = 1.0, strain_weight: float = 0.1):
    """Supervised loss for the default network output layout.

    ``prediction[..., :3]`` is position and ``prediction[..., 6:9]`` is curvature strain ``u``.
    """

    _require_torch()
    position_loss = F.mse_loss(prediction[..., :3], target_p)
    strain_loss = F.mse_loss(prediction[..., 6:9], target_u)
    return float(position_weight) * position_loss + float(strain_weight) * strain_loss


def contact_clearance_loss(prediction, obstacle_slots, arm_radius: float = 0.004, margin: float = 0.0):
    """Soft collision loss against active spherical/ellipsoidal obstacle slots."""

    _require_torch()
    points = prediction[..., :3]
    slots = obstacle_slots
    active = slots[..., 0] > 0.5
    if not torch.any(active):
        return points.sum() * 0.0
    centers = slots[..., 2:5]
    radii = torch.clamp(slots[..., 5:8], min=1e-6)
    residuals = []
    for obstacle_index in range(slots.shape[-2]):
        mask = active[..., obstacle_index]
        if not torch.any(mask):
            continue
        center = centers[..., obstacle_index, :].unsqueeze(-2)
        radius = radii[..., obstacle_index, :].unsqueeze(-2) + float(arm_radius) + float(margin)
        scaled = (points - center) / radius
        clearance = torch.linalg.norm(scaled, dim=-1) - 1.0
        residuals.append(F.relu(-clearance) ** 2)
    if not residuals:
        return points.sum() * 0.0
    return torch.stack([item.mean() for item in residuals]).mean()


__all__ = ["contact_clearance_loss", "supervised_shape_loss"]
