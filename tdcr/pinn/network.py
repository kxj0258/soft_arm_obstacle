from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch is optional for simulator-only use.
    torch = None
    nn = None


@dataclass(frozen=True)
class TDCRPINNConfig:
    tau_dim: int = 9
    contact_dim: int = 4 * 24
    obstacle_dim: int = 8 * 10
    robot_param_dim: int = 10
    hidden_dim: int = 128
    hidden_layers: int = 4
    output_dim: int = 15  # p(3), rotvec(3), u(3), v(3), contact force(3)


if nn is not None:

    class ContactAwareTDCRPINN(nn.Module):
        """Small contact-conditioned PINN head for 3-segment TDCR samples.

        Inputs are arclength ``s`` plus per-sample context: tendon tensions,
        contact slots, obstacle slots, and normalized robot parameters.
        """

        def __init__(self, config: TDCRPINNConfig | None = None):
            super().__init__()
            self.config = config or TDCRPINNConfig()
            input_dim = (
                1
                + int(self.config.tau_dim)
                + int(self.config.contact_dim)
                + int(self.config.obstacle_dim)
                + int(self.config.robot_param_dim)
            )
            layers: list[nn.Module] = []
            current = input_dim
            for _ in range(int(self.config.hidden_layers)):
                layers.append(nn.Linear(current, int(self.config.hidden_dim)))
                layers.append(nn.SiLU())
                current = int(self.config.hidden_dim)
            layers.append(nn.Linear(current, int(self.config.output_dim)))
            self.net = nn.Sequential(*layers)

        def forward(self, s, tau, contact_slots, obstacle_slots, robot_params):
            if s.ndim == 1:
                s = s[:, None]
            if s.ndim == 2 and tau.ndim == 2 and s.shape[0] == tau.shape[0]:
                context = torch.cat(
                    [
                        tau.reshape(tau.shape[0], -1),
                        contact_slots.reshape(contact_slots.shape[0], -1),
                        obstacle_slots.reshape(obstacle_slots.shape[0], -1),
                        robot_params.reshape(robot_params.shape[0], -1),
                    ],
                    dim=-1,
                )
                features = torch.cat([s, context], dim=-1)
                return self.net(features)

            if s.ndim != 3:
                raise ValueError("s must have shape (batch, points, 1) for batched sequence prediction.")
            batch, points, _ = s.shape
            context = torch.cat(
                [
                    tau.reshape(batch, -1),
                    contact_slots.reshape(batch, -1),
                    obstacle_slots.reshape(batch, -1),
                    robot_params.reshape(batch, -1),
                ],
                dim=-1,
            )
            tiled_context = context[:, None, :].expand(batch, points, context.shape[-1])
            features = torch.cat([s, tiled_context], dim=-1)
            return self.net(features.reshape(batch * points, -1)).reshape(batch, points, -1)

else:

    class ContactAwareTDCRPINN:  # pragma: no cover - exercised only when torch is absent.
        def __init__(self, *args, **kwargs):
            raise ImportError("ContactAwareTDCRPINN requires PyTorch.")


__all__ = ["ContactAwareTDCRPINN", "TDCRPINNConfig"]
