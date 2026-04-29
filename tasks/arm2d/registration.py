from __future__ import annotations

from gymnasium.envs.registration import register, registry

SOFT_ARM_2D_ENV_ID = "SoftArmObstacle2D-v0"
SOFT_ARM_2D_ENTRY_POINT = "tasks.arm2d.env_2d:SoftArmObstacle2DEnv"


def register_soft_arm_2d_env() -> str:
    if SOFT_ARM_2D_ENV_ID not in registry:
        register(
            id=SOFT_ARM_2D_ENV_ID,
            entry_point=SOFT_ARM_2D_ENTRY_POINT,
        )
    return SOFT_ARM_2D_ENV_ID
