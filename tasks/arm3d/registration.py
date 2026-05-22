from __future__ import annotations

from gymnasium.envs.registration import register, registry

SOFT_ARM_3D_ENV_ID = "SoftArmObstacle3D-v0"
SOFT_ARM_3D_ENTRY_POINT = "tasks.arm3d.env_3d:SoftArmObstacle3DEnv"


def register_soft_arm_3d_env() -> str:
    if SOFT_ARM_3D_ENV_ID not in registry:
        register(
            id=SOFT_ARM_3D_ENV_ID,
            entry_point=SOFT_ARM_3D_ENTRY_POINT,
        )
    return SOFT_ARM_3D_ENV_ID


__all__ = ["SOFT_ARM_3D_ENV_ID", "register_soft_arm_3d_env"]
