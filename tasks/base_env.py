# tasks/base_env.py
import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np
import mujoco


class BaseSoftArmEnv(gym.Env, ABC):
    """
    2D/3D 连续体障碍环境的公共基类
    子类必须实现：
        observation_space, action_space
        _get_obs()  -> np.ndarray
        _apply_action(action) -> None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 125}

    def __init__(self, xml_file: str, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self._mujoco_model = mujoco.MjModel.from_xml_path(xml_file)
        self._mujoco_data  = mujoco.MjData(self._mujoco_model)

        # 子类负责定义空间
        self.observation_space = None
        self.action_space      = None

        # 随机障碍/目标生成器（可 override）
        self._obstacle_radius = 0.10
        self._target_radius   = 0.04

    # ---------------- 公共工具 ---------------- #
    def _random_circle(self, low, high, r_min=0.15):
        """在 low..high 矩形内随机采样，且离原点 > r_min"""
        while True:
            pos = np.random.uniform(low, high)
            if np.linalg.norm(pos) > r_min:
                return pos

    def _regenerate_obstacles_target(self, n_obst=3, x_range=(0.3, 1.2), y_range=(-0.4, 0.4)):
        """把 worldbody 里名字是 obs* 和 target 的 body 移动到新位置"""
        for i in range(n_obst):
            name = f"obs{i+1}"
            bid = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                new_pos = self._random_circle([x_range[0], y_range[0]],
                                              [x_range[1], y_range[1]])
                self._mujoco_model.body_pos[bid][:2] = new_pos

        tid = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if tid != -1:
            tgt = self._random_circle([x_range[0], y_range[0]],
                                      [x_range[1], y_range[1]])
            self._mujoco_model.body_pos[tid][:2] = tgt
        self.target_pos = self._mujoco_model.body_pos[tid][:3]

    # ---------------- 子类必须实现 ---------------- #
    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """返回当前观测向量"""
        raise NotImplementedError

    @abstractmethod
    def _apply_action(self, action: np.ndarray):
        """把 action 施加到仿真"""
        raise NotImplementedError

    # ---------------- 标准 gym 接口 ---------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._mujoco_model, self._mujoco_data)
        self._regenerate_obstacles_target()
        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self._mujoco_model, self._mujoco_data)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = truncated = False
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs):
        # 子类可 override；默认用末端-目标距离
        ee_pos = obs[:3] if obs.size >= 3 else obs[:2]
        return -np.linalg.norm(ee_pos - self.target_pos[:len(ee_pos)])

    # ---------------- 渲染 ---------------- #
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # human 模式由 MuJoCo GUI 自动处理，无需额外代码
        return None

    def _render_frame(self):
        """返回 480×640×3 RGB 数组"""
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self._mujoco_model, 480, 640)
        self._renderer.update_scene(self._mujoco_data)
        return self._renderer.render()

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()