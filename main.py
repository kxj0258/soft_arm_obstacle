# 在 main.py 里加
import time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["2d", "3d", "2d_py", "3d_py"], default="2d_py")
    args = parser.parse_args()

    if args.task == "2d_py":
        from tasks.arm2d.env_2d import SoftArmObstacle2DPyEnv as Env
    elif args.task == "3d_py":
        from tasks.arm3d.env_3d import SoftArmObstacle3DPyEnv as Env

    env = Env(render_mode="human")
    obs, _ = env.reset()
    for _ in range(300):
        obs, rew, *_ = env.step(env.action_space.sample())
        time.sleep(0.5)
        # print(rew)