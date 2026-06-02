import sys
import os
import mujoco
import time
import glfw
import numpy as np
import torch


from ppo import PPO_Clip
from env import Go2Env


actor_dim=[512, 256, 128]
critic_dim=[512, 256, 128]


def train(ent_coef, steps, from_last=False):
    envs = [Go2Env() for _ in range(256)]
    ppo = PPO_Clip(
        envs, 
        actor_dim=actor_dim,
        critic_dim=critic_dim,
        init_log_std=-0.,
        tanh_transformed=False,
        learning_rate=2e-4,
        n_steps=128,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=1.,
        # target_kl=0.05,
    )

    if from_last:
        ppo.load("./ppo.pth")

    ppo.train(steps, print_every=1)
    ppo.save("./ppo.pth")

def visualize():
    env = Go2Env()
    ppo = PPO_Clip(
        [env], 
        actor_dim=actor_dim,
        critic_dim=critic_dim,
        init_log_std=-0.,
        tanh_transformed=False,
        learning_rate=2e-4,
        n_steps=128,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.,
        vf_coef=0.5,
        max_grad_norm=1.,
        # target_kl=0.05,
    )

    ppo.load("./ppo.pth")

    command = env.commands["forward"]
    # def key_callback(keycode: int):
    #     nonlocal command
    #     if keycode == glfw.KEY_UP: command = env.commands["forward"]
    #     elif keycode == glfw.KEY_DOWN: command = env.commands["backward"]
    #     elif keycode == glfw.KEY_LEFT: command = env.commands["left"]
    #     elif keycode == glfw.KEY_RIGHT: command = env.commands["right"]
    #     elif keycode == ord(','): command = env.commands["left_turn"]
    #     elif keycode == ord('.'): command = env.commands["right_turn"]
    #     elif keycode == glfw.KEY_SPACE: command = env.commands["stand"]

    def key_callback(keycode: int):
        nonlocal command
        if keycode == glfw.KEY_UP: command[0] += 0.1
        elif keycode == glfw.KEY_DOWN: command[0] -= 0.1
        elif keycode == glfw.KEY_LEFT: command[1] += 0.1
        elif keycode == glfw.KEY_RIGHT: command[1] -= 0.1
        elif keycode == ord(','): command[2] += 0.1
        elif keycode == ord('.'): command[2] -= 0.1
        elif keycode == glfw.KEY_SPACE: command = env.commands["stand"]
        with np.printoptions(precision=2, suppress=True):
            print(command)

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        state, _ = env.reset()
        reward_total = 0.
        while viewer.is_running():
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0

            state[-3:] = command

            state_th = torch.from_numpy(state)
            state_th = ppo.running_stat.normalize(state_th)[None,:]
            normal, action_th, _ = ppo.pi(state_th)
            # action = action_th[0].detach().numpy()
            action = normal.mean[0].detach().numpy()



            state, reward, terminated, truncated, info = env.step(action)

            reward_total += reward

            if terminated:
                break


            viewer.cam.lookat[:] = env.data.qpos[0:3]
            # viewer.cam.distance = 3.0
            # viewer.cam.azimuth = 90
            # viewer.cam.elevation = -20
            
            viewer.sync()
            time.sleep(env.dt)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == "train":
        # train(ent_coef=5e-3, steps=5e6, from_last=False) 
        # train(ent_coef=5e-3, steps=5e6, from_last=True)
        # train(ent_coef=1e-3, steps=5e6, from_last=True)
        train(ent_coef=1e-4, steps=5e6, from_last=True)





    elif mode == "test":
        visualize()
