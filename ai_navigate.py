import os
import sys
import time
import matplotlib.pyplot as plt


import glfw
import mujoco
import numpy as np
import torch

from ppo import PPO_Clip
from env import Go2Env
from agent import ReActAgent

actor_dim=[512, 256, 128]
critic_dim=[512, 256, 128]

env = Go2Env(scene_xml="./unitree_go2/scene_obstacle.xml")
state, _ = env.reset()
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

fpv_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
glfw.init()
window = glfw.create_window(1280, 720, "Go2 Third Person + FPV", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)
option = mujoco.MjvOption()
scene = mujoco.MjvScene(env.model, maxgeom=10000)
context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

third_cam = mujoco.MjvCamera()
third_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
third_cam.distance = 4.
third_cam.azimuth = 0.0
third_cam.elevation = -60.0


fpv_cam = mujoco.MjvCamera()
fpv_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
fpv_cam.fixedcamid = fpv_cam_id


def execute_command(command, dt=1.):
    global state
    state[-3:] = command
    state_th = torch.from_numpy(state)
    state_th = ppo.running_stat.normalize(state_th)[None,:]
    normal, action_th, _ = ppo.pi(state_th)
    action = normal.mean[0].detach().numpy()

    for _ in range(int(dt / env.dt)):
        frame_start = time.time()

        state[-3:] = command
        state_th = torch.from_numpy(state)
        state_th = ppo.running_stat.normalize(state_th)[None,:]
        normal, action_th, _ = ppo.pi(state_th)
        action = normal.mean[0].detach().numpy()
        state, reward, terminated, truncated, info = env.step(action)

        width, height = glfw.get_framebuffer_size(window)
        half_width = width // 2
        left_viewport = mujoco.MjrRect(0, 0, half_width, height)
        right_viewport = mujoco.MjrRect(half_width, 0, width - half_width, height)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0.05, 0.05, 0.05, 1.0)

        third_cam.lookat[:] = env.data.qpos[:3]

        mujoco.mjv_updateScene(
            env.model,
            env.data,
            option,
            None,
            third_cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        mujoco.mjr_render(left_viewport, scene, context)

        mujoco.mjv_updateScene(
            env.model,
            env.data,
            option,
            None,
            fpv_cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        mujoco.mjr_render(right_viewport, scene, context)


        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            left_viewport,
            "Third person",
            "",
            context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            right_viewport,
            "FPV",
            "",
            context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            left_viewport,
            "command",
            f"x {command[0]:.2f}  y {command[1]:.2f}  yaw {command[2]:.2f}",
            context,
        )

        third_rgb = np.empty((height, half_width, 3), dtype=np.uint8)
        third_depth = np.empty((height, half_width), dtype=np.float32)
        mujoco.mjr_readPixels(third_rgb, third_depth, left_viewport, context)
        third_person_image = np.flipud(third_rgb).copy()

        fpv_rgb = np.empty((height, width - half_width, 3), dtype=np.uint8)
        fpv_depth = np.empty((height, width - half_width), dtype=np.float32)
        mujoco.mjr_readPixels(fpv_rgb, fpv_depth, right_viewport, context)
        fpv_image = np.flipud(fpv_rgb).copy()

        glfw.swap_buffers(window)
        glfw.poll_events()

        elapsed = time.time() - frame_start
        if elapsed < env.dt:
            time.sleep(env.dt - elapsed)
    
    
    return fpv_image, third_person_image

try:
    agent = ReActAgent("gpt-5.5")
    # agent = ReActAgent("gpt-5.4")

    fpv_image, third_person_image = execute_command(env.commands["stand"].astype(np.float32).copy(), env.dt)
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(third_person_image)
    # ax2.imshow(fpv_image)
    # plt.show()
    result = agent.run(
        user_input="Navigate to the red ball without hitting the obstacles.",
        fpv_img=fpv_image,
        third_person_img=third_person_image,
        execute_command=execute_command,
    )

    # print(result)

    # while not glfw.window_should_close(window):
    #     execute_command(
    #         np.array([0.0, 0.0, 0.0], dtype=np.float32),
    #         dt=1.,
    #     )

finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
        env.close()
