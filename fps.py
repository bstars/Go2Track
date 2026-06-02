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

actor_dim=[512, 256, 128]
critic_dim=[512, 256, 128]

env = Go2Env()
state, _ = env.reset()
command = env.commands["forward"].astype(np.float32).copy()

def _make_key_callback(command: np.ndarray):
    def key_callback(window, key, scancode, action, mods):
        del scancode, mods
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_UP:
            command[0] += 0.1
        elif key == glfw.KEY_DOWN:
            command[0] -= 0.1
        elif key == glfw.KEY_LEFT:
            command[1] += 0.1
        elif key == glfw.KEY_RIGHT:
            command[1] -= 0.1
        elif key == glfw.KEY_COMMA:
            command[2] += 0.1
        elif key == glfw.KEY_PERIOD:
            command[2] -= 0.1
        elif key == glfw.KEY_SPACE:
            command[:] = env.commands["stand"]
        elif key == glfw.KEY_R:
            command[:] = env.commands["forward"]

        np.clip(command, env.command_lb, env.command_ub, out=command)
        with np.printoptions(precision=2, suppress=True):
            print("command", command)

    return key_callback

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
glfw.set_key_callback(window, _make_key_callback(command))

option = mujoco.MjvOption()
scene = mujoco.MjvScene(env.model, maxgeom=10000)
context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

third_cam = mujoco.MjvCamera()
third_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
third_cam.distance = 5.0
third_cam.azimuth = 90.0
third_cam.elevation = -20.0


fpv_cam = mujoco.MjvCamera()
fpv_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
fpv_cam.fixedcamid = fpv_cam_id




def execute_command(command, dt):
    global state
    state[-3:] = command
    state_th = torch.from_numpy(state)
    state_th = ppo.running_stat.normalize(state_th)[None,:]
    normal, action_th, _ = ppo.pi(state_th)
    # action = action_th[0].detach().numpy()
    action = normal.mean[0].detach().numpy()

    for _ in range(int(dt / env.dt)):
        frame_start = time.time()
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

        rgb = np.empty((height, width - half_width, 3), dtype=np.uint8)
        depth = np.empty((height, width - half_width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, right_viewport, context)
        fpv_image = np.flipud(rgb).copy()

        glfw.swap_buffers(window)
        glfw.poll_events()

        elapsed = time.time() - frame_start
        if elapsed < env.dt:
            time.sleep(env.dt - elapsed)
    
    
    return fpv_image

try:
    while not glfw.window_should_close(window):
        fpv_image = execute_command(command, env.dt)
    

finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
        env.close()