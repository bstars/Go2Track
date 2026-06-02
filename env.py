import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import Env, spaces
import time


def quat_to_rotmat(q: np.ndarray):
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)

def quat_to_euler(q):
    """
    Convert quaternion (w, x, y, z) to Euler angles (yaw, pitch, roll)
    using ZYX convention.

    Returns:
        yaw, pitch, roll  (in radians)
    """
    w, x, y, z = q

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)  # numerical safety
    pitch = np.arcsin(sinp)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    return yaw, pitch, roll

class Go2Env(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        # mujoco model

        self.model = mujoco.MjModel.from_xml_path("./unitree_go2/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        self.home_qpos = self.model.key_qpos[0].copy()
        self.home_ctrl = self.model.key_ctrl[0].copy()
        self.default_joint_pos = self.home_qpos[7:].copy()
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.feet_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("FL", "FR", "RL", "RR")
        ]
        self.thigh_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")
        ]
        self.calf_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
        ]

        self.bad_collision_body_ids = set(self.thigh_body_ids + self.calf_body_ids + [self.base_body_id])
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


        # 
        # Hyperparameters
        #
        self.dt = 0.02 # 50 Hz
        self.sim_step = int(self.dt / self.model.opt.timestep)
        self.episode_second = 20.
        self.resample_second = 4.
        self.episode_step = int(self.episode_second / self.dt)
        self.resample_step = int(self.resample_second / self.dt)
        self.action_scale = np.array([0.25, 0.25, 0.25] * 4, dtype=np.float32) * 1
        self.kp = np.array([20.0, 20.0, 20.0] * 4, dtype=np.float32) * 2
        self.kd = np.array([0.5, 0.5, 0.5] * 4, dtype=np.float32)

        #
        # MDP
        #
        # Observation contains
        #       linear vel in body frame: 3
        #       angular vel in body frame: 3
        #       gravity in body frame: 3
        #       joint position: 12
        #       joint vel: 12
        #       previous action: 12
        #       command: 3
        #
        # Command contains
        #       [x vel, y vel, yaw rate]
        #
        self.action_space = spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
        self.obs_size = 3 + 3 + 3 + 12 + 12 + 12 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_size,), dtype=np.float32)
        self.command_lb = np.array([-0.5, -0.3, -0.3], dtype=np.float32)
        self.command_ub = np.array([0.8, 0.3, 0.3], dtype=np.float32)
        self.num_step = 0
        self.rng = np.random.default_rng()

        self.commands = {
            "forward" : np.array([0.4, 0., 0.]),
            "backward" : np.array([-0.4, 0., 0.]),
            "left" : np.array([0., 0.3, 0.]),
            "right" : np.array([0., -0.3, 0.]),
            "left_turn" : np.array([0., 0., 0.3]),
            "right_turn" : np.array([0., 0., -0.3]),
            "stand" : np.array([0., 0., 0.]),
        }


        #
        # State place holder
        #
        self.base_ypr = np.zeros([3,], dtype=np.float32)
        self.projected_gravity = np.zeros([3,], dtype=np.float32)
        self.lin_vel_body = np.zeros([3,], dtype=np.float32)
        self.command = np.zeros([3,], dtype=np.float32)
        self.feet_in_contact = np.zeros([4,], dtype=np.float32)
        self.feet_contact_force = np.zeros([4,], dtype=np.float32)
        self.feet_contact_normal = np.zeros([4,3], dtype=np.float32)
        self.feet_tan_vel = np.zeros([4,], dtype=np.float32)
        self.bad_collision = False

        self.prev_action = np.zeros([self.model.nu,], dtype=np.float32)
        self.prev_joint_vel = np.zeros([12,], dtype=np.float32)
        self.feet_air_time = np.zeros([4,], dtype=np.float32)

    def _sample_command(self):
        # command = np.zeros(3, dtype=np.float32)
        # mode = self.rng.integers(3)
        # if mode == 0:
        #     command[0] = self.rng.uniform(self.command_lb[0], self.command_ub[0])
        # elif mode == 1:
        #     command[1] = self.rng.uniform(self.command_lb[1], self.command_ub[1])
        # else:
        #     command[2] = self.rng.uniform(self.command_lb[2], self.command_ub[2])

        command = self.rng.uniform(
            low=self.command_lb,
            high=self.command_ub,
        ).astype(np.float32)

        if np.linalg.norm(command) < 0.1:
            command[:] = 0.0

        self.command = command

        # items = list(self.commands.items())
        # k, v = items[self.rng.integers(len(items))]
        # self.command = v.copy()

    def _update_state(self):
        rot = quat_to_rotmat(self.data.qpos[3:7])
        self.base_ypr = np.array([*quat_to_euler(self.data.qpos[3:7])], dtype=np.float32)
        self.projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        self.lin_vel_body = rot.T @ self.data.qvel[:3]


        self.feet_in_contact[:] = 0.0
        self.feet_contact_force[:] = 0.0
        self.feet_contact_normal[:] = 0.0
        self.feet_contact_normal[:,-1] = 1.
        self.feet_tan_vel[:] = 0.0
        self.bad_collision = False
        


        contact_ft = np.zeros([6,]) # contact force and torque
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # any non-foot collision that involves bad_collision_body_ids
            if body1 in self.bad_collision_body_ids or body2 in self.bad_collision_body_ids:
                if geom1 not in self.feet_geom_ids and geom2 not in self.feet_geom_ids:
                    self.bad_collision = True
                    continue

            if geom1 in self.feet_geom_ids:
                fid = geom1
                fidx = self.feet_geom_ids.index(geom1)
                self.feet_in_contact[fidx] = 1.0
            elif geom2 in self.feet_geom_ids:
                fid = geom2
                fidx = self.feet_geom_ids.index(geom2)
                self.feet_in_contact[fidx] = 1.0
            else:
                continue

            # get foot collision info
            mujoco.mj_contactForce(self.model, self.data, ci, contact_ft)
            fnorm = float(np.linalg.norm(contact_ft[:3]))
            normal = np.asarray(contact.frame[:3]).copy()
            normal_norm = np.linalg.norm(normal)
            fidx = self.feet_geom_ids.index(fid)
            normal = normal / normal_norm if normal_norm > 1e-8 else [0., 0., 1.]
            if fnorm >= self.feet_contact_force[fidx]:
                self.feet_contact_force[fidx] = fnorm
                self.feet_contact_normal[fidx] = normal

        jacp = np.zeros([3, self.model.nv])
        for fidx in range(4):
            if not self.feet_in_contact[fidx]:
                continue
            mujoco.mj_jacGeom(self.model, self.data, jacp, None, self.feet_geom_ids[fidx])
            v = jacp @ self.data.qvel
            n = self.feet_contact_normal[fidx]
            v_tan = v - (v @ n) * n
            self.feet_tan_vel[fidx] = np.linalg.norm(v_tan)

    def _get_obs(self):
        #       linear vel in body frame: 3
        #       angular vel in body frame: 3
        #       gravity in body frame: 3
        #       joint position: 12
        #       joint vel: 12
        #       previous action: 12
        #       command: 3
        return np.concatenate([
            self.lin_vel_body,
            self.data.qvel[3:6],
            self.projected_gravity,
            self.data.qpos[7:] - self.default_joint_pos,
            self.data.qvel[6:],
            self.prev_action,
            self.command
        ]).astype(np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_ctrl
        self.data.qpos[7:] += self.rng.normal(0.0, 0.015, size=12)
        mujoco.mj_forward(self.model, self.data)

        self.num_step = 0
        self._sample_command()
        self._update_state()

        self.prev_action[:] = 0.
        self.prev_joint_vel[:] = 0.
        self.feet_air_time[:] = 0.

        return self._get_obs(), {}

    def _reward(self, action):

        if np.any(np.abs(self.base_ypr[1:]) >= np.deg2rad(30)) or self.bad_collision:
            terminated = True
            reward = -1.
            return terminated, reward


        stand_still_cmd = np.linalg.norm(self.command) < 0.1
        #
        # velocity tracking
        #
        lin_err = np.sum((self.lin_vel_body[:2] - self.command[:2]) ** 2)
        yaw_err = (self.data.qvel[5] - self.command[2]) ** 2
        linear_tracking_r = np.exp(-lin_err/0.1)
        angular_tracking_r = np.exp(-yaw_err/0.1)

        #
        # feet in air
        #
        first_contact = (self.feet_air_time > 0.0) & (self.feet_in_contact > 0.5)
        # feet_air_time_r = np.sum(
        #     (self.feet_air_time - 0.5) * first_contact
        # ) * (1 - stand_still_cmd)

        feet_air_time_r = np.sum(
            np.exp(-(self.feet_air_time - 0.5)**2 / 0.1) * first_contact
        ) * (1 - stand_still_cmd)


        #
        # regularization
        #
        vz_p = self.data.qvel[2]**2
        rp_rate_p = np.sum(self.data.qvel[3:5]**2)
        joint_acc_p = np.sum(
            ((self.data.qvel[6:] - self.prev_joint_vel) / self.dt)**2
        )
        torque_p = np.sum(self.data.ctrl ** 2)
        action_rate_p = np.sum((action - self.prev_action)**2)
        joint_diff_p = np.sum((self.data.qpos[7:] - self.home_qpos[7:]) ** 2)
        feet_slip_p = np.sum(self.feet_tan_vel * self.feet_in_contact)

        # print(linear_tracking_r)


        # stablebaseline
        # total_reward = (
        #     1.0 * linear_tracking_r
        #     + 0.5 * angular_tracking_r
        #     + 1. * feet_air_time_r
        #     - 2.0 * vz_p
        #     - 0.05 * rp_rate_p
        #     - 1e-5 * torque_p
        #     - 2.5e-7 * joint_acc_p
        #     - 0.001 * action_rate_p
        #     - 0.005 * joint_diff_p
        #     - 0.05 * feet_slip_p
        # ) * self.dt
        
        total_reward = (
            1.0 * linear_tracking_r
            + 0.5 * angular_tracking_r
            + 1. * feet_air_time_r
            - 2.0 * vz_p
            - 0.05 * rp_rate_p
            - 1e-5 * torque_p
            - 2.5e-7 * joint_acc_p
            - 0.001 * action_rate_p
            - 0.005 * joint_diff_p
            - 0.05 * feet_slip_p
        ) * self.dt

        return False, total_reward

    def step(self, action):

        action = np.clip(action, -1., 1.)
        target_qpos = self.default_joint_pos + action * self.action_scale
        ctrl = self.kp * (target_qpos - self.data.qpos[7:]) - self.kd * self.data.qvel[6:]
        self.data.ctrl[:] = np.clip(ctrl, self.ctrl_low, self.ctrl_high)

        mujoco.mj_step(self.model, self.data, self.sim_step)
        self._update_state()
        terminated, reward = self._reward(action)

        self.num_step += 1

        truncated = self.num_step >= self.episode_step
        self.prev_action[:] = action
        self.prev_joint_vel[:] = self.data.qvel[6:]
        self.feet_air_time = (self.feet_air_time + self.dt) * (1 - self.feet_in_contact)

        if self.num_step % self.resample_step == 0:
            self._sample_command()

        return self._get_obs(), reward, terminated, truncated, {}
        





            
            
