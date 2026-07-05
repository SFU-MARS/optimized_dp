import os
import hashlib

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

try:
    from utils.plot_utils import visual_map
except ImportError:
    print("Warning: Could not import visual_map from utils.plot_utils. Visualization will be skipped.")
    visual_map = None
    

class ToyDubinsCarNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        map_size=5.0,
        goal_pos=[1.5, 1.5],
        obs_pos=[0.0, 0.0],
        hazards_size=0.6,
        goal_size=0.3,
        max_episode_steps=1000,
        agent_radius=0.1,
        num_envs=1,  # num_envs = 1 when training
        # offline data info, based on the collected data
        min_episode_reward=None,
        max_episode_reward=None,
        min_episode_cost=0.0,
        max_episode_cost=None,
        target_cost=1.0,
        **kwargs
    ):
        self.num_envs = num_envs
        self.is_vectorized = num_envs > 1
        
        self.map_size = map_size
        self.max_episode_steps = max_episode_steps
        self.agent_radius = agent_radius
        
        self.goal_pos_coords = np.array(goal_pos, dtype=np.float32)
        self.obs_pos_coords = np.array(obs_pos, dtype=np.float32)
        self.hazards_size = hazards_size
        self.goal_threshold = goal_size
        
        # DSRL Params for offline usage
        self.min_episode_reward = min_episode_reward
        self.max_episode_reward = max_episode_reward
        self.min_episode_cost = min_episode_cost
        self.max_episode_cost = max_episode_cost
        self.target_cost = target_cost
        self.epsilon = 1 if self.target_cost == 0 else 0
        
        # Physics params
        self.dt = 0.1
        self.v_max = 1.0
        self.w_max = 1.0
        self.max_dist = np.sqrt(2 * (self.map_size ** 2))

        self.goal_pos = np.tile(self.goal_pos_coords, (self.num_envs, 1))
        self.obstacle_pos = np.tile(self.obs_pos_coords, (self.num_envs, 1))
        
        self._build_space()
        
        # Internal State
        self.state = np.zeros((self.num_envs, 3), dtype=np.float32)  # (x, y, theta)
        self._step_count = np.zeros(self.num_envs, dtype=np.int32)
        self.last_dist_goal = np.zeros(self.num_envs, dtype=np.float32)
        
        self.env_hash = self._generate_hash()

        # Overridden per ``reset(..., options=…)``; reused for auto-reset inside ``step``.
        self._initial_heading_mode = "random"
        self._initial_heading_noise_std = 0.0
        self._spawn_box = None  # optional (xmin, ymin, xmax, ymax) in world coords

    def _build_space(self):
        """
        observation: [d_g, sin(alpha_g), cos(alpha_g), d_o, sin(alpha_o), cos(alpha_o)]
                d_g: normalized distance to the goal (relative distance / max distance of the map)
                sin(alpha_g), cos(alpha_g): relative heading to the goal
                d_o: normalized distance to the obstacle
                sin(alpha_o), cos(alpha_o): relative heading to the obstacle
        action: [v, w]
            v: speed
            w: turning rate
        """
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low =  np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
    def _sample_positions_in_spawn(self, n_invalid):
        """Sample (n_invalid, 2) candidate positions; full map or ``self._spawn_box``."""
        bound = self.map_size / 2.0
        box = self._spawn_box
        if box is None:
            return self.np_random.uniform(-bound, bound, size=(n_invalid, 2))
        xmin, ymin, xmax, ymax = [float(x) for x in box]
        low = np.array([xmin, ymin], dtype=np.float32)
        high = np.array([xmax, ymax], dtype=np.float32)
        return self.np_random.uniform(low, high, size=(n_invalid, 2))

    def _apply_initial_heading(self, env_idxs):
        """Set yaw after positions are fixed (``toward_goal`` or uniform random)."""
        if self._initial_heading_mode == "toward_goal":
            x = self.state[env_idxs, 0]
            y = self.state[env_idxs, 1]
            gx = self.goal_pos[env_idxs, 0]
            gy = self.goal_pos[env_idxs, 1]
            theta = np.arctan2(gy - y, gx - x)
            noise = float(self._initial_heading_noise_std)
            if noise > 0.0:
                theta = theta + self.np_random.normal(0.0, noise, size=len(env_idxs))
            self.state[env_idxs, 2] = (theta + np.pi) % (2 * np.pi) - np.pi
        else:
            self.state[env_idxs, 2] = self.np_random.uniform(-np.pi, np.pi, size=len(env_idxs))

    def _reset_idx(self, env_idxs):
        """
        Reset specific environments by index.
        """
        n_reset = len(env_idxs)
        if n_reset == 0:
            return

        bound = self.map_size / 2.0
        pos = np.zeros((n_reset, 2), dtype=np.float32)
        valid_mask = np.zeros(n_reset, dtype=bool)
        margin = 0.2  # leave some margin for initial pos
        # Use  agent_radius + hazards_size to make sure the collision does not happen
        safe_obs_thr = self.agent_radius + self.hazards_size + margin
        safe_goal_thr = self.goal_threshold + margin
        
        while not np.all(valid_mask):
            invalid_local_idx = np.where(~valid_mask)[0]
            n_invalid = len(invalid_local_idx)
            
            candidates = self._sample_positions_in_spawn(n_invalid)
            # Reject samples outside the legal square if spawn_box spills (misconfig)
            candidates[:, 0] = np.clip(candidates[:, 0], -bound, bound)
            candidates[:, 1] = np.clip(candidates[:, 1], -bound, bound)
            
            global_curr_idxs = env_idxs[invalid_local_idx]
            
            d_goal = np.linalg.norm(candidates - self.goal_pos[global_curr_idxs], axis=1)
            d_obs = np.linalg.norm(candidates - self.obstacle_pos[global_curr_idxs], axis=1)
            
            is_valid = (d_goal > safe_goal_thr) & (d_obs > safe_obs_thr)
            
            valid_loc_idx = invalid_local_idx[is_valid]
            pos[valid_loc_idx] = candidates[is_valid]
            valid_mask[valid_loc_idx] = True

        self.state[env_idxs, 0:2] = pos
        self._apply_initial_heading(env_idxs)
        
        # Step counter
        self._step_count[env_idxs] = 0
        
        # Distance to goal
        dist_to_goal = np.linalg.norm(self.state[env_idxs, :2] - self.goal_pos[env_idxs], axis=1)
        self.last_dist_goal[env_idxs] = dist_to_goal
    
    def _dynamics(self, action):
        """Pure physics update."""
        action = np.clip(action, -1.0, 1.0)
        v = (action[:, 0] + 1) / 2 * self.v_max 
        w = action[:, 1] * self.w_max          
        
        theta = self.state[:, 2]
        self.state[:, 0] += v * np.cos(theta) * self.dt
        self.state[:, 1] += v * np.sin(theta) * self.dt
        self.state[:, 2] += w * self.dt
        
        self.state[:, 2] = (self.state[:, 2] + np.pi) % (2 * np.pi) - np.pi
    
    def _get_obs(self):
        x = self.state[:, 0]
        y = self.state[:, 1]
        theta = self.state[:, 2]
        
        # Relative info to the goal
        dx_g = self.goal_pos[:, 0] - x
        dy_g = self.goal_pos[:, 1] - y
        dist_g = np.sqrt(dx_g**2 + dy_g**2)
        alpha_g = np.arctan2(dy_g, dx_g) - theta
        
        # Relative info to the obstacle
        dx_o = self.obstacle_pos[:, 0] - x
        dy_o = self.obstacle_pos[:, 1] - y
        dist_o = np.sqrt(dx_o**2 + dy_o**2)
        alpha_o = np.arctan2(dy_o, dx_o) - theta
        
        norm_dg = np.clip(dist_g / self.max_dist, 0, 1)
        norm_do = np.clip(dist_o / self.max_dist, 0, 1)
        
        obs = np.stack([
            norm_dg, np.sin(alpha_g), np.cos(alpha_g),
            norm_do, np.sin(alpha_o), np.cos(alpha_o)
        ], axis=1).astype(np.float32)
        
        return obs
    
    def _generate_hash(self):
        """
        A unique hash is generated based on the map, target, obstacles, and their dimensions.
        The hash will be completely different as long as these physical parameters change.
        """
        hash_str = (
            f"map:{float(self.map_size)}|"
            f"goal:{np.round(self.goal_pos_coords, 4).tolist()}|"
            f"obs:{np.round(self.obs_pos_coords, 4).tolist()}|"
            f"h_size:{float(self.hazards_size)}|"
            f"g_size:{float(self.goal_threshold)}|"
            f"a_rad:{float(self.agent_radius)}"
        )
        
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initial_heading_mode = "random"
        self._initial_heading_noise_std = 0.0
        self._spawn_box = None
        if options:
            mode = options.get("initial_heading", "random")
            if mode not in ("random", "toward_goal"):
                raise ValueError(
                    "ToyDubinsCarNavEnv: initial_heading must be 'random' or 'toward_goal', "
                    f"got {mode!r}"
                )
            self._initial_heading_mode = mode
            self._initial_heading_noise_std = float(options.get("heading_noise_std", 0.0))
            sb = options.get("spawn_box")
            if sb is not None:
                self._spawn_box = np.asarray(sb, dtype=np.float64).reshape(4)
        all_idxs = np.arange(self.num_envs)
        self._reset_idx(all_idxs)
        obs = self._get_obs()
        return (obs, {}) if self.is_vectorized else (obs[0], {})

    def step(self, action):
        action = np.asarray(action)
        if not self.is_vectorized and action.ndim == 1:
            action = action[np.newaxis, :]
            
        self._dynamics(action)
        self._step_count += 1
        
        current_pos = self.state[:, :2]
        dist_to_goal = np.linalg.norm(current_pos - self.goal_pos, axis=1)
        dist_to_obs = np.linalg.norm(current_pos - self.obstacle_pos, axis=1)
        
        # Cost computation
        collision_threshold = self.agent_radius + self.hazards_size
        cost = (dist_to_obs < collision_threshold).astype(np.float32)
        
        # Reward computation
        reached_goal = dist_to_goal < self.goal_threshold
        reward_dist = self.last_dist_goal - dist_to_goal
        self.last_dist_goal = dist_to_goal
        reward_goal = reached_goal.astype(np.float32) * 1.0
        rewards = reward_dist + reward_goal
        
        # Terminated / Truncated
        terminated = reached_goal | \
                     ((np.abs(self.state[:,0]) > self.map_size/2) | \
                      (np.abs(self.state[:,1]) > self.map_size/2))     
        truncated = self._step_count >= self.max_episode_steps
        
        info = {
            "cost": cost, 
            "min_dist_to_obstacle": dist_to_obs,
            "goal_met": reached_goal,
            "reward_dist": reward_dist,
            "reward_goal": reward_goal,
            "episode_len": self._step_count.copy(), 
        }
        
        obs = self._get_obs()

        # --- Auto-Reset (Vectorized) ---
        if self.is_vectorized:
            dones = terminated | truncated
            if np.any(dones):
                # Save terminal transition info before reset
                info["final_info"] = {
                    k: v[dones] for k, v in info.items() if isinstance(v, np.ndarray)
                }
                info["final_observation"] = obs[dones].copy()

                done_idxs = np.where(dones)[0]
                self._reset_idx(done_idxs)
                new_obs = self._get_obs()
                obs[dones] = new_obs[dones]

            return obs, rewards, terminated, truncated, info

        # --- Single Env Return (scalars for FSRL/Tianshou GAE compatibility) ---
        else:
            is_done = bool(terminated[0] or truncated[0])

            # Cache terminal info BEFORE reset
            episode_len = int(self._step_count[0])
            final_observation = obs[0].copy() if is_done else None

            if is_done:
                self._reset_idx(np.array([0]))
                obs = self._get_obs()

            # Return scalars so batch.rew / batch.info["cost"]
            # stay 1D (batch_len,) not (batch_len, 1)
            obs_out = obs[0]
            rew_out = float(rewards[0])
            term_out = bool(terminated[0])
            trunc_out = bool(truncated[0])

            info_out = {
                "cost": float(cost[0]),
                "goal_met": bool(reached_goal[0]),
                "reward_dist": float(reward_dist[0]),
                "reward_goal": float(reward_goal[0]),
                "episode_len": episode_len,
            }

            if final_observation is not None:
                info_out["final_observation"] = final_observation

            return obs_out, rew_out, term_out, trunc_out, info_out
    
    # Added Methods for DSRL Pipeline
    def set_target_cost(self, target_cost):
        self.target_cost = target_cost
        self.epsilon = 1 if self.target_cost == 0 else 0
    
    def set_episode_info(self, **kwargs):
        """
        Dynamically update environment metadata (e.g., normalization stats) 
        after the environment has been initialized.
        """
        allowed_keys = [
            'min_episode_reward', 
            'max_episode_reward', 
            'min_episode_cost', 
            'max_episode_cost', 
        ]

        for key in allowed_keys:
            if key in kwargs and kwargs[key] is not None:
                setattr(self, key, kwargs[key])
    
    def get_normalized_score(self, reward, cost):
        if (self.max_episode_reward is None) or \
            (self.min_episode_reward is None) or \
            (self.target_cost is None):
            raise ValueError("Reference score not provided for env")
        normalized_reward = (reward - self.min_episode_reward
                             ) / (self.max_episode_reward - self.min_episode_reward)
        normalized_cost = (cost + self.epsilon) / (self.target_cost + self.epsilon)
        
        return normalized_reward, normalized_cost