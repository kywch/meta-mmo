import numpy as np
import gymnasium as gym

from nmmo.entity.entity import EntityState

import reinforcement_learning.wrapper_helper as whp
from reinforcement_learning.stat_wrapper import BaseStatWrapper

EntityAttr = EntityState.State.attr_name_to_col


PASSIVE_REPR = 1  # matched to npc_type
NEUTRAL_REPR = 2
HOSTILE_REPR = 3
ENEMY_REPR = 4
DESTROY_TARGET_REPR = 5
TEAMMATE_REPR = 6
PROTECT_TARGET_REPR = 7


class TeamWrapper(BaseStatWrapper):
    def __init__(
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        use_custom_reward=True,
        augment_obs=True,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)
        self.config = env.config
        self._augment_obs = augment_obs

        # Team/agent, system states, task embedding
        self._task = {}
        self._task_obs = {  # Task dim: 2059
            agent_id: np.zeros(
                1 + len(self.config.system_states) + self.config.TASK_EMBED_DIM, dtype=np.float16
            )
            for agent_id in self.env.possible_agents
        }

        # Used for obs augmentation
        self._extra_tile_obs = 3  # extra maps: dist, entity, rally point
        self._dummy_tile_obs = np.zeros(
            (self.config.MAP_N_OBS, self._extra_tile_obs), dtype=np.int16
        )
        self._dummy_entity_obs = np.zeros(
            (self.config.PLAYER_N_OBS, EntityState.State.num_attributes), dtype=np.int16
        )
        self._obs_data = {
            agent_id: {
                "entity_obs": self._dummy_entity_obs,
                "entity_map": np.zeros(
                    (self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16
                ),
                "rally_map": np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16),
                "rally_target": None,
                "can_see_target": False,
                "pass_to_whp": {
                    "my_team": set(),
                    "target_destroy": set(),
                    "target_protect": set(),
                    "ENEMY_REPR": ENEMY_REPR,
                    "DESTROY_TARGET_REPR": TEAMMATE_REPR,
                    "TEAMMATE_REPR": TEAMMATE_REPR,
                    "PROTECT_TARGET_REPR": PROTECT_TARGET_REPR,
                },
            }
            for agent_id in self.env.possible_agents
        }

        # Dist map should not change from episode to episode
        self._dist_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        center = self.config.MAP_SIZE // 2
        for i in range(center):
            lb, rb = i, self.config.MAP_SIZE - i
            self._dist_map[lb:rb, lb:rb] = center - i - 1

    def observation_space(self, agent_id):
        """If you modify the shape of features, you need to specify the new obs space"""
        obs_space = super().observation_space(agent_id)

        # Add system states to the task obs
        obs_space["Task"] = gym.spaces.Box(
            low=-(2**15), high=2**15 - 1, dtype=np.float16, shape=self._task_obs[1].shape
        )

        # NOTE: To see the effect of obs augmentation, make the obs space the same
        tile_dim = self._extra_tile_obs + 3  # 3 for the original tile obs
        obs_space["Tile"] = gym.spaces.Box(
            low=-(2**15), high=2**15 - 1, dtype=np.int16, shape=(self.config.MAP_N_OBS, tile_dim)
        )

        return obs_space

    def reset(self, **kwargs):
        """Called at the start of each episode"""
        obs, info = super().reset(**kwargs)
        obs = self._reset_team_vars(obs)
        return obs, info

    def _reset_team_vars(self, obs):
        self._task = {
            agent_id: self.env.agent_task_map[agent_id][0] for agent_id in self.env.possible_agents
        }

        for agent_id in self.env.possible_agents:
            self._task_obs[agent_id][0] = float(self._task[agent_id].reward_to == "team")
            self._task_obs[agent_id][1 : 1 + len(self.config.system_states)] = (
                self.config.system_states
            )
            self._task_obs[agent_id][1 + len(self.config.system_states) :] = self._task[
                agent_id
            ].embedding

            # Reset the _data
            self._obs_data[agent_id]["entity_obs"] = self._dummy_entity_obs
            self._obs_data[agent_id]["entity_map"][:] = 0
            self._obs_data[agent_id]["rally_map"][:] = 0
            self._obs_data[agent_id]["rally_target"] = None
            self._obs_data[agent_id]["can_see_target"] = False
            self._obs_data[agent_id]["pass_to_whp"]["my_team"] = set(self._task[agent_id].assignee)

            if (
                "SeizeCenter" in self._task[agent_id].name
                or "ProgressTowardCenter" in self._task[agent_id].name
            ):
                self._obs_data[agent_id]["rally_target"] = self.env.realm.map.center_coord
                self._obs_data[agent_id]["rally_map"][self.env.realm.map.center_coord] = 1

            # get target_protect, target_destroy from the task, for ProtectAgent and HeadHunting
            self._obs_data[agent_id]["pass_to_whp"]["target_protect"] = set()
            if "target_protect" in self._task[agent_id].kwargs:
                target = self._task[agent_id].kwargs["target_protect"]
                self._obs_data[agent_id]["pass_to_whp"]["target_protect"] = (
                    set([target]) if isinstance(target, int) else set(target)
                )

            self._obs_data[agent_id]["pass_to_whp"]["target_destroy"] = set()
            for key in ["target", "target_destroy"]:
                if key in self._task[agent_id].kwargs:
                    target = self._task[agent_id].kwargs[key]
                    self._obs_data[agent_id]["pass_to_whp"]["target_destroy"] = (
                        set([target]) if isinstance(target, int) else set(target)
                    )

            # Update the task obs
            obs[agent_id] = self.observation(agent_id, obs[agent_id])
        return obs

    def observation(self, agent_id, agent_obs):
        """Called before observations are returned from the environment
        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))"""
        agent_obs["Task"] = self._task_obs[agent_id]

        # Do NOT attack teammates
        agent_obs["ActionTargets"]["Attack"]["Target"] = self._process_attack_mask(
            agent_id, agent_obs
        )

        self._obs_data[agent_id]["entity_obs"] = agent_obs["Entity"]
        if self._augment_obs:
            whp.update_entity_map(
                self._obs_data[agent_id]["entity_map"],
                agent_obs["Entity"],
                EntityAttr,
                self._obs_data[agent_id]["pass_to_whp"],
            )

        agent_obs["Tile"] = self._augment_tile_obs(agent_id, agent_obs)

        return agent_obs

    def _process_attack_mask(self, agent_id, agent_obs):
        whole_mask = agent_obs["ActionTargets"]["Attack"]["Target"]
        entity_mask = whole_mask[:-1]
        if entity_mask.sum() == 0 and whole_mask[-1] == 1:  # no valid target
            return whole_mask
        if agent_id not in self._task or len(self._task[agent_id].assignee) == 1:  # no team
            return whole_mask

        # the order of entities in obs["Entity"] is the same as in the mask
        teammate = np.in1d(agent_obs["Entity"][:, EntityAttr["id"]], self._task[agent_id].assignee)
        # Do NOT attack teammates
        entity_mask[teammate] = 0
        if entity_mask.sum() == 0:
            whole_mask[-1] = 1  # if no valid target, make sure to turn on no-op

        return whole_mask

    def _augment_tile_obs(self, agent_id, agent_obs):
        tile_obs = agent_obs["Tile"][:, :3]

        if self._augment_obs:
            dist = self._dist_map[tile_obs[:, 0], tile_obs[:, 1]]

            # Entity map should be updated
            entity = self._obs_data[agent_id]["entity_map"][tile_obs[:, 0], tile_obs[:, 1]]

            # All zero if no rally point
            rally_point = self._obs_data[agent_id]["rally_map"][tile_obs[:, 0], tile_obs[:, 1]]

            # To communicate if the agent can see the target
            self._obs_data[agent_id]["can_see_target"] = (
                entity == DESTROY_TARGET_REPR
            ).sum() > 0 or (
                self._obs_data[agent_id]["rally_target"] is not None and rally_point.sum() > 0
            )

            maps = [tile_obs, dist[:, None], entity[:, None], rally_point[:, None]]

        else:
            # NOTE: Append a dummy obs, to keep the obs space consistent
            maps = [tile_obs, self._dummy_tile_obs]

        return np.concatenate(maps, axis=1).astype(np.int16)

    def action(self, agent_id, agent_atn):
        """Called before actions are passed to the environment"""

        # Override communication with manually computed one
        # NOTE: Can this be learned from scratch?
        agent_atn["Comm"]["Token"] = self._compute_comm_action(agent_id)
        return agent_atn

    def _compute_comm_action(self, agent_id):
        # comm action values range from 0 - 127, 0: dummy obs
        if agent_id not in self.env.realm.players:
            return 0

        agent = self.env.realm.players[agent_id]
        return whp.compute_comm_action(
            self._obs_data[agent_id]["can_see_target"],
            agent.resources.health.val,
            self._obs_data[agent_id]["entity_obs"],
            EntityAttr,
            self._obs_data[agent_id]["pass_to_whp"],
        )
