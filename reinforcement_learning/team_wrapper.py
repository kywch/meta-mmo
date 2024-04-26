import numpy as np
import gymnasium as gym

from nmmo.entity.entity import EntityState

from .stat_wrapper import BaseStatWrapper

EntityAttr = EntityState.State.attr_name_to_col


# # TODO: Implement realikun-style obs augmentation
# from nmmo.lib import material
# import reinforcement_learning.wrapper_helper as whp
# IMPASSIBLE = list(material.Impassible.indices)
# PASSIVE_REPR = 1  # matched to npc_type
# NEUTRAL_REPR = 2
# HOSTILE_REPR = 3
# ENEMY_REPR = 4
# DESTROY_TARGET_REPR = 5
# TEAMMATE_REPR = 6
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

        # # TODO: Used for tile obs augmentation
        # self._entity_obs = None  # placeholder
        # self._entity_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        # self._rally_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        # self._rally_target = None

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

        # if self._augment_obs:
        #     obs_space["Tile"] = gym.spaces.Box(
        #         low=0, high=2**15 - 1, dtype=np.float16, shape=(self.config.MAP_SIZE, self.config.MAP_SIZE)
        #     )

        return obs_space

    def reset(self, **kwargs):
        """Called at the start of each episode"""
        obs, info = super().reset(**kwargs)
        obs = self._reset_team_vars(obs)

        # # Set the task-related vars for obs augmentation
        # if self._augment_obs:
        #     self._rally_target = None
        #     self._rally_map[:] = 0
        #     if self._task[1] is not None:
        #         task_name = self._task[1].name
        #         # NOTE: Assume the mini games assign the same task for all agents
        #         if "SeizeCenter" in task_name or "ProgressTowardCenter" in task_name:
        #             self._rally_target = self.env.realm.map.center_coord
        #             self._rally_map[self._rally_target] = 1

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
