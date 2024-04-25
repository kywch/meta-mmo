from argparse import Namespace

import pufferlib
import pufferlib.emulation
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

import nmmo
import nmmo.core.config as nc
import nmmo.core.game_api as ng
from nmmo import minigames


def get_team_dict(num_agents, num_agents_per_team):
    assert (
        num_agents % num_agents_per_team == 0
    ), "Number of agents must be divisible by number of agents per team"
    return {
        i: [i * num_agents_per_team + j + 1 for j in range(num_agents_per_team)]
        for i in range(num_agents // num_agents_per_team)
    }


class TeamBattle(ng.TeamBattle):
    def _set_config(self):
        self.config.reset()
        self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
        self.config.set_for_episode("TERRAIN_WATER", 0.1)
        self.config.set_for_episode("TERRAIN_FOILAGE", 0.9)  # prop of stone tiles: 0.1
        self.config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", True)
        self.config.set_for_episode("DEATH_FOG_SPEED", 1 / 6)
        self.config.set_for_episode("DEATH_FOG_FINAL_SIZE", 8)


class EasyKingoftheHill(minigames.KingoftheHill):
    def _set_config(self):
        super()._set_config()
        # make the game easier by decreasing the resource demands/penalty
        self.config.set_for_episode("RESOURCE_DEPLETION_RATE", 3)  # from 5
        self.config.set_for_episode("RESOURCE_RESILIENT_POPULATION", 1)


class Sandwich(minigames.Sandwich):
    _next_grass_map = None

    def set_grass_map(self, grass_map):
        self._next_grass_map = grass_map

    def _set_config(self):
        # randomly select whether to use the terrain map or grass map
        self._grass_map = self._next_grass_map
        if self._grass_map is None:
            self._grass_map = self._np_random.choice([True, False], p=[0.2, 0.8])
        super()._set_config()


class RadioRaid(minigames.RadioRaid):
    _next_grass_map = None
    _next_goal_num_npc = None

    def set_grass_map(self, grass_map):
        self._next_grass_map = grass_map

    def set_goal_num_npc(self, num_npc):
        self._next_goal_num_npc = num_npc

    def _set_config(self):
        # randomly select whether to use the terrain map or grass map
        self._grass_map = self._next_grass_map
        if self._grass_map is None:
            self._grass_map = self._np_random.choice([True, False])
        self._goal_num_npc = self._next_goal_num_npc or self._goal_num_npc
        super()._set_config()


class MiniGameConfig(
    nc.Medium,
    nc.Terrain,
    nc.Resource,
    nc.Combat,
    nc.NPC,
    nc.Communication,
):
    """Configuration for Neural MMO."""

    def __init__(self, env_args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("TASK_EMBED_DIM", 16)  # Use the default hash embedding provided by env
        self.set("MAP_FORCE_GENERATION", env_args.map_force_generation)
        self.set("PLAYER_N", env_args.num_agents)
        self.set("HORIZON", env_args.max_episode_length)
        self.set("MAP_N", env_args.num_maps)
        self.set("TEAMS", get_team_dict(env_args.num_agents, env_args.num_agents_per_team))
        self.set(
            "DEATH_FOG_ONSET",
            env_args.death_fog_tick if isinstance(env_args.death_fog_tick, int) else None,
        )
        self.set("PATH_MAPS", f"{env_args.maps_path}/{env_args.map_size}/")
        self.set("MAP_CENTER", env_args.map_size)
        self.set("NPC_N", env_args.num_npcs)
        self.set("RESOURCE_RESILIENT_POPULATION", env_args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", env_args.spawn_immunity)

        # NOTE: Disabling curriculum file for now
        # self.set("CURRICULUM_FILE_PATH", env_args.curriculum_file_path)


class FullGameConfig(
    MiniGameConfig,
    nc.Progression,
    nc.Item,
    nc.Equipment,
    nc.Profession,
    nc.Exchange,
):
    pass


def make_env_creator(
    reward_wrapper_cls: BaseParallelWrapper,
    train_flag: str = None,
    use_mini: bool = False,
):
    def env_creator(*args, **kwargs):
        """Create an environment."""
        # args.env is provided as kwargs
        if use_mini is True:
            config = MiniGameConfig(kwargs["env"])
        else:
            config = FullGameConfig(kwargs["env"])

        if train_flag is None:
            game_packs = [
                (TeamBattle, 10),
                (minigames.RacetoCenter, 1),
                (EasyKingoftheHill, 1),
                (Sandwich, 1),
            ]
        elif train_flag == "tb_only":
            game_packs = [(TeamBattle, 1)]
        else:
            raise ValueError(f"Invalid train_flag: {train_flag}")
        config.set("GAME_PACKS", game_packs)

        env = nmmo.Env(config)
        env = reward_wrapper_cls(env, **kwargs["reward_wrapper"])
        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator
