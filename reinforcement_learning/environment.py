from argparse import Namespace

import pufferlib
import pufferlib.emulation
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

import nmmo
import nmmo.core.config as nc
import nmmo.core.game_api as ng
from nmmo.task import task_spec

def alt_combat_damage_formula(offense, defense, multiplier, minimum_proportion):
    return int(max(multiplier * offense - defense, offense * minimum_proportion))


class TeamBattle(ng.TeamBattle):
    def _set_config(self):
        self.config.reset()
        self.config.set_for_episode("DEATH_FOG_SPEED", 1/6)
        self.config.set_for_episode("DEATH_FOG_FINAL_SIZE", 8)

    def _define_tasks(self):
        sampled_spec = self._get_cand_team_tasks(num_tasks=1, tags="team_battle")[0]
        return task_spec.make_task_from_spec(self.config.TEAMS,
                                             [sampled_spec] * len(self.config.TEAMS))


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
        self.set("MAP_FORCE_GENERATION", env_args.map_force_generation)
        self.set("PLAYER_N", env_args.num_agents)
        self.set("HORIZON", env_args.max_episode_length)
        self.set("MAP_N", env_args.num_maps)
        self.set("TEAMS", {i: [i*env_args.num_agents_per_team+j+1 for j in range(env_args.num_agents_per_team)]
                           for i in range(env_args.num_agents // env_args.num_agents_per_team)})
        self.set(
            "PLAYER_DEATH_FOG",
            env_args.death_fog_tick if isinstance(env_args.death_fog_tick, int) else None,
        )
        self.set("PATH_MAPS", f"{env_args.maps_path}/{env_args.map_size}/")
        self.set("MAP_CENTER", env_args.map_size)
        self.set("NPC_N", env_args.num_npcs)
        self.set("TASK_EMBED_DIM", env_args.task_size)
        self.set("RESOURCE_RESILIENT_POPULATION", env_args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", env_args.spawn_immunity)

        self.set("GAME_PACKS", [(TeamBattle, 1)])
        self.set("CURRICULUM_FILE_PATH", env_args.curriculum_file_path)

        # Game-balancing related, making the game somewhat easier
        # since all agents are on their own (no team play)
        # self.set("NPC_LEVEL_DEFENSE", 8)  # from 15
        # self.set("NPC_BASE_DAMAGE", 0)  # from 15
        # self.set("NPC_LEVEL_DAMAGE", 8)  # from 15

        self.set("TERRAIN_SCATTER_EXTRA_RESOURCES", True)  # extra food/water

        # TODO: check if these values are good and then make them default
        self.set("COMBAT_DAMAGE_FORMULA", alt_combat_damage_formula)

        self.set("PROGRESSION_COMBAT_XP_SCALE", 6)  # from 3
        self.set("PROGRESSION_MELEE_BASE_DAMAGE", 10)  # from 20
        self.set("PROGRESSION_RANGE_BASE_DAMAGE", 10)
        self.set("PROGRESSION_MAGE_BASE_DAMAGE", 10)

        self.set("EQUIPMENT_WEAPON_BASE_DAMAGE", 5)  # from 15
        self.set("EQUIPMENT_WEAPON_LEVEL_DAMAGE", 5)  # from 15

        self.set("EQUIPMENT_AMMUNITION_BASE_DAMAGE", 0)  # from 15
        self.set("EQUIPMENT_AMMUNITION_LEVEL_DAMAGE", 10)  # from 15

        self.set("EQUIPMENT_TOOL_BASE_DEFENSE", 15)  # from 30

        self.set("EQUIPMENT_ARMOR_LEVEL_DEFENSE", 3)  # from 10


class FullGameConfig(
    MiniGameConfig,
    nc.Progression,
    nc.Item,
    nc.Equipment,
    nc.Profession,
    nc.Exchange,
):
    pass


def make_env_creator(reward_wrapper_cls: BaseParallelWrapper,
                     config_scope: str = None,
                     game_cls: ng.Game = None):
    def env_creator(*args, **kwargs):
        """Create an environment."""
        # args.env is provided as kwargs
        config = MiniGameConfig(kwargs["env"]) if config_scope == "mini" else FullGameConfig(kwargs["env"])

        # Default game is TeamBattle
        if game_cls and isinstance(game_cls, ng.Game):
            config.set("GAME_PACKS", [(game_cls, 1)])

        env = nmmo.Env(config)
        env = reward_wrapper_cls(env, **kwargs["reward_wrapper"])
        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator
