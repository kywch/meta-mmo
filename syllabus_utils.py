"""Syllabus task wrapper for NMMO."""

import gymnasium as gym
import nmmo
from nmmo.lib.material import Harvestable
from nmmo.task import base_predicates as bp
from nmmo.systems import item as i
from nmmo.entity import entity as e
from nmmo.task import task_spec
from nmmo.task.base_predicates import StayAlive
from nmmo.task.task_api import OngoingTask
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper
import pufferlib
import pufferlib.emulation
from syllabus.core import Curriculum, MultiagentSharedCurriculumWrapper, make_multiprocessing_curriculum
from syllabus.core import PettingZooMultiProcessingSyncWrapper as SyllabusSyncWrapper

from syllabus.curricula import SequentialCurriculum
from syllabus.core.task_interface import PettingZooTaskWrapper
from syllabus.task_space import TaskSpace

from reinforcement_learning import environment
from reinforcement_learning.environment import TeamBattle, RacetoCenter, EasyKingoftheHill, Sandwich, AmmoTraining, FullGameConfig, MiniGameConfig

def make_syllabus_env_creator(args, agent_module):
    sample_env_creator = environment.make_env_creator(
        reward_wrapper_cls=agent_module.RewardWrapper, syllabus_wrapper=True
    )
    sample_env = sample_env_creator(env=args.env, reward_wrapper=args.reward_wrapper)

    task_space = SyllabusTaskWrapper.task_space
    curriculum = create_sequential_curriculum(task_space)
    curriculum = MultiagentSharedCurriculumWrapper(curriculum, sample_env.possible_agents)
    curriculum = make_multiprocessing_curriculum(curriculum)

    return curriculum, environment.make_env_creator(
        reward_wrapper_cls=agent_module.RewardWrapper, syllabus=curriculum
    )


def make_syllabus_mini_env_creator(
    reward_wrapper_cls: BaseParallelWrapper,
    curriculum: Curriculum = None,
    syllabus_wrapper: bool = False,
    use_mini: bool = False,
):

    def env_creator(*args, **kwargs):
        """Create an environment."""
        # args.env is provided as kwargs
        if use_mini is True:
            config = MiniGameConfig(kwargs["env"])
        else:
            config = FullGameConfig(kwargs["env"])

        env = nmmo.Env(config)
        env = reward_wrapper_cls(env, **kwargs["reward_wrapper"])

        # Add Syllabus task wrapper
        if syllabus_wrapper or curriculum is not None:
            env = SyllabusMinigameTaskWrapper(env)

        # Use syllabus curriculum if provided
        if curriculum is not None:
            env = SyllabusSyncWrapper(
                env,
                curriculum.get_components(),
                update_on_step=False,
                task_space=env.task_space,
            )
        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator


def create_sequential_curriculum(task_space):
    curricula = []
    stopping = []

    stage1 = list(range(10))
    stopping.append("episode_return>=0.75&episodes>=20000")

    stage2 = list(range(10, 20))
    stopping.append("episode_return>=0.75&episodes>=20000")

    stage3 = list(range(20, 30))
    stopping.append("episode_return>=0.75&episodes>=20000")

    stage4 = list(range(30, 40))
    stopping.append("episode_return>=0.75&episodes>=20000")

    stage5 = list(range(40, 50))

    curricula = [stage1, stage2, stage3, stage4, stage5]
    return SequentialCurriculum(curricula, stopping, task_space, return_buffer_size=1000)


# TeamBattle
# RacetoCenter
# EasyKingoftheHill
# Sandwich
# AmmoTraining
def create_race_koth_curriculum(task_space):
    curricula = []
    stopping = []

    stage1 = 1  # RacetoCenter
    stopping.append("episode_return>=0.75&episodes>=10000")

    stage2 = 2  # EasyKingoftheHill

    curricula = [stage1, stage2]
    return SequentialCurriculum(curricula, stopping, task_space, return_buffer_size=1000)


def create_koth_tb_curriculum(task_space):
    curricula = []
    stopping = []

    stage1 = 2  # EasyKingoftheHill
    stopping.append("episode_return>=0.75&episodes>=10000")

    stage2 = 0  # TeamBattle

    curricula = [stage1, stage2]
    return SequentialCurriculum(curricula, stopping, task_space, return_buffer_size=1000)


def create_basic_tasks(unit_count):
    return [
        task_spec.TaskSpec(bp.TickGE, {"num_tick": 50 * unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "EAT_FOOD", "N": 5 * unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "DRINK_WATER", "N": 5 * unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "HARVEST_ITEM", "N": 3 * unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "GO_FARTHEST", "N": 3 * unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "LEVEL_UP", "N": 2 * unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "EQUIP_ITEM", "N": unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "CONSUME_ITEM", "N": unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "BUY_ITEM", "N": unit_count}),
        task_spec.TaskSpec(bp.CountEvent, {"event": "PLAYER_KILL", "N": unit_count}),
    ]


def create_original_task_list():
    return [("agent", StayAlive, {"task_cls": OngoingTask})]


def sequential_task_list():
    # Sanity checks
    stage1 = create_basic_tasks(1)  # Easiest
    stage2 = create_basic_tasks(3)  # Easier
    stage3 = create_basic_tasks(5)  # Moderate
    stage4 = create_basic_tasks(7)  # Somewhat difficult
    stage5 = create_basic_tasks(10)  # Challenging
    return stage1 + stage2 + stage3 + stage4 + stage5


def create_manual_task_list():
    STAY_ALIVE_GOAL = [50, 100, 150, 200, 300, 500]
    # AGENT_NUM_GOAL = [1]    # competition team size: 8
    task_specs = []
    task_names = []

    # Find resource tiles
    for resource in Harvestable:
        for reward_to in ["agent"]:
            spec = task_spec.TaskSpec(
                bp.CanSeeTile, {"tile_type": resource}, reward_to=reward_to
            )
            task_specs.append(spec)
            # task_names.append("see_" + resource.name)

    # Stay alive
    for reward_to in ["agent"]:
        for num_tick in STAY_ALIVE_GOAL:
            spec = task_spec.TaskSpec(bp.TickGE, {"num_tick": num_tick}, reward_to=reward_to)
            task_specs.append(spec)
            # task_names.append("stay_alive_" + str(num_tick))

    # Explore the map
    for dist in [10, 20, 30, 50, 100]:  # each agent
        spec = task_spec.TaskSpec(bp.DistanceTraveled, {"dist": dist}, reward_to=reward_to)
        task_specs.append(spec)
        # task_names.append("explore_" + str(dist) + "m")

    return task_specs, task_names


def _create_testing_task_list():
    """
    Manually generate a list of tasks used for testing.
    """
    EVENT_NUMBER_GOAL = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 30, 50]
    INFREQUENT_GOAL = list(range(1, 10))
    STAY_ALIVE_GOAL = [50, 100, 150, 200, 300, 500]
    TEAM_NUMBER_GOAL = [10, 20, 30, 50, 70, 100]
    LEVEL_GOAL = list(range(1, 10))  # TODO: get config
    AGENT_NUM_GOAL = [1]  # competition team size: 8
    ITEM_NUM_GOAL = AGENT_NUM_GOAL
    TEAM_ITEM_GOAL = [1, 3, 5, 7, 10, 15, 20]
    SKILLS = e.combat_skills + e.harvest_skills
    COMBAT_STYLE = e.combat_skills
    ALL_ITEM = i.armour + i.weapons + i.tools + i.ammunition + i.consumables
    EQUIP_ITEM = i.armour + i.weapons + i.tools + i.ammunition
    HARVEST_ITEM = i.weapons + i.ammunition + i.consumables

    """ task_specs is a list of tuple (reward_to, predicate class, kwargs)

        each tuple in the task_specswill create tasks for a team in teams

        reward_to: must be in ['team', 'agent']
        * 'team' create a single team task, in which all team members get rewarded
        * 'agent' create a task for each agent, in which only the agent gets rewarded

        predicate class from the base predicates or custom predicates like above

        kwargs are the additional args that go into predicate. There are also special keys
        * 'target' must be ['left_team', 'right_team', 'left_team_leader', 'right_team_leader']
            these str will be translated into the actual agent ids
        * 'task_cls' is optional. If not provided, the standard Task is used. """
    task_specs = []

    # explore, eat, drink, attack any agent, harvest any item, level up any skill
    #   which can happen frequently
    essential_skills = [
        "GO_FARTHEST",
        "EAT_FOOD",
        "DRINK_WATER",
        "SCORE_HIT",
        "HARVEST_ITEM",
        "LEVEL_UP",
    ]
    for event_code in essential_skills:
        task_specs += [
            ("agent", bp.CountEvent, {"event": event_code, "N": cnt})
            for cnt in EVENT_NUMBER_GOAL
        ]

    # item/market skills, which happen less frequently or should not do too much
    item_skills = [
        "CONSUME_ITEM",
        "GIVE_ITEM",
        "DESTROY_ITEM",
        "EQUIP_ITEM",
        "GIVE_GOLD",
        "LIST_ITEM",
        "EARN_GOLD",
        "BUY_ITEM",
    ]
    for event_code in item_skills:
        task_specs += [
            ("agent", bp.CountEvent, {"event": event_code, "N": cnt}) for cnt in INFREQUENT_GOAL
        ]  # less than 10

    # find resource tiles
    for resource in Harvestable:
        for reward_to in ["agent", "team"]:
            task_specs.append((reward_to, bp.CanSeeTile, {"tile_type": resource}))

    # stay alive ... like ... for 300 ticks
    # i.e., getting incremental reward for each tick alive as an individual or a team
    for reward_to in ["agent", "team"]:
        for num_tick in STAY_ALIVE_GOAL:
            task_specs.append((reward_to, bp.TickGE, {"num_tick": num_tick}))

    # protect the leader: get reward for each tick the leader is alive
    task_specs.append(
        ("team", bp.StayAlive, {"target": "my_team_leader", "task_cls": OngoingTask})
    )

    # want the other team or team leader to die
    for target in ["left_team", "left_team_leader", "right_team", "right_team_leader"]:
        task_specs.append(("team", bp.AllDead, {"target": target}))

    # occupy the center tile, assuming the Medium map size
    # TODO: it'd be better to have some intermediate targets toward the center
    for reward_to in ["agent", "team"]:
        task_specs.append(
            (reward_to, bp.OccupyTile, {"row": 80, "col": 80})
        )  # TODO: get config

    # form a tight formation, for a certain number of ticks
    def PracticeFormation(gs, subject, dist, num_tick):
        return bp.AllMembersWithinRange(gs, subject, dist) * bp.TickGE(gs, subject, num_tick)

    for dist in [1, 3, 5, 10]:
        task_specs += [
            ("team", PracticeFormation, {"dist": dist, "num_tick": num_tick})
            for num_tick in STAY_ALIVE_GOAL
        ]

    # find the other team leader
    for reward_to in ["agent", "team"]:
        for target in ["left_team_leader", "right_team_leader"]:
            task_specs.append((reward_to, bp.CanSeeAgent, {"target": target}))

    # find the other team (any agent)
    for reward_to in ["agent"]:  # , 'team']:
        for target in ["left_team", "right_team"]:
            task_specs.append((reward_to, bp.CanSeeGroup, {"target": target}))

    # explore the map -- sum the l-inf distance traveled by all subjects
    for dist in [10, 20, 30, 50, 100]:  # each agent
        task_specs.append(("agent", bp.DistanceTraveled, {"dist": dist}))
    for dist in [30, 50, 70, 100, 150, 200, 300, 500]:  # summed over all team members
        task_specs.append(("team", bp.DistanceTraveled, {"dist": dist}))

    # level up a skill
    for skill in SKILLS:
        for level in LEVEL_GOAL:
            # since this is an agent task, num_agent must be 1
            task_specs.append(
                ("agent", bp.AttainSkill, {"skill": skill, "level": level, "num_agent": 1})
            )

    # make attain skill a team task by varying the number of agents
    for skill in SKILLS:
        for level in LEVEL_GOAL:
            for num_agent in AGENT_NUM_GOAL:
                if level + num_agent <= 6 or num_agent == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.AttainSkill,
                            {"skill": skill, "level": level, "num_agent": num_agent},
                        )
                    )

    # practice specific combat style
    for style in COMBAT_STYLE:
        for cnt in EVENT_NUMBER_GOAL:
            task_specs.append(("agent", bp.ScoreHit, {"combat_style": style, "N": cnt}))
        for cnt in TEAM_NUMBER_GOAL:
            task_specs.append(("team", bp.ScoreHit, {"combat_style": style, "N": cnt}))

    # defeat agents of a certain level as a team
    for agent_type in ["player", "npc"]:  # c.AGENT_TYPE_CONSTRAINT
        for level in LEVEL_GOAL:
            for num_agent in AGENT_NUM_GOAL:
                if level + num_agent <= 6 or num_agent == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.DefeatEntity,
                            {"agent_type": agent_type, "level": level, "num_agent": num_agent},
                        )
                    )

    # hoarding gold -- evaluated on the current gold
    for amount in EVENT_NUMBER_GOAL:
        task_specs.append(("agent", bp.HoardGold, {"amount": amount}))
    for amount in TEAM_NUMBER_GOAL:
        task_specs.append(("team", bp.HoardGold, {"amount": amount}))

    # earning gold -- evaluated on the total gold earned by selling items
    # does NOT include looted gold
    for amount in EVENT_NUMBER_GOAL:
        task_specs.append(("agent", bp.EarnGold, {"amount": amount}))
    for amount in TEAM_NUMBER_GOAL:
        task_specs.append(("team", bp.EarnGold, {"amount": amount}))

    # spending gold, by buying items
    for amount in EVENT_NUMBER_GOAL:
        task_specs.append(("agent", bp.SpendGold, {"amount": amount}))
    for amount in TEAM_NUMBER_GOAL:
        task_specs.append(("team", bp.SpendGold, {"amount": amount}))

    # making profits by trading -- only buying and selling are counted
    for amount in EVENT_NUMBER_GOAL:
        task_specs.append(("agent", bp.MakeProfit, {"amount": amount}))
    for amount in TEAM_NUMBER_GOAL:
        task_specs.append(("team", bp.MakeProfit, {"amount": amount}))

    # managing inventory space
    def PracticeInventoryManagement(gs, subject, space, num_tick):
        return bp.InventorySpaceGE(gs, subject, space) * bp.TickGE(gs, subject, num_tick)

    for space in [2, 4, 8]:
        task_specs += [
            ("agent", PracticeInventoryManagement, {"space": space, "num_tick": num_tick})
            for num_tick in STAY_ALIVE_GOAL
        ]

    # own item, evaluated on the current inventory
    for item in ALL_ITEM:
        for level in LEVEL_GOAL:
            # agent task
            for quantity in ITEM_NUM_GOAL:
                if level + quantity <= 6 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "agent",
                            bp.OwnItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

            # team task
            for quantity in TEAM_ITEM_GOAL:
                if level + quantity <= 10 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.OwnItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

    # equip item, evaluated on the current inventory and equipment status
    for item in EQUIP_ITEM:
        for level in LEVEL_GOAL:
            # agent task
            task_specs.append(
                ("agent", bp.EquipItem, {"item": item, "level": level, "num_agent": 1})
            )

            # team task
            for num_agent in AGENT_NUM_GOAL:
                if level + num_agent <= 6 or num_agent == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.EquipItem,
                            {"item": item, "level": level, "num_agent": num_agent},
                        )
                    )

    # consume items (ration, potion), evaluated based on the event log
    for item in i.consumables:
        for level in LEVEL_GOAL:
            # agent task
            for quantity in ITEM_NUM_GOAL:
                if level + quantity <= 6 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "agent",
                            bp.ConsumeItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

            # team task
            for quantity in TEAM_ITEM_GOAL:
                if level + quantity <= 10 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.ConsumeItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

    # harvest items, evaluated based on the event log
    for item in HARVEST_ITEM:
        for level in LEVEL_GOAL:
            # agent task
            for quantity in ITEM_NUM_GOAL:
                if level + quantity <= 6 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "agent",
                            bp.HarvestItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

            # team task
            for quantity in TEAM_ITEM_GOAL:
                if level + quantity <= 10 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.HarvestItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

    # list items, evaluated based on the event log
    for item in ALL_ITEM:
        for level in LEVEL_GOAL:
            # agent task
            for quantity in ITEM_NUM_GOAL:
                if level + quantity <= 6 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "agent",
                            bp.ListItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

            # team task
            for quantity in TEAM_ITEM_GOAL:
                if level + quantity <= 10 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.ListItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

    # buy items, evaluated based on the event log
    for item in ALL_ITEM:
        for level in LEVEL_GOAL:
            # agent task
            for quantity in ITEM_NUM_GOAL:
                if level + quantity <= 6 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "agent",
                            bp.BuyItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

            # team task
            for quantity in TEAM_ITEM_GOAL:
                if level + quantity <= 10 or quantity == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.BuyItem,
                            {"item": item, "level": level, "quantity": quantity},
                        )
                    )

    # fully armed, evaluated based on the current player/inventory status
    for style in COMBAT_STYLE:
        for level in LEVEL_GOAL:
            for num_agent in AGENT_NUM_GOAL:
                if level + num_agent <= 6 or num_agent == 1:  # heuristic prune
                    task_specs.append(
                        (
                            "team",
                            bp.FullyArmed,
                            {"combat_style": style, "level": level, "num_agent": num_agent},
                        )
                    )

    packaged_task_specs = []
    for spec in task_specs:
        reward_to = spec[0]
        eval_fn = spec[1]
        eval_fn_kwargs = spec[2]
        packaged_task_specs.append(
            task_spec.TaskSpec(eval_fn, eval_fn_kwargs, reward_to=reward_to)
        )

    return packaged_task_specs


class SyllabusTaskWrapper(PettingZooTaskWrapper):
    """
    Wrapper to handle tasks for the Neural MMO environment.
    """

    # task_space = TaskSpace((18, 200), [tuple(np.arange(18)), tuple(np.arange(200))])
    task_space = TaskSpace(50)

    # task_space = TaskSpace((2719, 200), [tuple(np.arange(2719)), tuple(np.arange(200))])

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

        self.task_list = self.sequential_task_list()
        # self.task_list, task_names = self.create_manual_task_list()
        # self.task_list = self._reformat_tasks(self.task_list)
        self.task_space = SyllabusTaskWrapper.task_space
        self.task = None
        self._task_index = None
        self.task_fn = None

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        new_task = kwargs.pop("new_task", None)
        if new_task is not None:
            self.change_task(new_task)
            task = new_task
            self.task = new_task
            new_task_specs = self.task_list[task]
            self.task_fn = task_spec.make_task_from_spec(
                self.env.possible_agents, [new_task_specs] * len(self.env.possible_agents)
            )
            if seed is not None:
                self.env.seed(int(seed))
        if seed is not None:
            obs, info = self.env.reset(
                seed=int(seed),
                make_task_fn=(lambda: self.task_fn) if self.task_fn is not None else None,
                **kwargs,
            )
        else:
            obs, info = self.env.reset(
                make_task_fn=(lambda: self.task_fn) if self.task_fn is not None else None, **kwargs
            )

        return self.observation(obs), info

    def change_task(self, new_task):
        pass

    def step(self, action):
        obs, rew, terms, truncs, info = self.env.step(action)
        # obs[1]["Task"] = self._task_index
        return self.observation(obs), rew, terms, truncs, info

    def action_space(self, agent):
        """Implement Neural MMO's action_space method."""
        return self.env.action_space(agent)


class SyllabusMinigameTaskWrapper(PettingZooTaskWrapper):
    """
    Wrapper to handle tasks for the Neural MMO environment.
    """

    task_space = TaskSpace(5)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

        self.task_list = [
            TeamBattle,
            RacetoCenter,
            EasyKingoftheHill,
            Sandwich,
            AmmoTraining,
        ]
        self.task_space = SyllabusMinigameTaskWrapper.task_space
        self.task = None
        self._task_index = None
        self.task_fn = None

    def reset(self, **kwargs):
        new_task = kwargs.pop("new_task", None)
        if new_task is not None:
            self.task = new_task
        
        game = self.task_list[self.task]

        seed = kwargs.pop("seed", None)
        if seed is not None:
            obs, info = self.env.reset(
                seed=int(seed),
                game=game,
                **kwargs,
            )
        else:
            obs, info = self.env.reset(
                game=game, **kwargs
            )

        return self.observation(obs), info

    def change_task(self, new_task):
        pass

    def step(self, action):
        obs, rew, terms, truncs, info = self.env.step(action)
        return self.observation(obs), rew, terms, truncs, info

    def action_space(self, agent):
        """Implement Neural MMO's action_space method."""
        return self.env.action_space(agent)