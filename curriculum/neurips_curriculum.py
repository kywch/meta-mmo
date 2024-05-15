"""2023 NeurIPS competition training and eval tasks"""

from typing import List

import nmmo.lib.material as m
import nmmo.systems.item as Item
import nmmo.systems.skill as Skill
from nmmo.task.base_predicates import (
    AttainSkill,
    BuyItem,
    CanSeeAgent,
    CanSeeGroup,
    CanSeeTile,
    ConsumeItem,
    CountEvent,
    DefeatEntity,
    EarnGold,
    EquipItem,
    FullyArmed,
    GainExperience,
    HarvestItem,
    HoardGold,
    InventorySpaceGE,
    ListItem,
    MakeProfit,
    OccupyTile,
    OwnItem,
    ScoreHit,
    SpendGold,
    TickGE,
)
from nmmo.task.task_spec import TaskSpec, check_task_spec

SKILLS = Skill.COMBAT_SKILL + Skill.HARVEST_SKILL
COMBAT_STYLE = Skill.COMBAT_SKILL
ALL_ITEM = Item.ALL_ITEM
EQUIP_ITEM = Item.ARMOR + Item.WEAPON + Item.TOOL + Item.AMMUNITION
HARVEST_ITEM = Item.WEAPON + Item.AMMUNITION + Item.CONSUMABLE
TOOL_FOR_SKILL = {
    Skill.Melee: Item.Spear,
    Skill.Range: Item.Bow,
    Skill.Mage: Item.Wand,
    Skill.Fishing: Item.Rod,
    Skill.Herbalism: Item.Gloves,
    Skill.Carving: Item.Axe,
    Skill.Prospecting: Item.Pickaxe,
    Skill.Alchemy: Item.Chisel,
}

# Constants for the training tasks
EVENT_NUMBER_GOAL = [1, 2, 3, 5, 7, 9, 12, 15, 20, 30, 50]
INFREQUENT_GOAL = list(range(1, 10))
STAY_ALIVE_GOAL = EXP_GOAL = [50, 100, 150, 200, 300, 500, 700]
LEVEL_GOAL = list(range(2, 10))  # TODO: get config
AGENT_NUM_GOAL = ITEM_NUM_GOAL = [1, 2, 3, 4, 5]  # competition team size: 8

# Constants for the eval tasks
EVAL_EVENT_GOAL = 20
EVAL_LEVEL_GOAL = [1, 3]
EVAL_GOLD_GOAL = 100


curriculum: List[TaskSpec] = []

"""Training tasks"""

# The default task: stay alive for the whole episode
curriculum.append(TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": 1024}))

# explore, eat, drink, attack any agent, harvest any item, level up any skill
#   which can happen frequently
most_essentials = [
    "EAT_FOOD",
    "DRINK_WATER",
]
for event_code in most_essentials:
    for cnt in range(1, 10):
        curriculum.append(
            TaskSpec(
                eval_fn=CountEvent,
                eval_fn_kwargs={"event": event_code, "N": cnt},
                sampling_weight=100,
            )
        )

essential_skills = [
    "SCORE_HIT",
    "PLAYER_KILL",
    "HARVEST_ITEM",
    "EQUIP_ITEM",
    "CONSUME_ITEM",
    "LEVEL_UP",
    "EARN_GOLD",
    "LIST_ITEM",
    "BUY_ITEM",
]
for event_code in essential_skills:
    for cnt in EVENT_NUMBER_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=CountEvent,
                eval_fn_kwargs={"event": event_code, "N": cnt},
                sampling_weight=20,
            )
        )

# item/market skills, which happen less frequently or should not do too much
item_skills = [
    "GIVE_ITEM",
    "DESTROY_ITEM",
    "GIVE_GOLD",
]
for event_code in item_skills:
    curriculum += [
        TaskSpec(eval_fn=CountEvent, eval_fn_kwargs={"event": event_code, "N": cnt})
        for cnt in INFREQUENT_GOAL
    ]  # less than 10

# find resource tiles
for resource in m.Harvestable:
    curriculum.append(
        TaskSpec(
            eval_fn=CanSeeTile,
            eval_fn_kwargs={"tile_type": resource},
            sampling_weight=10,
        )
    )


# practice particular skill with a tool
def PracticeSkillWithTool(gs, subject, skill, exp):
    return 0.3 * EquipItem(
        gs, subject, item=TOOL_FOR_SKILL[skill], level=1, num_agent=1
    ) + 0.7 * GainExperience(gs, subject, skill, exp, num_agent=1)


for skill in SKILLS:
    # level up a skill
    for level in LEVEL_GOAL[1:]:
        # since this is an agent task, num_agent must be 1
        curriculum.append(
            TaskSpec(
                eval_fn=AttainSkill,
                eval_fn_kwargs={"skill": skill, "level": level, "num_agent": 1},
                sampling_weight=10 * (6 - level) if level < 6 else 5,
            )
        )

    # gain experience on particular skill
    for exp in EXP_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=PracticeSkillWithTool,
                eval_fn_kwargs={"skill": skill, "exp": exp},
                sampling_weight=50,
            )
        )

# stay alive ... like ... for 300 ticks
# i.e., getting incremental reward for each tick alive as an individual or a team
for num_tick in STAY_ALIVE_GOAL:
    curriculum.append(TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": num_tick}))

# occupy the center tile, assuming the Medium map size
# TODO: it'd be better to have some intermediate targets toward the center
curriculum.append(TaskSpec(eval_fn=OccupyTile, eval_fn_kwargs={"row": 80, "col": 80}))

# find the other team leader
for target in ["left_team_leader", "right_team_leader"]:
    curriculum.append(TaskSpec(eval_fn=CanSeeAgent, eval_fn_kwargs={"target": target}))

# find the other team (any agent)
for target in ["left_team", "right_team"]:
    curriculum.append(TaskSpec(eval_fn=CanSeeGroup, eval_fn_kwargs={"target": target}))

# practice specific combat style
for style in COMBAT_STYLE:
    for cnt in EVENT_NUMBER_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=ScoreHit,
                eval_fn_kwargs={"combat_style": style, "N": cnt},
                sampling_weight=5,
            )
        )

# hoarding gold -- evaluated on the current gold
for amount in EVENT_NUMBER_GOAL:
    curriculum.append(
        TaskSpec(eval_fn=HoardGold, eval_fn_kwargs={"amount": amount}, sampling_weight=10)
    )

# earning gold -- evaluated on the total gold earned by selling items
for amount in EVENT_NUMBER_GOAL:
    curriculum.append(
        TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={"amount": amount}, sampling_weight=10)
    )

# spending gold, by buying items
for amount in EVENT_NUMBER_GOAL:
    curriculum.append(
        TaskSpec(eval_fn=SpendGold, eval_fn_kwargs={"amount": amount}, sampling_weight=5)
    )

# making profits by trading -- only buying and selling are counted
for amount in EVENT_NUMBER_GOAL:
    curriculum.append(
        TaskSpec(eval_fn=MakeProfit, eval_fn_kwargs={"amount": amount}, sampling_weight=3)
    )


# managing inventory space
def PracticeInventoryManagement(gs, subject, space, num_tick):
    return InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)


for space in [2, 4, 8]:
    curriculum += [
        TaskSpec(
            eval_fn=PracticeInventoryManagement,
            eval_fn_kwargs={"space": space, "num_tick": num_tick},
        )
        for num_tick in STAY_ALIVE_GOAL
    ]

# own item, evaluated on the current inventory
for item in ALL_ITEM:
    for level in LEVEL_GOAL:
        # agent task
        for quantity in ITEM_NUM_GOAL:
            if level + quantity <= 6 or quantity == 1:  # heuristic prune
                curriculum.append(
                    TaskSpec(
                        eval_fn=OwnItem,
                        eval_fn_kwargs={
                            "item": item,
                            "level": level,
                            "quantity": quantity,
                        },
                        sampling_weight=4 - level if level < 4 else 1,
                    )
                )

# equip item, evaluated on the current inventory and equipment status
for item in EQUIP_ITEM:
    for level in LEVEL_GOAL:
        # agent task
        curriculum.append(
            TaskSpec(
                eval_fn=EquipItem,
                eval_fn_kwargs={"item": item, "level": level, "num_agent": 1},
                sampling_weight=4 - level if level < 4 else 1,
            )
        )

# consume items (ration, potion), evaluated based on the event log
for item in Item.CONSUMABLE:
    for level in LEVEL_GOAL:
        # agent task
        for quantity in ITEM_NUM_GOAL:
            if level + quantity <= 6 or quantity == 1:  # heuristic prune
                curriculum.append(
                    TaskSpec(
                        eval_fn=ConsumeItem,
                        eval_fn_kwargs={
                            "item": item,
                            "level": level,
                            "quantity": quantity,
                        },
                        sampling_weight=4 - level if level < 4 else 1,
                    )
                )

# harvest items, evaluated based on the event log
for item in HARVEST_ITEM:
    for level in LEVEL_GOAL:
        # agent task
        for quantity in ITEM_NUM_GOAL:
            if level + quantity <= 6 or quantity == 1:  # heuristic prune
                curriculum.append(
                    TaskSpec(
                        eval_fn=HarvestItem,
                        eval_fn_kwargs={
                            "item": item,
                            "level": level,
                            "quantity": quantity,
                        },
                        sampling_weight=4 - level if level < 4 else 1,
                    )
                )

# list items, evaluated based on the event log
for item in ALL_ITEM:
    for level in LEVEL_GOAL:
        # agent task
        for quantity in ITEM_NUM_GOAL:
            if level + quantity <= 6 or quantity == 1:  # heuristic prune
                curriculum.append(
                    TaskSpec(
                        eval_fn=ListItem,
                        eval_fn_kwargs={
                            "item": item,
                            "level": level,
                            "quantity": quantity,
                        },
                        sampling_weight=4 - level if level < 4 else 1,
                    )
                )

# buy items, evaluated based on the event log
for item in ALL_ITEM:
    for level in LEVEL_GOAL:
        # agent task
        for quantity in ITEM_NUM_GOAL:
            if level + quantity <= 6 or quantity == 1:  # heuristic prune
                curriculum.append(
                    TaskSpec(
                        eval_fn=BuyItem,
                        eval_fn_kwargs={
                            "item": item,
                            "level": level,
                            "quantity": quantity,
                        },
                        sampling_weight=4 - level if level < 4 else 1,
                    )
                )

"""Eval tasks"""

# Survive to the end
curriculum.append(TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": 1024}))

# Kill 20 players/npcs
curriculum.append(
    TaskSpec(
        eval_fn=CountEvent,
        eval_fn_kwargs={"event": "PLAYER_KILL", "N": EVAL_EVENT_GOAL},
        tags=["eval"],
    )
)

# Kill npcs of level 1+, 3+
for level in EVAL_LEVEL_GOAL:
    curriculum.append(
        TaskSpec(
            eval_fn=DefeatEntity,
            eval_fn_kwargs={"agent_type": "npc", "level": level, "num_agent": EVAL_EVENT_GOAL},
            tags=["eval"],
        )
    )

# Explore and reach the center (80, 80)
curriculum.append(
    TaskSpec(eval_fn=CountEvent, eval_fn_kwargs={"event": "GO_FARTHEST", "N": 64}, tags=["eval"])
)

curriculum.append(
    TaskSpec(eval_fn=OccupyTile, eval_fn_kwargs={"row": 80, "col": 80}, tags=["eval"])
)

# Reach skill level 10
for skill in SKILLS:
    curriculum.append(
        TaskSpec(
            eval_fn=AttainSkill,
            eval_fn_kwargs={"skill": skill, "level": 10, "num_agent": 1},
            tags=["eval"],
        )
    )

# Harvest 20 ammos of level 1+ or 3+
for ammo in Item.AMMUNITION:
    for level in EVAL_LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=HarvestItem,
                eval_fn_kwargs={"item": ammo, "level": level, "quantity": EVAL_EVENT_GOAL},
                tags=["eval"],
            )
        )

# Consume 10 ration/potions of level 1+ or 3+
for item in Item.CONSUMABLE:
    for level in EVAL_LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=ConsumeItem,
                eval_fn_kwargs={"item": item, "level": level, "quantity": EVAL_EVENT_GOAL},
                tags=["eval"],
            )
        )

# Equip armour, weapons, tools, and ammos
for item in EQUIP_ITEM:
    for level in EVAL_LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=EquipItem,
                eval_fn_kwargs={"item": item, "level": level, "num_agent": 1},
                tags=["eval"],
            )
        )

# Fully armed, level 1+ or 3+
for skill in Skill.COMBAT_SKILL:
    for level in EVAL_LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=FullyArmed,
                eval_fn_kwargs={"combat_style": skill, "level": level, "num_agent": 1},
                tags=["eval"],
            )
        )

# Buy and Sell 10 items (of any kind)
curriculum.append(
    TaskSpec(
        eval_fn=CountEvent,
        eval_fn_kwargs={"event": "EARN_GOLD", "N": EVAL_EVENT_GOAL},  # item sold
        tags=["eval"],
    )
)

curriculum.append(
    TaskSpec(
        eval_fn=CountEvent,
        eval_fn_kwargs={"event": "BUY_ITEM", "N": EVAL_EVENT_GOAL},  # item bought
        tags=["eval"],
    )
)

# Earn 100 gold (revenue), just by trading
curriculum.append(
    TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={"amount": EVAL_GOLD_GOAL}, tags=["eval"])
)

# Own and protect 100 gold by any means (looting or trading)
curriculum.append(
    TaskSpec(eval_fn=HoardGold, eval_fn_kwargs={"amount": EVAL_GOLD_GOAL}, tags=["eval"])
)

# Make profit of 100 gold by any means
curriculum.append(
    TaskSpec(eval_fn=MakeProfit, eval_fn_kwargs={"amount": EVAL_GOLD_GOAL}, tags=["eval"])
)


if __name__ == "__main__":
    import multiprocessing as mp
    from contextlib import contextmanager

    import dill
    import numpy as np
    import psutil

    @contextmanager
    def create_pool(num_proc):
        pool = mp.Pool(processes=num_proc)
        yield pool
        pool.close()
        pool.join()

    # divide the specs into chunks
    num_workers = round(psutil.cpu_count(logical=False) * 0.7)
    spec_chunks = np.array_split(curriculum, num_workers)
    with create_pool(num_workers) as pool:
        pool.map(check_task_spec, spec_chunks)

    # test if the task spec is pickalable
    with open("pickle_test.pkl", "wb") as f:
        dill.dump(curriculum, f)
