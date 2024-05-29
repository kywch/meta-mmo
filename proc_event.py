import os
import argparse
from collections import defaultdict

import dill
import numpy as np
import polars as pl
from tqdm import tqdm

from nmmo.lib.event_code import EventCode
from nmmo.systems.item import ALL_ITEM
from nmmo.systems.skill import COMBAT_SKILL, HARVEST_SKILL

CODE_TO_EVENT = {v: k for k, v in EventCode.__dict__.items() if not k.startswith("_")}

ITEM_ID_TO_NAME = {item.ITEM_TYPE_ID: item.__name__ for item in ALL_ITEM}

SKILL_ID_TO_NAME = {skill.SKILL_ID: skill.__name__ for skill in COMBAT_SKILL + HARVEST_SKILL}


# event tuple key to string
def event_key_to_str(event_key):
    if event_key[0] == EventCode.LEVEL_UP:
        return f"LEVEL_{SKILL_ID_TO_NAME[event_key[1]]}"

    elif event_key[0] == EventCode.SCORE_HIT:
        return f"ATTACK_NUM_{SKILL_ID_TO_NAME[event_key[1]]}"

    elif event_key[0] in [
        EventCode.HARVEST_ITEM,
        EventCode.CONSUME_ITEM,
        EventCode.EQUIP_ITEM,
        EventCode.LIST_ITEM,
        EventCode.BUY_ITEM,
        EventCode.FIRE_AMMO,
    ]:
        return f"{CODE_TO_EVENT[event_key[0]]}_{ITEM_ID_TO_NAME[event_key[1]]}"

    elif event_key[0] == EventCode.GO_FARTHEST:
        return "3_PROGRESS_TO_CENTER"

    elif event_key[0] == EventCode.AGENT_CULLED:
        return "2_AGENT_LIFESPAN"

    elif event_key[0] == EventCode.PLAYER_KILL:
        target = "NPC" if event_key[1] == 0 else "Agent"
        return f"KILLED_{target}"

    else:
        return CODE_TO_EVENT[event_key[0]]


def extract_policy_name(agent_policy_map, agent_id):
    if len(agent_policy_map) == 0:
        return "learner"

    assert agent_id in agent_policy_map, "Agent id not found in policy map"
    return agent_policy_map[agent_id]


def gather_agent_events_by_policy(data_dir):
    data_by_policy = defaultdict(list)
    data_list = [f for f in os.listdir(data_dir) if f.endswith(".metadata.pkl")]
    for file_name in tqdm(data_list):
        data = dill.load(open(f"{data_dir}/{file_name}", "rb"))
        final_tick = data["tick"]

        agent_policy_map = {}
        map_file = f"{data_dir}/{file_name.split('.metadata.pkl')[0]}.policy_map.pkl"
        if os.path.exists(map_file):
            with open(map_file, "rb") as f:
                agent_policy_map = dill.load(f)

        for agent_id, vals in data["event_stats"].items():
            policy_name = extract_policy_name(agent_policy_map, agent_id)

            # Agent survived until the end
            if (EventCode.AGENT_CULLED,) not in vals:
                vals[(EventCode.AGENT_CULLED,)] = final_tick
            data_by_policy[policy_name].append(vals)

    return data_by_policy


def get_event_stats(policy_name, grouped_data):
    num_agents = len(grouped_data)
    assert num_agents > 0, "There should be at least one agent"

    cnt_attack = 0
    cnt_buy = 0
    cnt_consume = 0
    cnt_equip = 0
    cnt_harvest = 0
    cnt_list = 0
    cnt_fire = 0
    cnt_eat = 0
    cnt_drink = 0
    cnt_kill_agent = 0
    cnt_kill_npc = 0

    results = {"0_NAME": policy_name, "1_COUNT": num_agents}
    event_data = defaultdict(list)
    for data in grouped_data:
        for event, val in data.items():
            event_data[event].append(val)

    total_ticks = 0
    for event, vals in event_data.items():
        if event[0] == EventCode.LEVEL_UP:
            # Base skill level is 1
            vals += [1] * (num_agents - len(vals))
            results[event_key_to_str(event)] = np.mean(vals)  # AVG skill level
        elif event[0] == EventCode.AGENT_CULLED:
            life_span = np.mean(vals)
            total_ticks = sum(vals)
            results["2_AGENT_LIFESPAN_AVG"] = life_span
            results["2_AGENT_LIFESPAN_SD"] = np.std(vals)
        elif event[0] == EventCode.GO_FARTHEST:
            results["3_PROGRESS_TO_CENTER_AVG"] = np.mean(vals)
            results["3_PROGRESS_TO_CENTER_SD"] = np.std(vals)
        else:
            results[event_key_to_str(event)] = sum(vals) / num_agents

        if event[0] == EventCode.SCORE_HIT:
            cnt_attack += sum(vals)
        if event[0] == EventCode.BUY_ITEM:
            cnt_buy += sum(vals)
        if event[0] == EventCode.CONSUME_ITEM:
            cnt_consume += sum(vals)
        if event[0] == EventCode.EQUIP_ITEM:
            cnt_equip += sum(vals)
        if event[0] == EventCode.FIRE_AMMO:
            cnt_fire += sum(vals)
        if event[0] == EventCode.HARVEST_ITEM:
            cnt_harvest += sum(vals)
        if event[0] == EventCode.LIST_ITEM:
            cnt_list += sum(vals)
        if event[0] == EventCode.EAT_FOOD:
            cnt_eat += sum(vals)
        if event[0] == EventCode.DRINK_WATER:
            cnt_drink += sum(vals)
        if event == (EventCode.PLAYER_KILL, 0):
            cnt_kill_npc += sum(vals)
        if event == (EventCode.PLAYER_KILL, 1):
            cnt_kill_agent += sum(vals)

    assert total_ticks > 0, "Total ticks should be greater than 0"

    # These normalized values represent the events per 100 ticks (per agent)
    results["4_NORM_ATTACK"] = 100 * cnt_attack / total_ticks
    results["4_NORM_BUY"] = 100 * cnt_buy / total_ticks
    results["4_NORM_CONSUME"] = 100 * cnt_consume / total_ticks
    results["4_NORM_EQUIP"] = 100 * cnt_equip / total_ticks
    results["4_NORM_FIRE"] = 100 * cnt_fire / total_ticks
    results["4_NORM_HARVEST"] = 100 * cnt_harvest / total_ticks
    results["4_NORM_LIST"] = 100 * cnt_list / total_ticks
    results["4_NORM_EAT"] = 100 * cnt_eat / total_ticks
    results["4_NORM_DRINK"] = 100 * cnt_drink / total_ticks
    results["4_NORM_KILL_NPC"] = 100 * cnt_kill_npc / total_ticks
    results["4_NORM_KILL_AGENT"] = 100 * cnt_kill_agent / total_ticks

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process replay data")
    parser.add_argument("policy_store_dir", type=str, help="Path to the policy directory")
    args = parser.parse_args()

    # Gather the event data by policies, across multiple replays
    data_by_policy = gather_agent_events_by_policy(args.policy_store_dir)

    policy_results = [
        get_event_stats(pol_name, pol_data) for pol_name, pol_data in data_by_policy.items()
    ]

    policy_df = pl.DataFrame(policy_results).fill_null(0).sort("0_NAME")
    policy_df = policy_df.select(sorted(policy_df.columns))
    policy_df.write_csv(
        os.path.join(args.policy_store_dir, "events_by_policy.tsv"),
        separator="\t",
        float_precision=6,
    )

    print("Result file saved as events_by_policy.tsv")
    print("Done.")
