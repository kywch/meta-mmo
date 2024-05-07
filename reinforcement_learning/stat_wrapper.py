from collections import defaultdict
import numpy as np

from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

from nmmo.lib.event_code import EventCode
import nmmo.systems.item as Item
from nmmo.minigames import RacetoCenter, KingoftheHill, Sandwich, RadioRaid

from reinforcement_learning.environment import TeamBattle


class BaseStatWrapper(BaseParallelWrapper):
    def __init__(
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        use_custom_reward=True,
    ):
        super().__init__(env)
        self.env_done = False
        self.early_stop_agent_num = early_stop_agent_num
        self.eval_mode = eval_mode
        self._reset_episode_stats()
        self._stat_prefix = stat_prefix
        self.use_custom_reward = use_custom_reward

        # Stats by each game for the whole training duration
        self.total_agent_steps = defaultdict(int)
        self.total_return = defaultdict(int)

    def observation(self, agent_id, agent_obs):
        """Called before observations are returned from the environment
        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))"""
        return agent_obs

    def action(self, agent_id, agent_atn):
        """Called before actions are passed from the model to the environment"""
        return agent_atn

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        """Called on reward, terminated, truncated, and info before they are returned from the environment
        Use this to define custom reward shaping."""
        return reward, terminated, truncated, info

    @property
    def agents(self):
        return [] if self.env_done else self.env.agents

    def reset(self, **kwargs):
        """Called at the start of each episode"""
        self._reset_episode_stats()
        obs, info = self.env.reset(**kwargs)

        for agent_id in self.env.agents:
            obs[agent_id] = self.observation(agent_id, obs[agent_id])

        # Attach the approx sampling info
        if "stats" not in info:
            info[agent_id]["stats"] = {}
        for game_name in self.total_agent_steps.keys():
            # Record the cumulative ratio of agent steps and return
            # It will give us a good estimation
            info[agent_id]["stats"][f"Sampling/{game_name}_agent_steps"] = self.total_agent_steps[
                game_name
            ] / sum(self.total_agent_steps.values())
            info[agent_id]["stats"][f"Sampling/{game_name}_returns"] = self.total_return[
                game_name
            ] / sum(self.total_return.values())

        return obs, info

    def step(self, action):
        assert len(self.env.agents) > 0, "No agents in the environment"  # xcxc: sanity check

        # Modify actions before they are passed to the environment
        for agent_id in self.env.agents:
            action[agent_id] = self.action(agent_id, action[agent_id])

        obs, rewards, terms, truncs, infos = self.env.step(action)

        # Stop early if there are too few agents generating the training data
        # Also env.agents is empty when the tick reaches the config horizon
        if len(self.env.agents) <= self.early_stop_agent_num or self.env.game.is_over:
            self.env_done = True

        # Modify reward and observation after they are returned from the environment
        agent_list = list(obs.keys())
        for agent_id in agent_list:
            trunc, info = self._process_stats_and_early_stop(
                agent_id, rewards[agent_id], terms[agent_id], truncs[agent_id], infos[agent_id]
            )

            if self.use_custom_reward is True:
                rew, term, trunc, info = self.reward_terminated_truncated_info(
                    agent_id, rewards[agent_id], terms[agent_id], trunc, info
                )
            else:
                # NOTE: Also disable death penalty, which is not from the task
                rew = 0 if terms[agent_id] is True else rewards[agent_id]
                term = terms[agent_id]

            rewards[agent_id] = rew
            terms[agent_id] = term
            truncs[agent_id] = trunc
            infos[agent_id] = info
            obs[agent_id] = self.observation(agent_id, obs[agent_id])

        if self.env_done:
            # Hack: To mark the end of the episode. Only one agent's done flag is enough.
            infos[agent_id]["episode_done"] = True

            game_scores = []
            for task in self.env.tasks:
                for a_id in task.assignee:
                    # Max progress is a float between 0 and 1
                    score = task._max_progress
                    if self.env.game.winners and a_id in self.env.game.winners:
                        # NOTE: this is a hack. Extra +16 for winning
                        # A heuristic to mark the winner policy when calculating ELO
                        score += 16

                    # NOTE: avoiding dict to bypass clean_pufferl's unroll_nested_dict
                    game_scores.append((a_id, score))

            # Hack: Putting the results in only one agent's info
            infos[agent_id]["game_scores"] = game_scores

            game_name = self.env.game.name
            for key, val in self.env.game.get_episode_stats().items():
                info["stats"][game_name + "/" + key] = val
            info["stats"][game_name + "/finished_tick"] = self.env.realm.tick

            if self.env.game.winners:
                info["stats"][game_name + "/winning_score"] = self.env.game.winning_score

            if (
                isinstance(self.env.game, RacetoCenter)
                or isinstance(self.env.game, KingoftheHill)
                or isinstance(self.env.game, Sandwich)
                or isinstance(self.env.game, RadioRaid)
            ):
                info["stats"][game_name + "/game_won"] = self.env.game.winners is not None
                info["stats"][game_name + "/map_size"] = self.env.game.map_size
                max_progress = [task.progress_info["max_progress"] for task in self.env.game.tasks]
                info["stats"][game_name + "/max_progress"] = max(max_progress)
                info["stats"][game_name + "/avg_max_prog"] = sum(max_progress) / len(max_progress)

            if isinstance(self.env.game, KingoftheHill):
                info["stats"][game_name + "/seize_duration"] = self.env.game.seize_duration

            if isinstance(self.env.game, Sandwich):
                info["stats"][game_name + "/inner_npc_num"] = self.env.game.inner_npc_num
                info["stats"][game_name + "/spawned_npc"] = abs(self.env.realm.npcs.next_id + 1)

            if isinstance(self.env.game, RadioRaid):
                info["stats"][game_name + "/goal_num_npc"] = self.env.game.goal_num_npc
                info["stats"][game_name + "/spawned_npc"] = abs(self.env.realm.npcs.next_id + 1)

        return obs, rewards, terms, truncs, infos

    # def reward_done_truncated_info(self, agent_id, reward, don
    def _reset_episode_stats(self):
        self.env_done = False
        self.cum_rewards = {agent_id: 0 for agent_id in self.env.possible_agents}
        self._unique_events = {
            agent_id: {
                "experienced": set(),
                "prev_count": 0,
                "curr_count": 0,
            }
            for agent_id in self.env.possible_agents
        }

    def _process_stats_and_early_stop(self, agent_id, reward, terminated, truncated, info):
        """Update stats + info and save replays."""
        # Remove the task from info. Curriculum info is processed in _update_stats()
        info.pop("task", None)

        # Handle early stopping
        if self.env_done and not terminated:
            truncated = True

        # Count and store unique event counts for easier use
        realm = self.env.realm
        tick_log = realm.event_log.get_data(agents=[agent_id], tick=-1)  # get only the last tick
        uniq = self._unique_events[agent_id]
        uniq["prev_count"] = uniq["curr_count"]
        uniq["curr_count"] += count_unique_events(tick_log, uniq["experienced"])

        if not (terminated or truncated):
            self.cum_rewards[agent_id] += reward
            return truncated, info

        # The agent is terminated or truncated, so recoding the episode stats
        if "stats" not in info:
            info["stats"] = {}

        agent = realm.players.dead_this_tick.get(agent_id, realm.players.get(agent_id))
        assert agent is not None

        # NOTE: this may not be true when players can be resurrected. Check back later
        info["length"] = realm.tick
        info["return"] = self.cum_rewards[agent_id]

        game_name = self.env.game.name
        self.total_agent_steps[game_name] += realm.tick
        self.total_return[game_name] += self.cum_rewards[agent_id]

        # Cause of Deaths
        if terminated:
            info["stats"]["cod/attacked"] = 1.0 if agent.damage.val > 0 else 0.0
            info["stats"]["cod/starved"] = 1.0 if agent.food.val == 0 else 0.0
            info["stats"]["cod/dehydrated"] = 1.0 if agent.water.val == 0 else 0.0
        else:
            info["stats"]["cod/attacked"] = 0
            info["stats"]["cod/starved"] = 0
            info["stats"]["cod/dehydrated"] = 0

        # Task-related stats
        task = self.env.agent_task_map[agent_id][0]  # consider only the first task
        info["stats"]["task/completed"] = 1.0 if task.completed else 0.0
        info["stats"]["task/pcnt_2_reward_signal"] = 1.0 if task.reward_signal_count >= 2 else 0.0
        info["stats"]["task/pcnt_0p2_max_progress"] = 1.0 if task._max_progress >= 0.2 else 0.0
        # info["curriculum"] = {task.spec_name: (task._max_progress, task.reward_signal_count)}

        if self.eval_mode:
            # 'return' is used for ranking in the eval mode, so put the task progress here
            info["return"] = task._max_progress  # this is 1 if done

        # Log the below stats ONLY for the team battle
        if isinstance(self.env.game, TeamBattle):
            # Max combat/harvest level achieved
            info["stats"]["achieved/max_combat_level"] = agent.attack_level
            info["stats"]["achieved/max_harvest_skill_ammo"] = max(
                agent.prospecting_level.val,
                agent.carving_level.val,
                agent.alchemy_level.val,
            )
            info["stats"]["achieved/max_harvest_skill_consum"] = max(
                agent.fishing_level.val,
                agent.herbalism_level.val,
            )

            # Event-based stats
            achieved, performed, _ = process_event_log(realm, [agent_id])
            for key, val in list(achieved.items()) + list(performed.items()):
                info["stats"][key] = float(val)

        if self._stat_prefix:
            info = {self._stat_prefix: info}

        return truncated, info


################################################################################
# Event processing utilities for Neural MMO.

INFO_KEY_TO_EVENT_CODE = {
    "event/" + evt.lower(): val for evt, val in EventCode.__dict__.items() if isinstance(val, int)
}

# convert the numbers into binary (performed or not) for the key events
KEY_EVENT = [
    "eat_food",
    "drink_water",
    "score_hit",
    "player_kill",
    "fire_ammo",
    "consume_item",
    "give_item",
    "destroy_item",
    "harvest_item",
    "give_gold",
    "list_item",
    "buy_item",
]

ITEM_TYPE = {
    "armor": [item.ITEM_TYPE_ID for item in Item.ARMOR],
    "weapon": [item.ITEM_TYPE_ID for item in Item.WEAPON],
    "tool": [item.ITEM_TYPE_ID for item in Item.TOOL],
    "ammo": [item.ITEM_TYPE_ID for item in Item.AMMUNITION],
    "consumable": [item.ITEM_TYPE_ID for item in Item.CONSUMABLE],
}


def process_event_log(realm, agent_list):
    """Process the event log and extract performed actions and achievements."""
    log = realm.event_log.get_data(agents=agent_list)
    attr_to_col = realm.event_log.attr_to_col

    # count the number of events
    event_cnt = {}
    for key, code in INFO_KEY_TO_EVENT_CODE.items():
        # count the freq of each event
        event_cnt[key] = int(sum(log[:, attr_to_col["event"]] == code))

    # record true or false for each event
    performed = {}
    for evt in KEY_EVENT:
        key = "event/" + evt
        performed[key] = event_cnt[key] > 0

    # check if tools, weapons, ammos, ammos were equipped
    for item_type, item_ids in ITEM_TYPE.items():
        if item_type == "consumable":
            continue
        key = "event/equip_" + item_type
        idx = (log[:, attr_to_col["event"]] == EventCode.EQUIP_ITEM) & np.in1d(
            log[:, attr_to_col["item_type"]], item_ids
        )
        performed[key] = sum(idx) > 0

    # check if weapon was harvested
    key = "event/harvest_weapon"
    idx = (log[:, attr_to_col["event"]] == EventCode.HARVEST_ITEM) & np.in1d(
        log[:, attr_to_col["item_type"]], ITEM_TYPE["weapon"]
    )
    performed[key] = sum(idx) > 0

    # record important achievements
    achieved = {}

    # get progress to center
    idx = log[:, attr_to_col["event"]] == EventCode.GO_FARTHEST
    achieved["achieved/max_progress_to_center"] = (
        int(max(log[idx, attr_to_col["distance"]])) if sum(idx) > 0 else 0
    )

    # get earned gold
    idx = log[:, attr_to_col["event"]] == EventCode.EARN_GOLD
    achieved["achieved/earned_gold"] = int(sum(log[idx, attr_to_col["gold"]]))

    # get max damage
    idx = log[:, attr_to_col["event"]] == EventCode.SCORE_HIT
    achieved["achieved/max_damage"] = (
        int(max(log[idx, attr_to_col["damage"]])) if sum(idx) > 0 else 0
    )

    # get max possessed item levels: from harvesting, looting, buying
    idx = np.in1d(
        log[:, attr_to_col["event"]],
        [EventCode.HARVEST_ITEM, EventCode.LOOT_ITEM, EventCode.BUY_ITEM],
    )
    if sum(idx) > 0:
        for item_type, item_ids in ITEM_TYPE.items():
            idx_item = np.in1d(log[idx, attr_to_col["item_type"]], item_ids)
            if sum(idx_item) > 0:
                achieved["achieved/max_" + item_type + "_level"] = int(
                    max(log[idx][idx_item, attr_to_col["level"]])
                )

    # other notable achievements
    idx = log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL
    achieved["achieved/agent_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] > 0)))
    achieved["achieved/npc_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] < 0)))
    achieved["achieved/unique_events"] = count_unique_events(log, set())

    return achieved, performed, event_cnt


# These events are important, so count them even though they are not unique
EVERY_EVENT_TO_COUNT = set([EventCode.PLAYER_KILL, EventCode.EARN_GOLD])


def count_unique_events(tick_log, experienced, every_event_to_count=EVERY_EVENT_TO_COUNT):
    cnt_unique = 0
    if len(tick_log) == 0:
        return cnt_unique

    for row in tick_log[:, 3:6]:  # only taking the event, type, level cols
        event = tuple(row)
        if event not in experienced:
            experienced.add(event)
            cnt_unique += 1

        elif row[0] in every_event_to_count:
            # These events are important, so count them even though they are not unique
            cnt_unique += 1

    return cnt_unique
