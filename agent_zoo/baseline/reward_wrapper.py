import numpy as np

from nmmo.entity.entity import EntityState

from reinforcement_learning.team_wrapper import TeamWrapper

EntityAttr = EntityState.State.attr_name_to_col
SKILL_LIST = ["melee", "range", "mage", "fishing", "herbalism", "prospecting", "carving", "alchemy"]


class RewardWrapper(TeamWrapper):
    def __init__(
        # BaseStatWrapper args
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        use_custom_reward=True,
        # TeamWrapper args
        augment_obs=True,
        # Custom reward wrapper args
        game_lost_penalty=-1.0,
        game_won_reward=None,
        hp_bonus_weight=0,
        exp_bonus_weight=0,
        defense_bonus_weight=0,
        attack_bonus_weight=0,
        gold_bonus_weight=0,
        nontask_bonus_scale=1,
    ):
        super().__init__(
            env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward, augment_obs
        )
        self.stat_prefix = stat_prefix
        self.game_lost_penalty = game_lost_penalty
        self.game_won_reward = game_won_reward  # if None, use the default game winning score

        self.hp_bonus_weight = hp_bonus_weight
        self.exp_bonus_weight = exp_bonus_weight
        self.defense_bonus_weight = defense_bonus_weight
        self.attack_bonus_weight = attack_bonus_weight
        self.gold_bonus_weight = gold_bonus_weight
        self.nontask_bonus_scale = nontask_bonus_scale

    def reset(self, **kwargs):
        """Called at the start of each episode"""
        self._reset_reward_vars()
        return super().reset(**kwargs)

    def _reset_reward_vars(self):
        self._data = {
            agent_id: {
                "hp": 100,
                "exp": 0,
                "damage_received": 0,
                "damage_inflicted": 0,
                "gold": 0,
            }
            for agent_id in self.env.possible_agents
        }

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        # Handle game over
        if self.env.game.is_over:
            if self.env.game.winners and agent_id in self.env.game.winners:
                reward = self.game_won_reward or self.env.game.winning_score
                truncated = True
            else:
                reward = self.game_lost_penalty
                terminated = True

        # Non-task rewards
        if not (terminated or truncated) and self.nontask_bonus_scale != 0:
            assert agent_id in self.env.realm.players, f"agent_id {agent_id} not in realm.players"
            agent_info = self.env.realm.players[agent_id]

            # HP bonus
            current_hp = agent_info.health.val
            hp_bonus = (current_hp - self._data[agent_id]["hp"]) * self.hp_bonus_weight
            self._data[agent_id]["hp"] = current_hp

            # Experience bonus
            current_exps = np.array(
                [getattr(agent_info, f"{skill}_exp").val for skill in SKILL_LIST]
            )
            current_exp = np.max(current_exps)
            exp_bonus = (current_exp - self._data[agent_id]["exp"]) * self.exp_bonus_weight
            assert exp_bonus >= 0, "exp bonus error"
            self._data[agent_id]["exp"] = current_exp

            # Defense bonus
            current_damage_received = agent_info.history.damage_received
            equipment = agent_info.inventory.equipment
            defense = (
                equipment.melee_defense + equipment.range_defense + equipment.mage_defense
            ) / (15 * 3)
            defense_bonus = self.defense_bonus_weight * defense
            self._data[agent_id]["damage_received"] = current_damage_received

            # Attack bonus
            current_damage_inflicted = agent_info.history.damage_inflicted
            attack_bonus = (
                current_damage_inflicted - self._data[agent_id]["damage_inflicted"]
            ) * self.attack_bonus_weight
            assert attack_bonus >= 0, "attack bonus error"
            self._data[agent_id]["damage_inflicted"] = current_damage_inflicted

            # Gold bonus
            current_gold = agent_info.gold.val
            gold_bonus = (current_gold - self._data[agent_id]["gold"]) * self.gold_bonus_weight
            self._data[agent_id]["gold"] = current_gold

            reward += (
                hp_bonus + exp_bonus + defense_bonus + attack_bonus + gold_bonus
            ) * self.nontask_bonus_scale

        return reward, terminated, truncated, info
