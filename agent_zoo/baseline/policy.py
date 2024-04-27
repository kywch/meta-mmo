import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models
import pufferlib.emulation

from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, layer_width=256, num_layers=1):
        # Use the same width for both the input and hidden
        super().__init__(env, policy, layer_width, layer_width, num_layers)


def orthogonal_init(layer, gain=1.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, 0)


class Policy(pufferlib.models.Policy):
    def __init__(self, env, layer_width=256):
        super().__init__(env)

        self.unflatten_context = env.unflatten_context
        self.enabled_systems = env.env.config.enabled_systems

        # NOTE: Models with different task size will not be compatible with each other
        task_size = self.unflatten_context.flat_observation_space["DTask.V"].shape[0]
        tile_attr_dim = self.unflatten_context.flat_observation_space["DTile.V"].shape[1]

        self.tile_encoder = TileEncoder(layer_width, tile_attr_dim)
        self.player_encoder = PlayerEncoder(layer_width)
        self.task_encoder = TaskEncoder(task_size, layer_width)
        self.item_encoder = ItemEncoder(layer_width) if "ITEM" in self.enabled_systems else None
        self.inventory_encoder = (
            InventoryEncoder(layer_width) if "ITEM" in self.enabled_systems else None
        )
        self.market_encoder = (
            MarketEncoder(layer_width) if "EXCHANGE" in self.enabled_systems else None
        )

        num_encoder = 3 + ("ITEM" in self.enabled_systems) + ("EXCHANGE" in self.enabled_systems)
        self.proj_fc = torch.nn.Linear(num_encoder * layer_width, layer_width)

        self.action_decoder = ActionDecoder(layer_width, self.enabled_systems)
        self.value_head = torch.nn.Linear(layer_width, 1)

        orthogonal_init(self.proj_fc)
        orthogonal_init(self.value_head)

    def encode_observations(self, flat_observations):
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            flat_observations, self.unflatten_context
        )

        encoded_obs = []

        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )
        encoded_obs += [tile, my_agent]

        item_embeddings = None
        if "ITEM" in self.enabled_systems:
            item_embeddings = self.item_encoder(env_outputs["Inventory"])
            inventory = self.inventory_encoder(item_embeddings)
            encoded_obs.append(inventory)

        market_embeddings = None
        if "EXCHANGE" in self.enabled_systems:
            market_embeddings = self.item_encoder(env_outputs["Market"])
            market = self.market_encoder(market_embeddings)
            encoded_obs.append(market)

        encoded_obs.append(self.task_encoder(env_outputs["Task"]))

        obs = torch.cat(encoded_obs, dim=-1)
        obs = F.relu(self.proj_fc(obs))

        return obs, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_planes, img_size=(15, 15)):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
            torch.nn.LayerNorm((in_planes, *img_size)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
            torch.nn.LayerNorm((in_planes, *img_size)),
        )

    def forward(self, x):
        out = self.model(x)
        out += x
        return out


class TileEncoder(torch.nn.Module):
    def __init__(self, output_size, tile_attr_dim):
        super().__init__()
        self.tile_attr_dim = tile_attr_dim
        self.type_embedding = torch.nn.Embedding(16, 30)
        self.entity_embedding = torch.nn.Embedding(8, 30)

        self.tile_resnet = ResnetBlock(64)
        self.tile_conv_1 = torch.nn.Conv2d(64, 32, 3)
        self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
        self.tile_fc = torch.nn.Linear(8 * 11 * 11, output_size)
        self.tile_norm = torch.nn.LayerNorm(output_size)
        orthogonal_init(self.tile_fc)

    def forward(self, tile):
        tile_position = tile[:, :, :2] / 128 - 0.5
        tile_type = tile[:, :, 2].long().clip(0, 15)
        tile_cat = [tile_position, self.type_embedding(tile_type)]  # 32

        # NOTE: forward() breaks if the tile_attr_dim is not 6
        if self.tile_attr_dim > 3:
            dist_map = tile[:, :, 3] / 128
            entity_map = tile[:, :, 4].long().clip(0, 8)
            rally_map = tile[:, :, 5]
            tile_cat += [
                dist_map.unsqueeze(-1),
                self.entity_embedding(entity_map),
                rally_map.unsqueeze(-1),
            ]  # 32

        tile = torch.cat(tile_cat, dim=-1)
        agents, _, features = tile.shape
        tile = tile.transpose(1, 2).view(agents, features, 15, 15).float()

        tile = F.relu(self.tile_resnet(tile))
        tile = F.relu(self.tile_conv_1(tile))
        tile = F.relu(self.tile_conv_2(tile))
        tile = tile.contiguous().view(agents, -1)
        tile = F.relu(self.tile_norm(self.tile_fc(tile)))
        return tile


class MLPBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.model = [
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            self.model += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
        self.model.append(torch.nn.Linear(hidden_size, output_size))
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                orthogonal_init(layer)
        self.model = torch.nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        return out


class PlayerEncoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.entity_dim = 31
        self.player_offset = torch.tensor([i * 256 for i in range(self.entity_dim)])
        self.embedding = torch.nn.Embedding(self.entity_dim * 256, 32)

        self.EntityId = EntityState.State.attr_name_to_col["id"]
        self.EntityAttackerId = EntityState.State.attr_name_to_col["attacker_id"]
        self.EntityMessage = EntityState.State.attr_name_to_col["message"]  # Communication token
        self.id_embedding = torch.nn.Embedding(512, 64)
        self.embedding_idx = [self.EntityId, self.EntityAttackerId]
        self.no_embedding_idx = [i for i in range(self.entity_dim)]
        self.no_embedding_idx.remove(self.EntityId)
        self.no_embedding_idx.remove(self.EntityAttackerId)
        # self.no_embedding_idx.remove(self.EntityMessage)  # NOTE: now include teammates' comm tokens

        self.agent_mlp = MLPBlock(64 + self.entity_dim - 2, output_size, output_size)
        self.agent_fc = torch.nn.Linear(output_size, output_size)
        self.my_agent_fc = torch.nn.Linear(output_size, output_size)
        self.agent_norm = torch.nn.LayerNorm(output_size)
        self.my_agent_norm = torch.nn.LayerNorm(output_size)
        orthogonal_init(self.agent_fc)
        orthogonal_init(self.my_agent_fc)

    def forward(self, agents, my_id):
        # Pull out rows corresponding to the agent
        agent_ids = agents[:, :, EntityId]
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(
            mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
        )

        batch, agent, _ = agents.shape
        agent_embeddings = self.embedding(
            (agents[:, :, self.embedding_idx].long() + 256).clip(0, 511)
        ).reshape(batch, agent, -1)
        agent_embeddings = torch.cat(
            (agent_embeddings, agents[:, :, self.no_embedding_idx]), dim=-1
        ).float()
        agent_embeddings = F.relu(self.agent_mlp(agent_embeddings))

        my_agent_embeddings = agent_embeddings[torch.arange(agents.shape[0]), row_indices]
        agent_embeddings = F.relu(self.agent_norm(self.agent_fc(agent_embeddings)))
        my_agent_embeddings = F.relu(self.my_agent_norm(self.my_agent_fc(my_agent_embeddings)))
        return agent_embeddings, my_agent_embeddings


class ItemEncoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 32)
        self.item_mlp = MLPBlock(2 * 32 + 12, output_size, output_size)
        self.item_norm = torch.nn.LayerNorm(output_size)

        self.discrete_idxs = [1, 14]
        self.discrete_offset = torch.Tensor([2, 0])
        self.continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
        self.continuous_scale = torch.Tensor(
            [
                1 / 10,
                1 / 10,
                1 / 10,
                1 / 100,
                1 / 100,
                1 / 100,
                1 / 40,
                1 / 40,
                1 / 40,
                1 / 100,
                1 / 100,
                1 / 100,
            ]
        )

    def forward(self, items):
        if self.discrete_offset.device != items.device:
            self.discrete_offset = self.discrete_offset.to(items.device)
            self.continuous_scale = self.continuous_scale.to(items.device)

        # Embed each feature separately
        discrete = items[:, :, self.discrete_idxs] + self.discrete_offset
        discrete = self.embedding(discrete.long().clip(0, 255))
        batch, item, attrs, embed = discrete.shape
        discrete = discrete.view(batch, item, attrs * embed)

        continuous = items[:, :, self.continuous_idxs] / self.continuous_scale

        item_embeddings = torch.cat([discrete, continuous], dim=-1).float()
        item_embeddings = F.relu(self.item_norm(self.item_mlp(item_embeddings)))
        return item_embeddings


class InventoryEncoder(torch.nn.Module):
    def __init__(self, output_size, inventory_size=12):
        super().__init__()
        self.fc = torch.nn.Linear(inventory_size * output_size, output_size)
        self.norm = torch.nn.LayerNorm(output_size)
        orthogonal_init(self.fc)

    def forward(self, inventory):
        agents, items, hidden = inventory.shape
        inventory = inventory.view(agents, items * hidden)
        return F.relu(self.norm(self.fc(inventory)))


class MarketEncoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc = torch.nn.Linear(output_size, output_size)
        self.norm = torch.nn.LayerNorm(output_size)
        orthogonal_init(self.fc)

    def forward(self, market):
        return F.relu(self.norm(self.fc(market).mean(-2)))


class TaskEncoder(torch.nn.Module):
    def __init__(self, task_size, output_size):
        super().__init__()
        self.fc = torch.nn.Linear(task_size, output_size)
        self.norm = torch.nn.LayerNorm(output_size)
        orthogonal_init(self.fc)

    def forward(self, task):
        return F.relu(self.norm(self.fc(task.clone().float())))


class ActionDecoder(torch.nn.Module):
    def __init__(self, input_size, enabled_systems):
        super().__init__()
        self.layers = {
            "attack_style": torch.nn.Linear(input_size, 3),
            "attack_target": torch.nn.Linear(input_size, input_size),
            "market_buy": torch.nn.Linear(input_size, input_size),
            "comm_token": torch.nn.Linear(input_size, 127),
            "inventory_destroy": torch.nn.Linear(input_size, input_size),
            "inventory_give_item": torch.nn.Linear(input_size, input_size),
            "inventory_give_player": torch.nn.Linear(input_size, input_size),
            "gold_quantity": torch.nn.Linear(input_size, 99),
            "gold_target": torch.nn.Linear(input_size, input_size),
            "move": torch.nn.Linear(input_size, 5),
            "inventory_sell": torch.nn.Linear(input_size, input_size),
            "inventory_price": torch.nn.Linear(input_size, 99),
            "inventory_use": torch.nn.Linear(input_size, input_size),
        }

        self.item_enabled = "ITEM" in enabled_systems
        if self.item_enabled is False:
            self.layers.pop("inventory_destroy")
            self.layers.pop("inventory_give_item")
            self.layers.pop("inventory_give_player")
            self.layers.pop("inventory_use")

        self.market_enabled = "EXCHANGE" in enabled_systems
        if self.market_enabled is False:
            self.layers.pop("market_buy")
            self.layers.pop("gold_quantity")
            self.layers.pop("gold_target")
            self.layers.pop("inventory_sell")
            self.layers.pop("inventory_price")

        for _, v in self.layers.items():
            orthogonal_init(v, gain=0.1)
        self.layers = torch.nn.ModuleDict(self.layers)

    def apply_layer(self, layer, embeddings, mask, hidden):
        hidden = layer(hidden)
        if hidden.dim() == 2 and embeddings is not None:
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            hidden = hidden.masked_fill(mask == 0, -1e9)

        return hidden

    def forward(self, hidden, lookup):
        (
            player_embeddings,
            inventory_embeddings,
            market_embeddings,
            action_targets,
        ) = lookup

        embeddings = {
            "attack_target": player_embeddings,
            "market_buy": market_embeddings,
            "inventory_destroy": inventory_embeddings,
            "inventory_give_item": inventory_embeddings,
            "inventory_give_player": player_embeddings,
            "gold_target": player_embeddings,
            "inventory_sell": inventory_embeddings,
            "inventory_use": inventory_embeddings,
        }

        action_targets = {
            "attack_style": action_targets["Attack"]["Style"],
            "attack_target": action_targets["Attack"]["Target"],
            "market_buy": action_targets["Buy"]["MarketItem"] if self.market_enabled else None,
            "comm_token": action_targets["Comm"]["Token"],
            "inventory_destroy": action_targets["Destroy"]["InventoryItem"]
            if self.item_enabled
            else None,
            "inventory_give_item": action_targets["Give"]["InventoryItem"]
            if self.item_enabled
            else None,
            "inventory_give_player": action_targets["Give"]["Target"]
            if self.item_enabled
            else None,
            "gold_quantity": action_targets["GiveGold"]["Price"] if self.market_enabled else None,
            "gold_target": action_targets["GiveGold"]["Target"] if self.market_enabled else None,
            "move": action_targets["Move"]["Direction"],
            "inventory_sell": action_targets["Sell"]["InventoryItem"]
            if self.market_enabled
            else None,
            "inventory_price": action_targets["Sell"]["Price"] if self.market_enabled else None,
            "inventory_use": action_targets["Use"]["InventoryItem"] if self.item_enabled else None,
        }

        # Pass the LSTM output through a ReLU
        # NOTE: The original implementation had relu after both LSTM layers
        hidden = F.relu(hidden)

        actions = []
        for key, layer in self.layers.items():
            mask = None
            mask = action_targets[key]
            embs = embeddings.get(key)
            if embs is not None and embs.shape[1] != mask.shape[1]:
                b, _, f = embs.shape
                zeros = torch.zeros([b, 1, f], dtype=embs.dtype, device=embs.device)
                embs = torch.cat([embs, zeros], dim=1)
            action = self.apply_layer(layer, embs, mask, hidden)
            actions.append(action)

        return actions
