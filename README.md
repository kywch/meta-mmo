# Meta MMO: Massively Multiagent Minigames for Training Generalist Agents

Meta MMO is a collection of many-agent minigames built on top of [Neural MMO](https://github.com/NeuralMMO/environment) to serve as a benchmark for reinforcement learning. It offers a diverse set of configurable minigames that allow fine-grained control over game objectives, agent spawning, team assignments, and various game elements. Meta MMO enables faster training (up to 3x speedup compared to Neural MMO), adaptive difficulty, and curriculum learning.

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/kywch/meta-mmo.git
   cd meta-mmo
   ```

2. Install the required dependencies:
   ```
   pip install -e .[dev]
   ```

3. Train specialists for each minigame or a generalist policy:
   ```
   # Train specialists for Team Battle (tb), Protect the King (pk),
   # Race to the Center (rc), King of the Hill (kh), and Sandwich (sw)
   python train.py --use-mini -t tb_only
   python train.py --use-mini -t pk_only
   python train.py --use-mini -t rc_only
   python train.py --use-mini -t kh_only
   python train.py --use-mini -t sw_only

   # Train a generalist for playing all five games
   python train.py --use-mini -t mini_gen --train.total-timesteps 400_000_000
   ```

4. Evaluate trained policies:
The script below evaluates the checkpoints included in thie repository. 
   ```
   # Mini config minigames: battle, ptk, race, koh, sandwich
   python evaluate.py experiments/mini_tb -g battle -r 10
   python proc_elo.py experiments/mini_tb battle
   
   python evaluate.py experiments/mini_pk -g ptk -r 10
   python proc_elo.py experiments/mini_pk ptk
   
   python evaluate.py experiments/mini_rc -g race -r 10
   python proc_elo.py experiments/mini_rc race
   
   python evaluate.py experiments/mini_kh -g koh -r 10
   python proc_elo.py experiments/mini_kh koh
   
   python evaluate.py experiments/mini_sw -g sandwich -r 10
   python proc_elo.py experiments/mini_sw sandwich
   ```

## Minigames

Meta MMO includes several minigames, each focusing on different aspects of gameplay:

- **Survival**: Agents must survive until the end of the episode
- **Team Battle**: Last team standing wins (The NeurIPS 2022 competition)
- **Multi-task Training/Evaluation**: Free-for-all with agents assigned random tasks (The NeurIPS 2023 competition)
- **Protect the King**: Teams must protect their leader while eliminating other teams
- **Race to the Center**: First agent to reach the center tile wins
- **King of the Hill**: Teams must capture and hold the center tile
- **Sandwich**: Teams must defeat all other teams and survive while fighting NPCs and a shrinking map

## Baselines

We provide baseline generalist and specialist policies trained using PPO with historical self-play. The generalist policy is capable of playing multiple minigames with a single set of weights, matching or outperforming specialist policies trained on the same number of environment steps for the target minigame.

## Citing Meta MMO

If you use Meta MMO in your research, please cite the following paper:

```
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Neural MMO](https://github.com/NeuralMMO/environment) - The base environment for Meta MMO
- [PufferLib](https://github.com/PufferAI/pufferlib) - A library for super efficient RL training for complex environments
