import os
import json
import argparse
from collections import defaultdict

from pufferlib.policy_ranker import update_elos
import polars as pl

# Make the table output simpler
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_hide_column_data_types(True)


def calculate_elo(game_results, anchor_elo=1000):
    elo_dict = defaultdict(lambda: anchor_elo)

    for episode in game_results:
        players = list(episode.keys())
        elos = update_elos(
            elos=[elo_dict[player] for player in players],
            scores=[episode[player] for player in players],
        )
        for player, elo in zip(players, elos):
            elo_dict[player] = elo

    return [{"policy": policy, "elo": elo} for policy, elo in elo_dict.items()]


def process_eval_files(results_dir, file_prefix):
    game_results = []
    for file in os.listdir(results_dir):
        # NOTE: assumes the file naming convention is '<file_prefix>_<seed>.json'
        if not file.startswith(file_prefix) or not file.endswith(".json"):
            continue

        with open(os.path.join(results_dir, file), "r") as f:
            # NOTE: Assuming that the data is a list of dictionaries
            game_results += json.load(f)

    return calculate_elo(game_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ELO scores")
    parser.add_argument("results_dir", type=str, help="Path to the results file")
    parser.add_argument("file_prefix", type=str, help="Prefix of the results file")
    args = parser.parse_args()

    results = process_eval_files(args.results_dir, args.file_prefix)

    # TODO: make it better?
    print(pl.DataFrame(results).sort("elo", descending=True))
