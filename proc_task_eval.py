import os
import json
import logging
import argparse

import numpy as np
import polars as pl

# Make the table output simpler
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_hide_column_data_types(True)

# string matching for task names
WEIGHT_DICT = {
    "TickGE": ("survival", 100 / 6),  # 1 survival task
    "PLAYER_KILL": ("combat", 100 / (6 * 3)),  # 3 combat tasks
    "DefeatEntity": ("combat", 100 / (6 * 3)),
    "GO_FARTHEST": ("exploration", 100 / (6 * 2)),  # 2 exploration tasks
    "OccupyTile": ("exploration", 100 / (6 * 2)),
    "AttainSkill": ("skill", 100 / (6 * 8)),  # 8 skill tasks
    "HarvestItem": ("item", 100 / (6 * 44)),  # 44 item tasks
    "ConsumeItem": ("item", 100 / (6 * 44)),
    "EquipItem": ("item", 100 / (6 * 44)),
    "FullyArmed": ("item", 100 / (6 * 44)),
    "EARN_GOLD": ("market", 100 / (6 * 5)),  # 5 market tasks
    "BUY_ITEM": ("market", 100 / (6 * 5)),
    "EarnGold": ("market", 100 / (6 * 5)),
    "HoardGold": ("market", 100 / (6 * 5)),
    "MakeProfit": ("market", 100 / (6 * 5)),
}


def get_task_weight(task_name):
    for key, val in WEIGHT_DICT.items():
        if key in task_name:
            return val
    logging.warning(f"Task name {task_name} not found in weight dict")
    return "etc", 0


def get_summary_dict(progress, key):
    # progress = vals if key == "length" else [v[0] for v in vals]
    summ = {"count": len(progress), "mean": np.mean(progress), "median": np.median(progress)}

    if key == "length":
        progress = np.array(progress) / 1023  # full episode length

    summ["completed"] = np.mean([1 if v >= 1 else 0 for v in progress])
    summ["over30pcnt"] = np.mean([1 if v >= 0.3 else 0 for v in progress])
    return summ


def summarize_single_eval(data, weighted_score=False):
    summary = {}

    # task-level info
    for key, vals in data.items():
        if key.startswith("curriculum") or key == "length":
            summary[key] = get_summary_dict(vals, key)

            if weighted_score and key.startswith("curriculum"):
                category, weight = get_task_weight(key)
                summary[key]["category"] = category
                summary[key]["weight"] = weight
                summary[key]["weighted_score"] = summary[key]["mean"] * weight

    # meta info
    summary["avg_progress"] = np.mean(
        [v["mean"] for k, v in summary.items() if k.startswith("curriculum")]
    )
    if weighted_score:
        summary["weighted_score"] = np.sum(
            [v["weighted_score"] for k, v in summary.items() if k.startswith("curriculum")]
        )
    return summary


def process_eval_files(policy_store_dir, eval_prefix):
    summ_policy = []
    summ_task = []

    for file in os.listdir(policy_store_dir):
        # NOTE: assumes the file naming convention is 'curriculum_info_<seed>.json'
        if not file.startswith(eval_prefix) or not file.endswith(".json"):
            continue

        random_seed = file.split("_")[2].replace(".json", "")

        with open(os.path.join(policy_store_dir, file), "r") as f:
            data = json.load(f)

        for pol_name, pol_data in data.items():
            if len(pol_data) == 0:
                continue

            mode = "pvp" if len(pol_data) > 1 else "pve"

            summary = summarize_single_eval(pol_data, weighted_score=True)
            summ_policy.append(
                {
                    "policy_name": pol_name,
                    "mode": mode,
                    "seed": random_seed,
                    "count": summary["length"]["count"],
                    "length": summary["length"]["mean"],
                    "score": summary["avg_progress"],
                    "weighted_score": summary["weighted_score"],
                }
            )

            # also gather the results across random seeds for each task, then average
            for task_name, task_data in summary.items():
                if not task_name.startswith("curriculum"):
                    continue
                summ_task.append(
                    {
                        "category": task_data["category"],
                        "task_name": task_name,
                        "weight": task_data["weight"],
                        "policy_name": pol_name,
                        "mode": mode,
                        "seed": random_seed,
                        "count": task_data["count"],
                        "score": task_data["mean"],
                    }
                )

    summ_df = pl.DataFrame(summ_policy).sort(["policy_name", "mode", "seed"])
    summ_grp = summ_df.group_by(["policy_name", "mode"]).agg(
        pl.col("length").mean(),
        pl.col("score").mean(),
        pl.col("weighted_score").mean(),
    )
    summ_grp = summ_grp.sort("weighted_score", descending=True)
    summ_grp.write_csv(
        os.path.join(policy_store_dir, "score_summary.tsv"), separator="\t", float_precision=6
    )
    print("\nPolicy score summary, sorted by weighted_score:")
    print(summ_grp)

    task_df = pl.DataFrame(summ_task).sort(["mode", "category", "task_name", "policy_name", "seed"])
    task_grp = task_df.group_by(["mode", "category", "task_name", "policy_name"]).agg(
        pl.col("score").mean()
    )
    task_grp = task_grp.sort(["mode", "category", "task_name", "policy_name"])
    task_grp.write_csv(
        os.path.join(policy_store_dir, "score_task_summary.tsv"), separator="\t", float_precision=6
    )
    cate_grp = task_df.group_by(["mode", "category", "policy_name"]).agg(pl.col("score").mean())
    cate_grp = cate_grp.sort(["mode", "category", "policy_name"])
    cate_grp.write_csv(
        os.path.join(policy_store_dir, "score_category_summary.tsv"),
        separator="\t",
        float_precision=6,
    )

    if len(summ_df["seed"].unique()) > 1:
        summ_df.write_csv(
            os.path.join(policy_store_dir, "score_by_seed.tsv"), separator="\t", float_precision=6
        )
        task_df.write_csv(
            os.path.join(policy_store_dir, "score_by_task_seed.tsv"),
            separator="\t",
            float_precision=6,
        )

    return summ_df, summ_grp, task_df, task_grp, cate_grp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the evaluation result files")
    parser.add_argument("policy_store_dir", type=str, help="Path to the policy directory")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="curriculum_",
        help="Prefix of the evaluation result files",
    )
    args = parser.parse_args()

    process_eval_files(args.policy_store_dir, args.prefix)
