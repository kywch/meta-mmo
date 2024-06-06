# %%
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
import proc_elo
import proc_task_eval


# Figure style-related
TITLE_FONT = 24
LABEL_FONT = 18
TICK_FONT = 11
ELO_SAMPLE_TICKS = list(range(0, 150, 25))
ELO_SAMPLE_LABELS = ["0", "25M", "50M", "75M", "100M", "125M"]
ELO_YTIKCS = list(range(850, 1101, 50))

MT_SAMPLE_TICKS = list(range(0, 200, 50))
MT_SAMPLE_LABELS = ["0", "50M", "100M", "150M"]
MT_YTICKS = [y / 100 for y in range(0, 21, 4)]

# blue-ish: #4c72b0, orange: #dd8452
SPECIALIST_MARKER = {
    "marker": "o",
    "s": 100,
    "c": "white",
    "edgecolors": "#4c72b0",
    "label": "Specialist",
    "linewidths": 3,
}
GENERALIST_MARKER = {"marker": "o", "s": 120, "c": "#dd8452", "label": "Generalist"}

# Manually extract sampling ratio for each minigame, after running average smoothing with 100
# https://wandb.ai/kywch/meta-mmo/runs/lf95vvxr, see stats/Sampling/{minigame}_agent_steps
generalist_sample_ratio = {
    "svonly": [0.288, 0.290, 0.293, 0.291],  # Survival
    "tbonly": [0.305, 0.304, 0.299, 0.296],  # Team Battle
    "mtonly": [0.407, 0.406, 0.408, 0.412],  # Multi-task Traiing
}


def process_eval_data(
    data, specialist_prefix, sample_ratio, value_key="elo", generalist_prefix="general"
):
    results = {"generalist": {"steps": [], "values": []}, "specialist": {"steps": [], "values": []}}
    for pol_data in data:
        pol_info = pol_data["policy"].split("_")
        assert pol_info[0] in [specialist_prefix, generalist_prefix], ""
        pol_type = "generalist" if pol_info[0] == generalist_prefix else "specialist"
        results[pol_type]["values"].append(pol_data[value_key])
        results[pol_type]["steps"].append(int(pol_info[1][:-1]))

    # Apply sampling ratio to correct the steps
    gen_idx = np.argsort(results["generalist"]["steps"])
    for ii, idx in enumerate(gen_idx):
        results["generalist"]["steps"][ii] *= sample_ratio[idx]

    return results


if __name__ == "__main__":
    # Data for each minigame
    survive_elo = process_eval_data(
        proc_elo.process_eval_files("full_sv", "survive"),
        "svonly",
        generalist_sample_ratio["svonly"],
    )

    battle_elo = process_eval_data(
        proc_elo.process_eval_files("full_tb", "battle"),
        "tbonly",
        generalist_sample_ratio["tbonly"],
    )

    task_data = proc_task_eval.process_eval_files("full_mt", "curriculum").to_dicts()
    task_progress = process_eval_data(
        task_data, "mtonly", generalist_sample_ratio["mtonly"], value_key="task_progress"
    )

    # Create subplots with specified figure size
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Remove top and right edges for each subplot
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", length=6, labelsize=TICK_FONT)

    # Marker styles for generalist and specialist

    # Survival subplot
    ax1.set_title("Survival", fontsize=TITLE_FONT)
    ax1.set_ylabel("Elo", fontsize=TITLE_FONT)
    ax1.set_xticks(ELO_SAMPLE_TICKS)
    ax1.set_xticklabels(ELO_SAMPLE_LABELS, fontsize=TICK_FONT)
    ax1.set_xlim(0, 125)
    ax1.set_yticks(ELO_YTIKCS)
    ax1.set_ylim(850, 1100)
    ax1.plot([0, 150], [1000, 1000], "k:")  # Anchor line for ELO
    ax1.scatter(
        survive_elo["generalist"]["steps"], survive_elo["generalist"]["values"], **GENERALIST_MARKER
    )
    ax1.scatter(
        survive_elo["specialist"]["steps"], survive_elo["specialist"]["values"], **SPECIALIST_MARKER
    )
    ax1.legend(loc="lower right", fontsize=13)

    # Team Battle subplot
    ax2.set_title("Team Battle", fontsize=TITLE_FONT)
    ax2.set_xlabel("Training samples", fontsize=TITLE_FONT, labelpad=13)
    ax2.set_xticks(ELO_SAMPLE_TICKS)
    ax2.set_xticklabels(ELO_SAMPLE_LABELS, fontsize=TICK_FONT)
    ax2.set_xlim(0, 125)
    ax2.set_yticks(ELO_YTIKCS)
    ax2.set_ylim(850, 1100)
    ax2.plot([0, 150], [1000, 1000], "k:")  # Anchor line for ELO
    ax2.scatter(
        battle_elo["specialist"]["steps"], battle_elo["specialist"]["values"], **SPECIALIST_MARKER
    )
    ax2.scatter(
        battle_elo["generalist"]["steps"], battle_elo["generalist"]["values"], **GENERALIST_MARKER
    )

    # Multi-task Eval subplot
    ax3.set_title("Multi-task Eval", fontsize=TITLE_FONT)
    ax3.set_ylabel("Task progress", fontsize=TITLE_FONT)
    ax3.set_xticks(MT_SAMPLE_TICKS)
    ax3.set_xticklabels(MT_SAMPLE_LABELS, fontsize=TICK_FONT)
    ax3.set_xlim(0, 175)
    ax3.set_yticks(MT_YTICKS)
    ax3.set_ylim(0, 0.2)
    ax3.scatter(
        task_progress["specialist"]["steps"],
        task_progress["specialist"]["values"],
        **SPECIALIST_MARKER,
    )
    ax3.scatter(
        task_progress["generalist"]["steps"],
        task_progress["generalist"]["values"],
        **GENERALIST_MARKER,
    )

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)

    # Save the figure as a PNG file with specified size
    plt.savefig("fig_4.png", dpi=300, bbox_inches="tight")

    # Display the figure
    plt.show()
