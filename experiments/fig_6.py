# %%
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
import proc_elo
from fig_4 import process_eval_data


# Figure style-related
TITLE_FONT = 26
TICK_FONT = 12
ELO_SAMPLE_TICKS = list(range(0, 150, 25))
ELO_SAMPLE_LABEL = ["0", "25M", "50M", "75M", "100M", "125M"]
ELO_YTIKCS = list(range(850, 1101, 50))

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
# https://wandb.ai/kywch/meta-mmo/runs/53n3yvnj, see stats/Sampling/{minigame}_agent_steps
generalist_sample_ratio = {
    "battle": [0.218, 0.261, 0.274, 0.280],  # Team Battle
    "ptk": [0.153, 0.188, 0.201, 0.217],  # Protect the King
    "race": [0.164, 0.120, 0.100, 0.097],  # Race to the Center
    "koh": [0.228, 0.231, 0.223, 0.210],  # King of the Hill
    "sandwich": [0.237, 0.200, 0.202, 0.196],  # Sandwich
}

minigame_info = {
    "battle": ("Team Battle", "mini_tb", "tbonly"),  # name, directory, prefix
    "ptk": ("Protect the King", "mini_pk", "pkonly"),
    "race": ("Race to the Center", "mini_rc", "rconly"),
    "koh": ("King of the Hill", "mini_kh", "khonly"),
    "sandwich": ("Sandwich", "mini_sw", "swonly"),
}

if __name__ == "__main__":
    # Data for each minigame
    elo_data = {
        game: process_eval_data(
            proc_elo.process_eval_files(dir, game), prefix, generalist_sample_ratio[game]
        )
        for game, (_, dir, prefix) in minigame_info.items()
    }

    # Create subplots with specified figure size
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Remove top and right edges for each subplot
    for ax, game in zip(axes, minigame_info.keys()):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", length=6, labelsize=TICK_FONT)

        ax.set_title(minigame_info[game][0], fontsize=TITLE_FONT)
        ax.set_xticks(ELO_SAMPLE_TICKS)
        ax.set_xticklabels(ELO_SAMPLE_LABEL, fontsize=TICK_FONT)
        ax.set_xlim(0, 105)
        if max(elo_data[game]["generalist"]["steps"]) > 100:
            ax.set_xlim(0, 125)
        ax.set_yticks(ELO_YTIKCS)
        ax.set_ylim(880, 1100)
        ax.plot([0, 150], [1000, 1000], "k:")  # Anchor line for ELO
        ax.scatter(
            elo_data[game]["generalist"]["steps"],
            elo_data[game]["generalist"]["values"],
            **GENERALIST_MARKER,
        )
        ax.scatter(
            elo_data[game]["specialist"]["steps"],
            elo_data[game]["specialist"]["values"],
            **SPECIALIST_MARKER,
        )

    axes[0].set_ylabel("Elo", fontsize=TITLE_FONT + 2)
    axes[0].legend(loc="lower right", fontsize=15)
    axes[2].set_xlabel("Training samples", fontsize=TITLE_FONT + 6, labelpad=15)

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    # Save the figure as a PNG file with specified size
    plt.savefig("fig_6.png", dpi=300, bbox_inches="tight")

    # Display the figure
    plt.show()
