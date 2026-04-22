"""Recreate activation_asr.png with the AAAI Table 1 numbers."""
import matplotlib.pyplot as plt
import numpy as np

methods = ["Baseline", "Ablation", "GCG", "Single", "Negative", "All", "Layer", "Token"]
substring   = [0.39, 0.98, 0.76, 0.84, 0.81, 0.91, 0.81, 0.76]
llamaguard2 = [0.00, 0.50, 0.00, 0.06, 0.03, 0.09, 0.03, 0.00]
harmbench   = [0.00, 0.62, 0.00, 0.04, 0.03, 0.01, 0.01, 0.00]

# Match seaborn-deep palette colors used in the original figure
COLOR_SUBSTRING   = "#4C72B0"  # blue
COLOR_LLAMAGUARD2 = "#C44E52"  # red
COLOR_HARMBENCH   = "#55A868"  # green

x = np.arange(len(methods))
width = 0.27

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
ax.set_facecolor("#FAFAFA")

ax.bar(x - width, substring,   width, label="Substring",   color=COLOR_SUBSTRING,   edgecolor="black", linewidth=0.6)
ax.bar(x,         llamaguard2, width, label="LlamaGuard2", color=COLOR_LLAMAGUARD2, edgecolor="black", linewidth=0.6)
ax.bar(x + width, harmbench,   width, label="HarmBench",   color=COLOR_HARMBENCH,   edgecolor="black", linewidth=0.6)

ax.set_title("Attack Success by Method", fontsize=14, pad=10)
ax.set_ylabel("Attack Success Rate", fontsize=11)
ax.set_ylim(0.0, 1.05)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)

ax.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#888888")

leg = ax.legend(title="Metrics", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=10, title_fontsize=10)
leg.get_frame().set_edgecolor("#CCCCCC")
leg.get_frame().set_facecolor("white")

plt.tight_layout()
out = "/home/kayden/Common/Harvard/Y3/CS2881/final/activation_asr.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
