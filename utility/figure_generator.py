import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Ensure the Figures directory exists
output_dir = "Figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = "H1_H2_merged_evaluations.json"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    exit()


def get_display_name(key):
    key = key.replace("_h1", "")
    if "claude-haiku" in key:
        return "Claude Haiku 4.5"
    if "claude-sonnet" in key:
        return "Claude Sonnet 4.5"
    if "gemini-3-flash" in key:
        return "Gemini 3 Flash"
    if "gemini-3-pro" in key:
        return "Gemini 3 Pro"
    if "gpt-5-mini" in key:
        return "GPT-5-mini"
    if "gpt-5.1" in key:
        return "GPT-5.1"
    return key


def get_h2_metric(stage_data, model_key_part, profile, metric):
    for vendor in ["claude", "gemini", "openai"]:
        if vendor in stage_data:
            for model_key, model_data in stage_data[vendor].items():
                if model_key_part in model_key:
                    return model_data["profile_summaries"][profile][metric]
    return None


def save_publication_figures(name):
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight")


models_h1_keys = [
    "claude-haiku-4.5_h1",
    "claude-sonnet-4.5_h1",
    "gemini-3-flash_h1",
    "gemini-3-pro_h1",
    "gpt-5-mini_h1",
    "gpt-5.1_h1",
]
models_h2_keys = [
    "claude-haiku-4.5",
    "claude-sonnet-4.5",
    "gemini-3-flash",
    "gemini-3-pro",
    "gpt-5-mini",
    "gpt-5.1",
]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# --- Figure 5 (Performance Trajectories) ---
plt.figure(figsize=(10, 6))
conditions, x_coords = ["Baseline", "Method A"], [0, 1]
for i, model_key in enumerate(models_h1_keys):
    y_vals = [
        data["H1_Baseline"][model_key]["overall_f1"],
        data["H1_Method-A"][model_key]["overall_f1"],
    ]
    plt.plot(
        x_coords,
        y_vals,
        marker="o",
        label=get_display_name(model_key),
        color=colors[i],
        linewidth=2,
    )
plt.xticks(x_coords, conditions)
plt.ylabel("Overall F1 Score")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
save_publication_figures("figure_5")
plt.close()

# --- Figure 6 (Convergence Pattern) ---
plt.figure(figsize=(10, 6))
conditions, x_coords = ["Baseline", "Method A", "Method B"], [0, 1, 2]
for i, model_key in enumerate(models_h1_keys):
    y_vals = [
        data["H1_Baseline"][model_key]["overall_f1"],
        data["H1_Method-A"][model_key]["overall_f1"],
        data["H1_Method-B"][model_key]["overall_f1"],
    ]
    plt.plot(
        x_coords,
        y_vals,
        marker="o",
        label=get_display_name(model_key),
        color=colors[i],
        linewidth=2,
    )
plt.xticks(x_coords, conditions)
plt.ylabel("Overall F1 Score")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
save_publication_figures("figure_6")
plt.close()

# --- Figure 7 (PR Trade-off) ---
plt.figure(figsize=(10, 7))
profile_a = "Profile_A_Vegan_Diabetic_Gluten"
stages = [
    ("H2_Baseline", "o", "Baseline"),
    ("H2_Method A", "^", "Method A"),
    ("H2_Method B", "s", "Method B"),
]
legend_elements = [
    Line2D(
        [0], [0], marker=m, color="w", label=l, markerfacecolor="gray", markersize=10
    )
    for _, m, l in stages
]
for i, m_key in enumerate(models_h2_keys):
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=get_display_name(m_key),
            markerfacecolor=colors[i],
            markersize=10,
        )
    )
for i, model_key in enumerate(models_h2_keys):
    for stage_key, marker, _ in stages:
        prec = get_h2_metric(data[stage_key], model_key, profile_a, "average_precision")
        rec = get_h2_metric(data[stage_key], model_key, profile_a, "average_recall")
        if prec is not None and rec is not None:
            plt.scatter(rec, prec, color=colors[i], marker=marker, s=100, alpha=0.8)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(
    handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3
)
save_publication_figures("figure_7")
plt.close()

# --- Figure 8 (Profile B Stability) ---
plt.figure(figsize=(12, 6))
profile_b = "Profile_B_Halal_Hypertension_Lactose"
bar_width, index = 0.25, np.arange(len(models_h2_keys))
y_base = [
    get_h2_metric(data["H2_Baseline"], m, profile_b, "average_f1_score")
    for m in models_h2_keys
]
y_a = [
    get_h2_metric(data["H2_Method A"], m, profile_b, "average_f1_score")
    for m in models_h2_keys
]
y_b = [
    get_h2_metric(data["H2_Method B"], m, profile_b, "average_f1_score")
    for m in models_h2_keys
]
plt.bar(index - bar_width, y_base, bar_width, label="Baseline", color="skyblue")
plt.bar(index, y_a, bar_width, label="Method A", color="lightgreen")
plt.bar(index + bar_width, y_b, bar_width, label="Method B", color="salmon")
plt.ylabel("F1 Score")
plt.xticks(index, [get_display_name(m) for m in models_h2_keys], rotation=15)
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
save_publication_figures("figure_8")
plt.close()

# --- Figure 9 (Non-monotonic Relationship) ---
plt.figure(figsize=(10, 6))
profile_c, x_coords = "Profile_C_Keto_NutAllergy_Heart", [0, 1, 2]
for i, model_key in enumerate(models_h2_keys):
    y_vals = [
        get_h2_metric(data[s], model_key, profile_c, "average_f1_score")
        for s in ["H2_Baseline", "H2_Method A", "H2_Method B"]
    ]
    plt.plot(
        x_coords,
        y_vals,
        marker="o",
        label=get_display_name(model_key),
        color=colors[i],
        linewidth=2,
    )
plt.xticks(x_coords, ["Baseline", "Method A", "Method B"])
plt.ylabel("F1 Score")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
save_publication_figures("figure_9")
plt.close()

# --- Figure 10 (Performance Inversion) ---
plt.figure(figsize=(10, 7))
target_models = ["claude-sonnet-4.5", "gpt-5.1", "gpt-5-mini"]
x_labels, x_coords = [
    "Baseline\n(Noisy)",
    "Method A\n(Heuristic)",
    "Method B\n(Fidelity)",
], [0, 1, 2]
style_map = {
    "gpt-5-mini": {
        "color": "#c62828",
        "width": 4,
        "style": "-",
        "label": "GPT-5-mini (Light)",
    },
    "gpt-5.1": {
        "color": "#6e6e6e",
        "width": 2.5,
        "style": "--",
        "label": "GPT-5.1 (Heavy)",
    },
    "claude-sonnet-4.5": {
        "color": "#1f4e79",
        "width": 2.5,
        "style": "--",
        "label": "Claude Sonnet 4.5 (Heavy)",
    },
}
for model_key in target_models:
    y_vals = []
    vendor_key = "openai" if "gpt" in model_key else "claude"
    for stage in ["H2_Baseline", "H2_Method A", "H2_Method B"]:
        y_vals.append(
            data[stage][vendor_key][model_key]["overall_statistics"][
                "overall_average_f1"
            ]
        )
    p = style_map[model_key]
    plt.plot(
        x_coords,
        y_vals,
        marker="o",
        markersize=8,
        color=p["color"],
        linewidth=p["width"],
        linestyle=p["style"],
        label=p["label"],
    )
    plt.text(
        x_coords[-1] + 0.05,
        y_vals[-1],
        f"{y_vals[-1]:.3f}",
        va="center",
        color=p["color"],
        fontweight="bold",
    )
plt.xticks(x_coords, x_labels)
plt.ylabel("Macro-Averaged Risk Assessment F1")
plt.grid(True, linestyle=":", alpha=0.6)
plt.annotate(
    "Performance Inversion",
    xy=(2, 0.894),
    xytext=(1.85, 0.92),
    arrowprops=dict(facecolor="black", shrink=0.15, width=1.5),
    fontweight="bold",
)
plt.legend(loc="lower left", frameon=True)
save_publication_figures("figure_10")
plt.close()
