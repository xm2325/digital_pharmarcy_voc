"""Build a PNG summary image for the V2 repository preview."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .analytics import emerging_issues
from .routing import deflection_scenario


def main() -> None:
    data = Path("data")
    reports = Path("reports")
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    pred = pd.read_csv(data / "prediction_sample.csv", parse_dates=["timestamp"])
    metrics = json.loads((reports / "model_metrics.json").read_text(encoding="utf-8"))
    governance = json.loads((reports / "governance_summary.json").read_text(encoding="utf-8"))
    scenario = deflection_scenario(pred)
    issues = emerging_issues(pred)
    topics = pd.read_csv(reports / "topic_discovery_clusters.csv")
    top = pred["predicted_intent"].value_counts().head(9).sort_values()
    routes = pred["recommended_route"].value_counts().sort_values()

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Digital Pharmacy Voice-of-the-Customer Workbench · V2", fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.06, 0.935, "Synthetic contacts · NLP benchmark · taxonomy evolution · safe automation · governance", fontsize=11)

    metric_labels = ["Contacts analysed", "Intent Macro-F1", "Urgent-route recall", "Potential deflection", "Contacts redacted"]
    metric_values = [f"{len(pred):,}", f"{metrics['macro_f1']:.3f}", f"{metrics['urgent_route_recall']:.1%}", f"{scenario['potential_deflection_rate']:.1%}", f"{governance['contacts_with_redaction_rate']:.1%}"]
    lefts = [0.06, 0.25, 0.44, 0.63, 0.82]
    for x, label, value in zip(lefts, metric_labels, metric_values):
        fig.text(x, 0.865, label, fontsize=10)
        fig.text(x, 0.825, value, fontsize=18, fontweight="bold")

    ax1 = fig.add_axes([0.07, 0.48, 0.42, 0.28])
    top.plot(kind="barh", ax=ax1)
    ax1.set_title("Top predicted contact drivers")
    ax1.set_xlabel("Contacts in held-out synthetic sample")
    ax1.set_ylabel("")
    ax1.grid(axis="x", alpha=0.25)

    ax2 = fig.add_axes([0.57, 0.48, 0.36, 0.28])
    routes.plot(kind="barh", ax=ax2)
    ax2.set_title("Recommended routes at threshold 0.72")
    ax2.set_xlabel("Contacts")
    ax2.set_ylabel("")
    ax2.grid(axis="x", alpha=0.25)

    ax3 = fig.add_axes([0.07, 0.10, 0.86, 0.27])
    weekly = pred.assign(week=pred["timestamp"].dt.to_period("W-MON").dt.start_time)
    series = weekly[weekly["predicted_intent"].isin(["delivery_delayed", "unclear_or_out_of_scope", "waiting_gp_approval"])].groupby(["week", "predicted_intent"]).size().unstack(fill_value=0)
    series.plot(ax=ax3, marker="o")
    ax3.set_title("Weekly contact volume for selected operational drivers")
    ax3.set_xlabel("Week")
    ax3.set_ylabel("Contacts")
    ax3.grid(alpha=0.25)

    alerts = issues[issues["alert"]]["intent"].tolist()
    top_topic = topics.sort_values("synthetic_novel_pattern_rate", ascending=False).iloc[0]
    fig.text(0.07, 0.047, "Flagged synthetic emerging issues: " + (", ".join(alerts) if alerts else "none"), fontsize=10, fontweight="bold")
    fig.text(0.07, 0.022, f"Unknown-topic cluster {int(top_topic['cluster_id'])}: {top_topic['top_terms']} · action: {top_topic['recommended_review_action']}", fontsize=9)
    fig.text(0.93, 0.022, "All values are demonstration outputs.", fontsize=9, ha="right")
    fig.savefig(assets / "dashboard_preview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Wrote assets/dashboard_preview.png")


if __name__ == "__main__":
    main()
