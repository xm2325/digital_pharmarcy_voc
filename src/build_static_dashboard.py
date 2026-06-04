"""Build a portable HTML dashboard preview from generated V2 outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .analytics import emerging_issues
from .routing import deflection_scenario


def fig_html(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def pct(series: pd.Series) -> pd.Series:
    return (100 * series).map(lambda value: f"{value:.1f}%")


def main() -> None:
    data_dir = Path("data")
    reports_dir = Path("reports")
    predictions = pd.read_csv(data_dir / "prediction_sample.csv", parse_dates=["timestamp"])
    metrics = json.loads((reports_dir / "model_metrics.json").read_text(encoding="utf-8"))
    sentiment_metrics = json.loads((reports_dir / "sentiment_metrics.json").read_text(encoding="utf-8"))
    governance = json.loads((reports_dir / "governance_summary.json").read_text(encoding="utf-8"))
    crm_metrics = json.loads((reports_dir / "crm_driver_metrics.json").read_text(encoding="utf-8"))
    benchmark = pd.read_csv(reports_dir / "intent_model_benchmark.csv")
    topics = pd.read_csv(reports_dir / "topic_discovery_clusters.csv")
    scenario = deflection_scenario(predictions)
    issues = emerging_issues(predictions)

    top = predictions["predicted_intent"].value_counts().head(10).rename_axis("intent").reset_index(name="contacts")
    routes = predictions["recommended_route"].value_counts().rename_axis("route").reset_index(name="contacts")
    weekly = predictions.assign(week=predictions["timestamp"].dt.to_period("W-MON").dt.start_time).groupby(["week", "category"]).size().rename("contacts").reset_index()
    topfig = px.bar(top.sort_values("contacts"), x="contacts", y="intent", orientation="h", title="Top predicted contact drivers")
    routefig = px.pie(routes, names="route", values="contacts", hole=0.5, title="Recommended contact routes")
    weeklyfig = px.line(weekly, x="week", y="contacts", color="category", title="Weekly contact trend by category")
    benchfig = px.bar(benchmark, x="model", y="macro_f1", color="split", barmode="group", title="Intent benchmark: random and temporal holdout")

    issue_rows = issues.head(8).copy()
    if not issue_rows.empty:
        issue_rows["latest_share"] = pct(issue_rows["latest_share"])
        issue_rows["recent_mean_share"] = pct(issue_rows["recent_mean_share"])
        issue_rows["z_score"] = issue_rows["z_score"].map(lambda value: f"{value:.2f}")
    issue_table = issue_rows[["intent", "latest_contacts", "latest_share", "recent_mean_share", "z_score", "alert"]].to_html(index=False, escape=True, classes="dataframe")

    topic_rows = topics.head(6).copy().rename(columns={"synthetic_novel_pattern_rate": "demo_novel_pattern_rate"})
    topic_rows["recent_contact_rate"] = pct(topic_rows["recent_contact_rate"])
    topic_rows["demo_novel_pattern_rate"] = pct(topic_rows["demo_novel_pattern_rate"])
    topic_table = topic_rows[["cluster_id", "contacts", "top_terms", "recent_contact_rate", "recommended_review_action", "demo_novel_pattern_rate"]].to_html(index=False, escape=True, classes="dataframe")

    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Digital Pharmacy VoC Workbench V2</title>
<script>{get_plotlyjs()}</script>
<style>
body{{font-family:Arial,sans-serif;background:#f5f7fb;color:#17223b;margin:0}} .wrap{{max-width:1400px;margin:auto;padding:26px}}
h1{{margin-bottom:4px}} .sub{{color:#5b6476;margin-bottom:20px}} .cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}
.card,.panel{{background:white;border-radius:14px;padding:16px;box-shadow:0 2px 10px rgba(18,38,63,.08)}} .metric{{font-size:27px;font-weight:700;margin-top:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}} .panel{{margin-top:16px}} table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:10px;border-bottom:1px solid #e6eaf1;font-size:13px;vertical-align:top}} th{{background:#f8f9fc}}
.note{{font-size:13px;color:#5b6476}} code{{background:#f1f3f7;padding:2px 5px;border-radius:4px}} @media(max-width:900px){{.cards,.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><div class='wrap'>
<h1>Digital Pharmacy Voice-of-the-Customer Workbench · V2</h1><div class='sub'>Synthetic multi-channel contacts · NLP benchmarking · taxonomy evolution · safe automation · CRM insight · governance</div>
<div class='cards'>
<div class='card'><div>Contacts analysed</div><div class='metric'>{len(predictions):,}</div></div>
<div class='card'><div>Intent Macro-F1</div><div class='metric'>{metrics['macro_f1']:.3f}</div></div>
<div class='card'><div>Urgent-route recall</div><div class='metric'>{metrics['urgent_route_recall']:.1%}</div></div>
<div class='card'><div>Potential deflection</div><div class='metric'>{scenario['potential_deflection_rate']:.1%}</div></div>
<div class='card'><div>Contacts with redaction</div><div class='metric'>{governance['contacts_with_redaction_rate']:.1%}</div></div>
</div>
<div class='note' style='margin-top:10px'>All contact and CRM records are synthetic. Classification and scenario values validate the workflow; they are not company estimates.</div>
<div class='grid'><div class='panel'>{fig_html(topfig)}</div><div class='panel'>{fig_html(routefig)}</div></div>
<div class='panel'>{fig_html(weeklyfig)}</div>
<div class='panel'>{fig_html(benchfig)}</div>
<div class='grid'>
<div class='panel'><h2>Emerging issue monitor</h2><p class='note'>Recent three-week share compared with the earlier baseline.</p>{issue_table}</div>
<div class='panel'><h2>Taxonomy evolution queue</h2><p class='note'>Unknown-topic clusters from out-of-scope predictions. The validation-only novel-pattern field is available because the dataset is synthetic.</p>{topic_table}</div>
</div>
<div class='grid'>
<div class='panel'><h2>Additional V2 checks</h2><table><tr><th>Check</th><th>Result</th></tr>
<tr><td>Sentiment Macro-F1</td><td>{sentiment_metrics['macro_f1']:.3f}</td></tr>
<tr><td>Expected calibration error</td><td>{metrics['expected_calibration_error']:.3f}</td></tr>
<tr><td>Low-confidence review rate at 0.72</td><td>{metrics['low_confidence_rate_at_0_72']:.1%}</td></tr>
<tr><td>Total rule-based redactions</td><td>{governance['total_redactions']:,}</td></tr>
<tr><td>Low-CSAT driver model ROC-AUC</td><td>{crm_metrics['low_csat']['roc_auc']:.3f}</td></tr>
</table></div>
<div class='panel'><h2>Dashboard scope</h2><p>Run <code>streamlit run app.py</code> for ten interactive pages: filters, journey analysis, emerging issues, taxonomy evolution, contact deflection, model benchmark, annotation workflow, synthetic CRM drivers, governance, and chatbot knowledge-base readiness.</p></div>
</div>
</div></body></html>"""
    (reports_dir / "dashboard_preview.html").write_text(html_text, encoding="utf-8")
    print("Wrote reports/dashboard_preview.html")


if __name__ == "__main__":
    main()
