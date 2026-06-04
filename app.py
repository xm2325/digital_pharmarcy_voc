"""Streamlit dashboard for the V2 Digital Pharmacy VoC Workbench."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import emerging_issues, weekly_intent_trends
from src.faq_retrieval import FAQRetriever
from src.preprocess import clean_text
from src.routing import add_routes, assign_route, deflection_scenario

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(page_title="Digital Pharmacy VoC Workbench V2", page_icon="💊", layout="wide")


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_data() -> dict[str, object]:
    return {
        "predictions": pd.read_csv(DATA / "prediction_sample.csv", parse_dates=["timestamp"]),
        "crm": pd.read_csv(DATA / "synthetic_crm.csv"),
        "faq": pd.read_csv(DATA / "faq_knowledge_base.csv"),
        "taxonomy": pd.read_csv(DATA / "intent_taxonomy.csv"),
        "taxonomy_versions": pd.read_csv(DATA / "taxonomy_versions.csv"),
        "annotation_queue": pd.read_csv(DATA / "annotation_review_queue.csv", parse_dates=["timestamp"]),
        "coverage": pd.read_csv(REPORTS / "coverage_accuracy_curve.csv"),
        "metrics": read_json(REPORTS / "model_metrics.json"),
        "sentiment_metrics": read_json(REPORTS / "sentiment_metrics.json"),
        "benchmark": pd.read_csv(REPORTS / "intent_model_benchmark.csv"),
        "channel_metrics": pd.read_csv(REPORTS / "intent_metrics_by_channel.csv"),
        "reliability": pd.read_csv(REPORTS / "intent_reliability_diagram.csv"),
        "issues": pd.read_csv(REPORTS / "emerging_issues.csv", parse_dates=["latest_week"]),
        "topic_clusters": pd.read_csv(REPORTS / "topic_discovery_clusters.csv"),
        "topic_examples": pd.read_csv(REPORTS / "topic_discovery_examples.csv", parse_dates=["timestamp"]),
        "crm_linked": pd.read_csv(REPORTS / "crm_linked_patient_features.csv"),
        "crm_driver_metrics": read_json(REPORTS / "crm_driver_metrics.json"),
        "crm_coefficients": pd.read_csv(REPORTS / "crm_driver_coefficients.csv"),
        "governance": read_json(REPORTS / "governance_summary.json"),
        "redaction_counts": pd.read_csv(REPORTS / "entity_redaction_counts.csv"),
        "quality_by_channel": pd.read_csv(REPORTS / "data_quality_by_channel.csv"),
        "redaction_audit": pd.read_csv(REPORTS / "redaction_audit_sample.csv"),
        "faq_metrics": pd.read_csv(REPORTS / "faq_retrieval_metrics.csv"),
    }


@st.cache_resource
def load_models():
    return (
        joblib.load(MODELS / "intent_classifier.joblib"),
        joblib.load(MODELS / "sentiment_classifier.joblib"),
    )


@st.cache_resource
def load_retriever():
    faq = pd.read_csv(DATA / "faq_knowledge_base.csv")
    return FAQRetriever(faq, mode="lsa", answerability_threshold=0.22)


def metric_cards(items: list[tuple[str, str]], columns: int | None = None) -> None:
    slots = st.columns(columns or len(items))
    for slot, (label, value) in zip(slots, items):
        slot.metric(label, value)


data = load_data()
predictions: pd.DataFrame = data["predictions"]  # type: ignore[assignment]
crm: pd.DataFrame = data["crm"]  # type: ignore[assignment]
faq: pd.DataFrame = data["faq"]  # type: ignore[assignment]
taxonomy: pd.DataFrame = data["taxonomy"]  # type: ignore[assignment]
metrics: dict = data["metrics"]  # type: ignore[assignment]
sentiment_metrics: dict = data["sentiment_metrics"]  # type: ignore[assignment]
intent_model, sentiment_model = load_models()

st.title("Digital Pharmacy Voice-of-the-Customer Workbench · V2")
st.caption("Synthetic multi-channel contacts · NLP benchmarking · taxonomy evolution · safe automation · CRM insight · governance")
st.info("Portfolio demonstration only. All contacts and CRM outcomes are synthetic. This is not a Pharmacy2U internal system and it does not use patient records.")

with st.sidebar:
    st.header("Filters")
    date_min, date_max = predictions["timestamp"].min().date(), predictions["timestamp"].max().date()
    dates = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    channels = st.multiselect("Channel", sorted(predictions["channel"].unique()), default=sorted(predictions["channel"].unique()))
    categories = st.multiselect("Category", sorted(predictions["category"].unique()), default=sorted(predictions["category"].unique()))
    service_lines = st.multiselect("Service line", sorted(predictions["service_line"].unique()), default=sorted(predictions["service_line"].unique()))
    threshold = st.slider("Auto-routing confidence threshold", 0.40, 0.95, 0.72, 0.01)
    st.caption("Threshold changes routing simulation only. It does not retrain a model.")

if isinstance(dates, tuple) and len(dates) == 2:
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
else:
    start, end = predictions["timestamp"].min(), predictions["timestamp"].max() + pd.Timedelta(days=1)

filtered = predictions[
    (predictions["timestamp"] >= start)
    & (predictions["timestamp"] < end)
    & predictions["channel"].isin(channels)
    & predictions["category"].isin(categories)
    & predictions["service_line"].isin(service_lines)
].copy()
filtered = add_routes(filtered, threshold)
scenario = deflection_scenario(filtered, threshold)

pages = [
    "Executive overview",
    "Journey and root causes",
    "Emerging issues",
    "Taxonomy evolution",
    "Contact deflection",
    "Model benchmark",
    "Annotation workflow",
    "Satisfaction and retention drivers",
    "Data quality and governance",
    "Chatbot knowledge-base readiness",
]
page = st.sidebar.radio("Page", pages)

if page == "Executive overview":
    metric_cards([
        ("Contacts", f"{len(filtered):,}"),
        ("Predicted negative sentiment", f"{(filtered['predicted_sentiment'] == 'negative').mean():.1%}" if len(filtered) else "0.0%"),
        ("Repeat-contact flag", f"{filtered['repeat_contact_14d'].mean():.1%}" if len(filtered) else "0.0%"),
        ("Potential deflection", f"{scenario['potential_deflection_rate']:.1%}"),
        ("Clinical escalation", f"{scenario['clinical_escalation']:,}"),
    ])
    left, right = st.columns(2)
    top = filtered["predicted_intent"].value_counts().head(12).rename_axis("intent").reset_index(name="contacts")
    left.plotly_chart(px.bar(top.sort_values("contacts"), x="contacts", y="intent", orientation="h", title="Top predicted contact drivers"), width="stretch")
    routes = filtered["recommended_route"].value_counts().rename_axis("route").reset_index(name="contacts")
    right.plotly_chart(px.pie(routes, names="route", values="contacts", hole=0.5, title="Recommended contact routes"), width="stretch")
    weekly = filtered.assign(week=filtered["timestamp"].dt.to_period("W-MON").dt.start_time).groupby(["week", "category"]).size().rename("contacts").reset_index()
    st.plotly_chart(px.line(weekly, x="week", y="contacts", color="category", title="Weekly contacts by category"), width="stretch")

elif page == "Journey and root causes":
    stage = filtered.groupby(["journey_stage", "root_cause"]).size().rename("contacts").reset_index()
    st.plotly_chart(px.treemap(stage, path=["journey_stage", "root_cause"], values="contacts", title="Contact drivers across the patient journey"), width="stretch")
    left, right = st.columns(2)
    channel = filtered.groupby(["channel", "category"]).size().rename("contacts").reset_index()
    left.plotly_chart(px.bar(channel, x="channel", y="contacts", color="category", barmode="stack", title="Channel mix by contact category"), width="stretch")
    driver_outcomes = filtered.groupby("predicted_intent").agg(contacts=("contact_id", "size"), repeat_contact_rate=("repeat_contact_14d", "mean"), negative_sentiment_rate=("predicted_sentiment", lambda x: float((x == "negative").mean()))).reset_index()
    right.plotly_chart(px.scatter(driver_outcomes, x="repeat_contact_rate", y="negative_sentiment_rate", size="contacts", hover_name="predicted_intent", title="Contact-driver outcome signals"), width="stretch")
    st.dataframe(stage.sort_values("contacts", ascending=False), width="stretch", hide_index=True)

elif page == "Emerging issues":
    issues = emerging_issues(filtered)
    alerts = issues[issues["alert"]] if not issues.empty else issues
    metric_cards([("Flagged emerging issues", f"{len(alerts):,}"), ("Evaluation window", "Recent 3 weeks")], columns=2)
    st.caption("The monitor compares the recent multi-week share with an earlier baseline. Synthetic data includes an injected delivery-delay rise for validation.")
    st.dataframe(issues, width="stretch", hide_index=True)
    trends = weekly_intent_trends(filtered)
    defaults = list(issues.head(4)["intent"]) if not issues.empty else []
    selected = st.multiselect("Plot intent trends", sorted(trends["predicted_intent"].unique()), default=defaults)
    chart = trends[trends["predicted_intent"].isin(selected)]
    st.plotly_chart(px.line(chart, x="week", y="share", color="predicted_intent", markers=True, title="Weekly contact-driver share"), width="stretch")

elif page == "Taxonomy evolution":
    clusters: pd.DataFrame = data["topic_clusters"]  # type: ignore[assignment]
    examples: pd.DataFrame = data["topic_examples"]  # type: ignore[assignment]
    metric_cards([
        ("Unknown-topic clusters", f"{len(clusters):,}"),
        ("Suggested taxonomy actions", f"{clusters['recommended_review_action'].eq('Create or refine intent').sum():,}"),
        ("Current taxonomy version", "v2.0"),
    ])
    st.caption("The default offline workflow clusters contacts predicted as unclear or out-of-scope. It uses TF-IDF vectors projected into a local latent-semantic embedding space. The synthetic novel-pattern rate is a validation-only field and would not exist in operational data.")
    display_clusters = clusters.rename(columns={"synthetic_novel_pattern_rate": "demo_novel_pattern_rate"})
    st.dataframe(display_clusters, width="stretch", hide_index=True)
    selected_cluster = st.selectbox("Inspect representative contacts", clusters["cluster_id"].tolist())
    st.dataframe(examples[examples["cluster_id"].eq(selected_cluster)], width="stretch", hide_index=True)
    st.subheader("Taxonomy version history")
    st.dataframe(data["taxonomy_versions"], width="stretch", hide_index=True)

elif page == "Contact deflection":
    c1, c2 = st.columns(2)
    contact_cost = c1.number_input("Assumed handled-contact value (£)", min_value=0.0, value=4.50, step=0.25)
    proactive_cost = c2.number_input("Assumed proactive-message cost (£)", min_value=0.0, value=0.12, step=0.01)
    scenario = deflection_scenario(filtered, threshold, contact_cost, proactive_cost)
    metric_cards([
        ("Potentially deflected", f"{scenario['potential_deflected_contacts']:,}"),
        ("Deflection rate", f"{scenario['potential_deflection_rate']:.1%}"),
        ("Human adviser", f"{scenario['human_adviser']:,}"),
        ("Clinical escalation", f"{scenario['clinical_escalation']:,}"),
        ("Assumed net value", f"£{scenario['assumed_net_value_gbp']:,.0f}"),
    ])
    st.caption("Financial values are demonstration assumptions, not company financial results. Safety-sensitive language always overrides automated routing.")
    routes = add_routes(filtered, threshold)["recommended_route"].value_counts().rename_axis("route").reset_index(name="contacts")
    st.plotly_chart(px.bar(routes, x="route", y="contacts", title="Routing volume at the selected threshold"), width="stretch")
    scenario_rows = []
    for value in np.arange(0.40, 0.96, 0.05):
        row = deflection_scenario(filtered, float(value), contact_cost, proactive_cost)
        row["threshold"] = round(float(value), 2)
        scenario_rows.append(row)
    scenarios = pd.DataFrame(scenario_rows)
    st.plotly_chart(px.line(scenarios, x="threshold", y="potential_deflection_rate", markers=True, title="Potential deflection rate by confidence threshold"), width="stretch")

elif page == "Model benchmark":
    benchmark: pd.DataFrame = data["benchmark"]  # type: ignore[assignment]
    channel_metrics: pd.DataFrame = data["channel_metrics"]  # type: ignore[assignment]
    reliability: pd.DataFrame = data["reliability"]  # type: ignore[assignment]
    metric_cards([
        ("Packaged intent Macro-F1", f"{metrics['macro_f1']:.3f}"),
        ("Temporal benchmark Macro-F1", f"{benchmark[(benchmark['model'] == 'tfidf_char_word_sgd') & (benchmark['split'] == 'temporal_holdout')]['macro_f1'].iloc[0]:.3f}"),
        ("Expected calibration error", f"{metrics['expected_calibration_error']:.3f}"),
        ("Sentiment Macro-F1", f"{sentiment_metrics['macro_f1']:.3f}"),
    ])
    st.caption("Metrics are held-out synthetic-data checks. They validate the workflow and are not production estimates. The packaged model uses character and word TF-IDF features for spelling-noise robustness and fast local rebuilds.")
    st.plotly_chart(px.bar(benchmark, x="model", y="macro_f1", color="split", barmode="group", title="Intent benchmark: random and temporal holdout"), width="stretch")
    left, right = st.columns(2)
    left.subheader("Channel-level intent metrics")
    left.dataframe(channel_metrics, width="stretch", hide_index=True)
    calibration = go.Figure()
    calibration.add_trace(go.Scatter(x=reliability["mean_confidence"], y=reliability["observed_accuracy"], mode="lines+markers", name="Observed"))
    calibration.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal"))
    calibration.update_layout(title="Reliability diagram", xaxis_title="Mean predicted confidence", yaxis_title="Observed accuracy")
    right.plotly_chart(calibration, width="stretch")
    st.subheader("Independent sentiment model")
    st.json({k: v for k, v in sentiment_metrics.items() if k != "classification_report"})
    st.caption("The sentiment model is intentionally evaluated separately from intent classification. Positive sentiment is the smallest synthetic class, which lowers Macro-F1 and illustrates why per-class review matters.")

elif page == "Annotation workflow":
    queue: pd.DataFrame = data["annotation_queue"]  # type: ignore[assignment]
    st.caption("Review priority combines low intent confidence, out-of-scope predictions, and clinical-escalation flags. Taxonomy version is stored with every reviewed row.")
    metric_cards([
        ("Review queue", f"{len(queue):,}"),
        ("Unreviewed", f"{queue['review_status'].eq('unreviewed').sum():,}"),
        ("Taxonomy version", str(queue["taxonomy_version"].iloc[0])),
    ])
    subset = queue.head(120).copy()
    edited = st.data_editor(
        subset,
        width="stretch",
        hide_index=True,
        column_config={
            "human_intent_label": st.column_config.SelectboxColumn("human_intent_label", options=sorted(taxonomy["intent"].tolist())),
            "human_sentiment_label": st.column_config.SelectboxColumn("human_sentiment_label", options=["negative", "neutral", "positive"]),
            "review_status": st.column_config.SelectboxColumn("review_status", options=["unreviewed", "reviewed", "needs_second_review"]),
        },
    )
    st.download_button("Download reviewed annotation queue", edited.to_csv(index=False), "reviewed_annotation_queue.csv", "text/csv")

elif page == "Satisfaction and retention drivers":
    linked: pd.DataFrame = data["crm_linked"]  # type: ignore[assignment]
    crm_metrics: dict = data["crm_driver_metrics"]  # type: ignore[assignment]
    coefficients: pd.DataFrame = data["crm_coefficients"]  # type: ignore[assignment]
    metric_cards([
        ("Linked synthetic CRM rows", f"{len(linked):,}"),
        ("Low-CSAT rate", f"{linked['low_csat'].mean():.1%}"),
        ("Not retained at 90 days", f"{linked['not_retained_90d'].mean():.1%}"),
        ("Low-CSAT ROC-AUC", f"{crm_metrics['low_csat']['roc_auc']:.3f}"),
    ])
    st.caption("These models describe associations in synthetic CRM data. They do not estimate causal effects and the results are not company estimates.")
    outcome = st.radio("Outcome model", ["low_csat", "not_retained_90d"], horizontal=True)
    top_coefficients = coefficients[coefficients["outcome"].eq(outcome)].head(16).sort_values("coefficient")
    st.plotly_chart(px.bar(top_coefficients, x="coefficient", y="feature", orientation="h", title=f"Interpretable logistic-regression drivers: {outcome}"), width="stretch")
    left, right = st.columns(2)
    intent_outcomes = linked.groupby("dominant_predicted_intent").agg(patients=("pseudo_patient_id", "size"), mean_csat=("csat_score", "mean"), low_csat_rate=("low_csat", "mean"), not_retained_rate=("not_retained_90d", "mean")).reset_index()
    left.plotly_chart(px.scatter(intent_outcomes, x="low_csat_rate", y="not_retained_rate", size="patients", hover_name="dominant_predicted_intent", title="Synthetic outcome signals by dominant intent"), width="stretch")
    right.json(crm_metrics[outcome])
    st.dataframe(intent_outcomes.sort_values("low_csat_rate", ascending=False), width="stretch", hide_index=True)

elif page == "Data quality and governance":
    governance: dict = data["governance"]  # type: ignore[assignment]
    counts: pd.DataFrame = data["redaction_counts"]  # type: ignore[assignment]
    quality: pd.DataFrame = data["quality_by_channel"]  # type: ignore[assignment]
    metric_cards([
        ("Contacts audited", f"{governance['contacts']:,}"),
        ("Contacts with redaction", f"{governance['contacts_with_redaction_rate']:.1%}"),
        ("Total redactions", f"{governance['total_redactions']:,}"),
        ("Short-text rate", f"{governance['short_text_rate']:.1%}"),
        ("Duplicate-contact rate", f"{governance['duplicate_contact_rate']:.1%}"),
    ])
    st.caption("The packaged redactor is local and rule-based so it is easy to audit. A production implementation would require reviewed rules, privacy assessment, access controls, and a stronger de-identification layer.")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(counts.sort_values("redactions"), x="redactions", y="entity_type", orientation="h", title="Rule-based redactions by entity type"), width="stretch")
    right.dataframe(quality, width="stretch", hide_index=True)
    st.subheader("Redaction audit sample")
    st.dataframe(data["redaction_audit"].head(30), width="stretch", hide_index=True)

elif page == "Chatbot knowledge-base readiness":
    st.subheader("Safe FAQ retrieval demonstration")
    query = st.text_input("Enter a customer message", "My GP has not approved my prescription yet. What should I do?")
    cleaned, redaction_count = clean_text(query)
    intent_proba = intent_model.predict_proba([cleaned])[0]
    predicted_intent = str(intent_model.classes_[int(np.argmax(intent_proba))])
    confidence = float(np.max(intent_proba))
    sentiment_proba = sentiment_model.predict_proba([cleaned])[0]
    predicted_sentiment = str(sentiment_model.classes_[int(np.argmax(sentiment_proba))])
    route = assign_route(predicted_intent, confidence, cleaned, threshold)
    retriever = load_retriever()
    answerable = retriever.is_answerable(cleaned) and route not in {"Clinical escalation", "Human adviser"}
    metric_cards([
        ("Predicted intent", predicted_intent),
        ("Confidence", f"{confidence:.1%}"),
        ("Predicted sentiment", predicted_sentiment),
        ("Recommended route", route),
        ("FAQ answerable", "Yes" if answerable else "No"),
    ])
    st.caption(f"FAQ search mode: {retriever.mode} · rule-based redactions in query: {redaction_count}")
    if not answerable:
        st.warning("The current rules do not allow an automated FAQ answer. Use the recommended human or clinical route, or review knowledge-base coverage.")
    for result in retriever.search(cleaned, top_k=3):
        with st.expander(f"{result.question} · similarity {result.score:.2f}"):
            st.write(result.answer)
            st.caption(f"Mapped intent: {result.mapped_intent}")
    st.subheader("Retrieval-quality benchmark")
    st.dataframe(data["faq_metrics"], width="stretch", hide_index=True)
    st.caption("The local semantic baseline remains modest on a deliberately small knowledge base. Optional Sentence Transformers and Hugging Face scripts are included for a connected environment; no transformer score is claimed in the packaged build.")
    st.subheader("Intent taxonomy")
    st.dataframe(taxonomy, width="stretch", hide_index=True)
