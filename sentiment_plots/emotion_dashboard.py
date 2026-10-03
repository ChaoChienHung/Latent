"""
Streamlit emotion analysis dashboard.

Usage:
    uv run --no-sync --with streamlit --with plotly streamlit run emotion_dashboard.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CSV = "intermediate_data/stopword_lemmatized_posts_0_labels_w_emot.csv"

EMOTION_COLORS = {
    "anger": "#d62728",
    "disgust": "#8c564b",
    "fear": "#9467bd",
    "joy": "#2ca02c",
    "neutral": "#7f7f7f",
    "sadness": "#1f77b4",
    "surprise": "#ff7f0e",
}
EMOTIONS = list(EMOTION_COLORS.keys())
PROB_COLS = [f"prob_{e}" for e in EMOTIONS]
BUCKET_ORDER = ["negative", "low", "medium", "high", "viral"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(
    page_title="r/singapore Emotion Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year_month"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df["month_date"] = pd.to_datetime(df["year_month"] + "-01", errors="coerce")
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.Categorical(
            df["day_of_week"], categories=DOW_ORDER, ordered=True
        )
    if "score_bucket" in df.columns:
        df["score_bucket"] = pd.Categorical(
            df["score_bucket"], categories=BUCKET_ORDER, ordered=True
        )
    # normalise flair duplicates
    if "link_flair_text" in df.columns:
        df["link_flair_text"] = (
            df["link_flair_text"]
            .str.strip()
            .replace({"Opinion / Fluff Post": "Opinion/Fluff Post"})
        )
    return df


# ---------------------------------------------------------------------------
# Sidebar – filters
# ---------------------------------------------------------------------------

df_raw = load_data(DEFAULT_CSV)

st.sidebar.title("Filters")

year_options = sorted(df_raw["year"].unique())
selected_years = st.sidebar.multiselect("Year", year_options, default=year_options)

month_options = sorted(df_raw["month"].unique())
selected_months = st.sidebar.multiselect(
    "Month", month_options, default=month_options,
    format_func=lambda m: pd.Timestamp(f"2000-{m:02d}-01").strftime("%b"),
)

emotion_options = sorted(df_raw["predicted_emotion"].dropna().unique())
selected_emotions = st.sidebar.multiselect(
    "Emotion", emotion_options, default=emotion_options
)

if "score_bucket" in df_raw.columns:
    bucket_options = [b for b in BUCKET_ORDER if b in df_raw["score_bucket"].cat.categories]
    selected_buckets = st.sidebar.multiselect(
        "Score Bucket", bucket_options, default=bucket_options
    )
else:
    selected_buckets = []

if "link_flair_text" in df_raw.columns:
    flair_options = sorted(df_raw["link_flair_text"].dropna().unique())
    top_flairs = (
        df_raw["link_flair_text"].value_counts().head(10).index.tolist()
    )
    selected_flairs = st.sidebar.multiselect(
        "Post Flair (top 10 pre-selected)", flair_options, default=top_flairs
    )
else:
    selected_flairs = []


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["year"].isin(selected_years)
        & df["month"].isin(selected_months)
        & df["predicted_emotion"].isin(selected_emotions)
    )
    if selected_buckets and "score_bucket" in df.columns:
        mask &= df["score_bucket"].isin(selected_buckets)
    if selected_flairs and "link_flair_text" in df.columns:
        mask &= df["link_flair_text"].isin(selected_flairs)
    return df[mask].copy()


df = apply_filters(df_raw)

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------

st.title("📊 r/singapore Emotion Analysis Dashboard")
st.caption(f"Source: `{DEFAULT_CSV}` · {len(df):,} posts shown (of {len(df_raw):,} total)")

c1, c2, c3, c4 = st.columns(4)

dominant = df["predicted_emotion"].value_counts().idxmax() if len(df) else "—"
dominant_pct = df["predicted_emotion"].value_counts(normalize=True).max() * 100 if len(df) else 0
avg_conf = df[[c for c in PROB_COLS if c in df.columns]].max(axis=1).mean() * 100 if len(df) else 0
months_span = df["year_month"].nunique()

c1.metric("Total Posts", f"{len(df):,}")
c2.metric("Dominant Emotion", f"{dominant} ({dominant_pct:.1f}%)")
c3.metric("Avg Model Confidence", f"{avg_conf:.1f}%")
c4.metric("Months Covered", months_span)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_time, tab_temporal, tab_engagement, tab_confidence, tab_explore = st.tabs([
    "Overview", "Time Trends", "Temporal Patterns", "Engagement", "Confidence", "Explore Posts",
])

# ── Tab 1: Overview ──────────────────────────────────────────────────────────
with tab_overview:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Emotion Distribution")
        counts = df["predicted_emotion"].value_counts().reset_index()
        counts.columns = ["emotion", "count"]
        counts["share"] = counts["count"] / counts["count"].sum()
        counts["color"] = counts["emotion"].map(EMOTION_COLORS)

        fig = px.bar(
            counts.sort_values("count"),
            x="count", y="emotion",
            orientation="h",
            color="emotion",
            color_discrete_map=EMOTION_COLORS,
            text=counts.sort_values("count")["share"].map(lambda x: f"{x:.1%}"),
            labels={"count": "Number of posts", "emotion": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=60, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Emotion Share")
        fig2 = px.pie(
            counts,
            values="count",
            names="emotion",
            color="emotion",
            color_discrete_map=EMOTION_COLORS,
            hole=0.45,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Emotion by Post Flair")
    if "link_flair_text" in df.columns:
        top_flair = df["link_flair_text"].value_counts().head(10).index
        flair_df = (
            df[df["link_flair_text"].isin(top_flair)]
            .groupby(["link_flair_text", "predicted_emotion"])
            .size()
            .reset_index(name="count")
        )
        flair_df["share"] = flair_df.groupby("link_flair_text")["count"].transform(
            lambda x: x / x.sum()
        )
        fig3 = px.bar(
            flair_df,
            x="link_flair_text", y="share",
            color="predicted_emotion",
            color_discrete_map=EMOTION_COLORS,
            barmode="stack",
            labels={"link_flair_text": "Post Flair", "share": "Share", "predicted_emotion": "Emotion"},
        )
        fig3.update_layout(height=380, xaxis_tickangle=-30, margin=dict(t=10))
        fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 2: Time Trends ────────────────────────────────────────────────────────
with tab_time:
    monthly = (
        df.groupby(["month_date", "year_month", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    monthly["share"] = monthly.groupby("month_date")["count"].transform(
        lambda x: x / x.sum()
    )
    monthly = monthly.sort_values("month_date")

    view = st.radio("View as", ["Share (%)", "Count"], horizontal=True)
    y_col = "share" if view == "Share (%)" else "count"
    y_fmt = ".0%" if view == "Share (%)" else None

    st.subheader("Monthly Emotion Trends (Line)")
    fig_line = px.line(
        monthly,
        x="month_date", y=y_col,
        color="predicted_emotion",
        color_discrete_map=EMOTION_COLORS,
        markers=True,
        labels={"month_date": "Month", y_col: view, "predicted_emotion": "Emotion"},
    )
    if y_fmt:
        fig_line.update_yaxes(tickformat=y_fmt)
    fig_line.update_layout(height=420, margin=dict(t=10))
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Monthly Emotion Composition (Stacked Area)")
    pivot = (
        monthly.pivot(index="month_date", columns="predicted_emotion", values=y_col)
        .fillna(0)
        .sort_index()
        .reset_index()
    )
    emotion_cols_present = [e for e in EMOTIONS if e in pivot.columns]
    fig_area = go.Figure()
    for emotion in emotion_cols_present:
        fig_area.add_trace(go.Scatter(
            x=pivot["month_date"], y=pivot[emotion],
            stackgroup="one",
            name=emotion,
            mode="lines",
            line=dict(width=0.5, color=EMOTION_COLORS.get(emotion)),
            fillcolor=EMOTION_COLORS.get(emotion),
        ))
    if y_fmt:
        fig_area.update_yaxes(tickformat=y_fmt)
    fig_area.update_layout(height=380, margin=dict(t=10), hovermode="x unified")
    st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("Emotion Share Heatmap (Month × Emotion)")
    heat_pivot = (
        monthly.pivot(index="year_month", columns="predicted_emotion", values="share")
        .fillna(0)
        .sort_index()
    )
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        text_auto=".1%",
        labels={"x": "Emotion", "y": "Month", "color": "Share"},
    )
    fig_heat.update_layout(height=max(300, len(heat_pivot) * 28), margin=dict(t=10))
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 3: Temporal Patterns ──────────────────────────────────────────────────
with tab_temporal:
    col_dow, col_hr = st.columns(2)

    with col_dow:
        st.subheader("By Day of Week")
        if "day_of_week" in df.columns:
            dow_df = (
                df.groupby(["day_of_week", "predicted_emotion"])
                .size()
                .reset_index(name="count")
            )
            dow_df["share"] = dow_df.groupby("day_of_week")["count"].transform(
                lambda x: x / x.sum()
            )
            fig_dow = px.bar(
                dow_df,
                x="day_of_week", y="share",
                color="predicted_emotion",
                color_discrete_map=EMOTION_COLORS,
                barmode="stack",
                category_orders={"day_of_week": DOW_ORDER},
                labels={"day_of_week": "", "share": "Share", "predicted_emotion": "Emotion"},
            )
            fig_dow.update_yaxes(tickformat=".0%")
            fig_dow.update_layout(height=380, margin=dict(t=10))
            st.plotly_chart(fig_dow, use_container_width=True)

    with col_hr:
        st.subheader("By Hour of Day (SGT)")
        if "hour" in df.columns:
            hour_df = (
                df.groupby(["hour", "predicted_emotion"])
                .size()
                .reset_index(name="count")
            )
            hour_df["share"] = hour_df.groupby("hour")["count"].transform(
                lambda x: x / x.sum()
            )
            fig_hr = px.bar(
                hour_df,
                x="hour", y="share",
                color="predicted_emotion",
                color_discrete_map=EMOTION_COLORS,
                barmode="stack",
                labels={"hour": "Hour (SGT)", "share": "Share", "predicted_emotion": "Emotion"},
            )
            fig_hr.update_xaxes(dtick=2)
            fig_hr.update_yaxes(tickformat=".0%")
            fig_hr.update_layout(height=380, margin=dict(t=10))
            st.plotly_chart(fig_hr, use_container_width=True)

    st.subheader("Emotion Heatmap: Hour × Day of Week")
    if "hour" in df.columns and "day_of_week" in df.columns:
        sel_emotion = st.selectbox(
            "Emotion to show", ["All (dominant)"] + emotion_options, key="heatmap_emotion"
        )
        if sel_emotion == "All (dominant)":
            heat_df = (
                df.groupby(["day_of_week", "hour"])["predicted_emotion"]
                .agg(lambda x: x.value_counts().idxmax())
                .reset_index()
                .rename(columns={"predicted_emotion": "dominant_emotion"})
            )
            heat_pivot2 = heat_df.pivot(
                index="day_of_week", columns="hour", values="dominant_emotion"
            ).reindex(DOW_ORDER)
            # encode as numeric for colour, label with text
            emotion_idx = {e: i for i, e in enumerate(EMOTIONS)}
            heat_num = heat_pivot2.applymap(lambda v: emotion_idx.get(v, -1) if pd.notna(v) else -1)
            fig_h2 = px.imshow(
                heat_num,
                color_continuous_scale=list(EMOTION_COLORS.values()),
                aspect="auto",
                labels={"x": "Hour (SGT)", "y": "Day"},
            )
            # overlay text
            for y_i, dow in enumerate(heat_pivot2.index):
                for x_i, hr in enumerate(heat_pivot2.columns):
                    val = heat_pivot2.iloc[y_i, x_i]
                    if pd.notna(val):
                        fig_h2.add_annotation(
                            x=x_i, y=y_i, text=str(val)[:3],
                            showarrow=False, font=dict(size=8, color="white"),
                        )
            fig_h2.update_coloraxes(showscale=False)
            fig_h2.update_layout(height=260, margin=dict(t=10))
            st.plotly_chart(fig_h2, use_container_width=True)
        else:
            heat_df2 = (
                df[df["predicted_emotion"] == sel_emotion]
                .groupby(["day_of_week", "hour"])
                .size()
                .reset_index(name="count")
            )
            heat_pivot3 = heat_df2.pivot(
                index="day_of_week", columns="hour", values="count"
            ).fillna(0).reindex(DOW_ORDER)
            fig_h3 = px.imshow(
                heat_pivot3,
                color_continuous_scale="Blues",
                aspect="auto",
                text_auto=True,
                labels={"x": "Hour (SGT)", "y": "Day", "color": "Posts"},
            )
            fig_h3.update_layout(height=260, margin=dict(t=10))
            st.plotly_chart(fig_h3, use_container_width=True)

# ── Tab 4: Engagement ────────────────────────────────────────────────────────
with tab_engagement:
    st.subheader("Emotion by Score Bucket")
    if "score_bucket" in df.columns:
        bkt_df = (
            df.groupby(["score_bucket", "predicted_emotion"])
            .size()
            .reset_index(name="count")
        )
        bkt_df["share"] = bkt_df.groupby("score_bucket")["count"].transform(
            lambda x: x / x.sum()
        )
        fig_bkt = px.bar(
            bkt_df,
            x="score_bucket", y="share",
            color="predicted_emotion",
            color_discrete_map=EMOTION_COLORS,
            barmode="stack",
            category_orders={"score_bucket": BUCKET_ORDER},
            labels={"score_bucket": "Score Bucket", "share": "Share", "predicted_emotion": "Emotion"},
        )
        fig_bkt.update_yaxes(tickformat=".0%")
        fig_bkt.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig_bkt, use_container_width=True)

    col_sc, col_cm = st.columns(2)

    with col_sc:
        st.subheader("Score Distribution by Emotion")
        fig_box = px.box(
            df[df["score"] < df["score"].quantile(0.95)],
            x="predicted_emotion", y="score",
            color="predicted_emotion",
            color_discrete_map=EMOTION_COLORS,
            points=False,
            category_orders={"predicted_emotion": emotion_options},
            labels={"predicted_emotion": "Emotion", "score": "Score (≤p95)"},
        )
        fig_box.update_layout(showlegend=False, height=380, margin=dict(t=10))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_cm:
        st.subheader("Comments Distribution by Emotion")
        fig_box2 = px.box(
            df[df["num_comments"] < df["num_comments"].quantile(0.95)],
            x="predicted_emotion", y="num_comments",
            color="predicted_emotion",
            color_discrete_map=EMOTION_COLORS,
            points=False,
            category_orders={"predicted_emotion": emotion_options},
            labels={"predicted_emotion": "Emotion", "num_comments": "Comments (≤p95)"},
        )
        fig_box2.update_layout(showlegend=False, height=380, margin=dict(t=10))
        st.plotly_chart(fig_box2, use_container_width=True)

    st.subheader("Upvote Ratio by Emotion")
    fig_vio = px.violin(
        df, x="predicted_emotion", y="upvote_ratio",
        color="predicted_emotion",
        color_discrete_map=EMOTION_COLORS,
        box=True,
        category_orders={"predicted_emotion": emotion_options},
        labels={"predicted_emotion": "Emotion", "upvote_ratio": "Upvote Ratio"},
    )
    fig_vio.update_layout(showlegend=False, height=380, margin=dict(t=10))
    st.plotly_chart(fig_vio, use_container_width=True)

# ── Tab 5: Model Confidence ───────────────────────────────────────────────────
with tab_confidence:
    prob_cols_present = [c for c in PROB_COLS if c in df.columns]

    if prob_cols_present:
        df_conf = df.copy()
        df_conf["max_prob"] = df_conf[prob_cols_present].max(axis=1)

        st.subheader("Confidence (max prob) by Predicted Emotion")
        fig_conf = px.violin(
            df_conf,
            x="predicted_emotion", y="max_prob",
            color="predicted_emotion",
            color_discrete_map=EMOTION_COLORS,
            box=True,
            points=False,
            category_orders={"predicted_emotion": emotion_options},
            labels={"predicted_emotion": "Emotion", "max_prob": "Max Class Probability"},
        )
        fig_conf.update_yaxes(tickformat=".0%", range=[0, 1])
        fig_conf.update_layout(showlegend=False, height=400, margin=dict(t=10))
        st.plotly_chart(fig_conf, use_container_width=True)

        st.subheader("Average Probability per Emotion Class")
        mean_probs = (
            df_conf[prob_cols_present].mean()
            .rename(index=lambda c: c.replace("prob_", ""))
            .reset_index()
        )
        mean_probs.columns = ["emotion", "avg_prob"]
        mean_probs["color"] = mean_probs["emotion"].map(EMOTION_COLORS)
        fig_avg = px.bar(
            mean_probs.sort_values("avg_prob", ascending=False),
            x="emotion", y="avg_prob",
            color="emotion",
            color_discrete_map=EMOTION_COLORS,
            labels={"emotion": "Emotion", "avg_prob": "Average Probability"},
            text_auto=".1%",
        )
        fig_avg.update_yaxes(tickformat=".0%")
        fig_avg.update_layout(showlegend=False, height=360, margin=dict(t=10))
        st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("Probability Correlation Heatmap")
        corr = df_conf[prob_cols_present].rename(
            columns=lambda c: c.replace("prob_", "")
        ).corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            labels={"color": "Correlation"},
        )
        fig_corr.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No probability columns found in dataset.")

# ── Tab 6: Explore Posts ──────────────────────────────────────────────────────
with tab_explore:
    st.subheader("Sample Posts by Emotion")

    col_e, col_n = st.columns([2, 1])
    with col_e:
        sel_em = st.selectbox("Emotion", emotion_options, key="explore_emotion")
    with col_n:
        n_samples = st.slider("Posts to show", 5, 50, 10)

    sort_by = st.radio(
        "Sort by", ["Random", "Highest Score", "Lowest Score", "Most Comments"],
        horizontal=True,
    )

    subset = df[df["predicted_emotion"] == sel_em].copy()
    if sort_by == "Random":
        subset = subset.sample(min(n_samples, len(subset)), random_state=42)
    elif sort_by == "Highest Score":
        subset = subset.nlargest(n_samples, "score")
    elif sort_by == "Lowest Score":
        subset = subset.nsmallest(n_samples, "score")
    else:
        subset = subset.nlargest(n_samples, "num_comments")

    display_cols = [
        c for c in ["title", "predicted_emotion", "score", "num_comments",
                     "upvote_ratio", "year_month", "link_flair_text", "max_prob"]
        if c in subset.columns or c == "max_prob"
    ]
    prob_cols_p = [c for c in PROB_COLS if c in subset.columns]
    if prob_cols_p:
        subset["max_prob"] = subset[prob_cols_p].max(axis=1).map("{:.1%}".format)

    st.dataframe(
        subset[[c for c in display_cols if c in subset.columns]].reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    st.subheader("Emotion Distribution within Filtered Set")
    filtered_counts = df["predicted_emotion"].value_counts().reset_index()
    filtered_counts.columns = ["emotion", "count"]
    fig_fc = px.bar(
        filtered_counts,
        x="emotion", y="count",
        color="emotion",
        color_discrete_map=EMOTION_COLORS,
        text_auto=True,
        labels={"emotion": "Emotion", "count": "Posts"},
    )
    fig_fc.update_layout(showlegend=False, height=320, margin=dict(t=10))
    st.plotly_chart(fig_fc, use_container_width=True)
