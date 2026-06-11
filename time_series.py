import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEGMENT_COLORS = {
    "Şampiyonlar": "#2563EB",
    "Sadık Müşteriler": "#16A34A",
    "Risk Altındakiler": "#D4845A",
    "Kaybedilenler": "#9CA3AF",
    "Diğer": "#CCCCCC",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="white",
    font=dict(family="IBM Plex Sans", size=12, color="#0D0D0D"),
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(showgrid=False, linecolor="#0D0D0D", linewidth=1),
    yaxis=dict(gridcolor="#F0EBE2", gridwidth=1, linecolor="#0D0D0D", linewidth=1),
    hovermode="x unified",
)


def compute_monthly_metrics(raw_df: pd.DataFrame, rfm_df: pd.DataFrame) -> dict:
    df = raw_df.copy()
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Segment bilgisini her işleme ekle
    df = df.merge(rfm_df[["Segment"]], left_on="CustomerID", right_index=True, how="left")
    df["Segment"] = df["Segment"].fillna("Diğer")

    monthly_revenue = (
        df.groupby("Month")["TotalPrice"].sum()
        .reset_index(name="Revenue")
    )

    monthly_customers = (
        df.groupby("Month")["CustomerID"].nunique()
        .reset_index(name="Customers")
    )

    monthly_orders = (
        df.groupby("Month")["InvoiceNo"].nunique()
        .reset_index(name="Orders")
    )

    segment_monthly = (
        df.groupby(["Month", "Segment"])["TotalPrice"].sum()
        .reset_index(name="Revenue")
    )

    # Birleştir
    summary = monthly_revenue.merge(monthly_customers, on="Month").merge(monthly_orders, on="Month")
    summary["AOV"] = (summary["Revenue"] / summary["Orders"]).round(2)  # avg order value
    summary["Revenue_Growth"] = summary["Revenue"].pct_change().mul(100).round(1)

    return {
        "summary": summary,
        "segment_monthly": segment_monthly,
    }


def build_revenue_chart(summary: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["Month"], y=summary["Revenue"],
        mode="lines+markers",
        line=dict(color="#1C3D2E", width=2.5),
        marker=dict(size=6, color="#1C3D2E"),
        fill="tozeroy",
        fillcolor="rgba(28,61,46,0.08)",
        name="Gelir (£)",
        hovertemplate="£%{y:,.0f}<extra></extra>",
    ))
    layout = {**CHART_LAYOUT}
    layout["yaxis"] = {**layout.get("yaxis", {}), "tickprefix": "£", "tickformat": ",.0f"}
    fig.update_layout(**layout)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_customers_chart(summary: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["Month"], y=summary["Customers"],
        marker_color="#2563EB",
        marker_line_color="#0D0D0D", marker_line_width=1,
        name="Aktif Müşteri",
        hovertemplate="%{y} müşteri<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_segment_area_chart(segment_monthly: pd.DataFrame) -> str:
    fig = go.Figure()
    for segment, color in SEGMENT_COLORS.items():
        seg_data = segment_monthly[segment_monthly["Segment"] == segment]
        if seg_data.empty:
            continue
        fig.add_trace(go.Scatter(
            x=seg_data["Month"], y=seg_data["Revenue"],
            name=segment,
            mode="lines",
            stackgroup="one",
            line=dict(color=color, width=1),
            fillcolor=color,
            hovertemplate=f"{segment}: £%{{y:,.0f}}<extra></extra>",
        ))
    layout = {**CHART_LAYOUT}
    layout["yaxis"] = {**layout.get("yaxis", {}), "tickprefix": "£", "tickformat": ",.0f"}
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    fig.update_layout(**layout)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_aov_chart(summary: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["Month"], y=summary["AOV"],
        mode="lines+markers",
        line=dict(color="#D4845A", width=2.5),
        marker=dict(size=6, color="#D4845A"),
        name="AOV (£)",
        hovertemplate="£%{y:,.2f}<extra></extra>",
    ))
    layout = {**CHART_LAYOUT}
    layout["yaxis"] = {**layout.get("yaxis", {}), "tickprefix": "£"}
    fig.update_layout(**layout)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def get_key_insights(summary: pd.DataFrame) -> dict:
    peak_month = summary.loc[summary["Revenue"].idxmax()]
    worst_month = summary.loc[summary["Revenue"].idxmin()]
    best_growth = summary.loc[summary["Revenue_Growth"].idxmax()]
    avg_customers = int(summary["Customers"].mean())

    return {
        "peak_month": peak_month["Month"],
        "peak_revenue": int(peak_month["Revenue"]),
        "worst_month": worst_month["Month"],
        "worst_revenue": int(worst_month["Revenue"]),
        "best_growth_month": best_growth["Month"],
        "best_growth_pct": float(best_growth["Revenue_Growth"]),
        "avg_monthly_customers": avg_customers,
        "total_months": len(summary),
    }
