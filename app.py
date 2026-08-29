import streamlit as st
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="Abuse-Ring Sentinel", layout="wide", page_icon="🕸️")

# --- Design pass: teal/near-black theme, Space Grotesk for headers, JetBrains
# Mono for data. Plain CSS + one Google Fonts import — no JS, nothing exotic. ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
    code, pre, [data-testid="stCode"] { font-family: 'JetBrains Mono', monospace !important; }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(45,212,191,0.07), transparent 45%),
            radial-gradient(circle at 85% 30%, rgba(56,189,248,0.05), transparent 40%),
            #0A0D13;
    }

    @keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .status-bar {
        display: flex; align-items: center; gap: 10px;
        font-family: 'JetBrains Mono', monospace; font-size: 12px;
        letter-spacing: 0.10em; color: #7D8797; text-transform: uppercase;
        margin-bottom: 18px;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #2DD4BF;
        box-shadow: 0 0 10px 2px rgba(45,212,191,0.6);
        animation: pulse-dot 2.4s ease-in-out infinite;
    }

    .hero { display: flex; align-items: center; gap: 16px; margin-bottom: 28px; }
    .hero-icon {
        width: 52px; height: 52px; border-radius: 14px; flex: none;
        background: linear-gradient(135deg,#0f2b28,#0a1a19);
        border: 1px solid rgba(45,212,191,0.25);
        display: flex; align-items: center; justify-content: center;
    }
    .hero h1 {
        margin: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        font-size: 32px; letter-spacing: -0.01em; color: #f5f7fa;
    }
    .hero p { margin: 4px 0 0; font-size: 14px; color: #8b95a3; }

    [data-testid="stMetric"] {
        background-color: #0F131B;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 16px 18px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(45,212,191,0.4);
    }
    [data-testid="stMetricLabel"] { font-size: 12px; opacity: 0.65; letter-spacing: 0.02em; }
    [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetricDelta"] {
        background: rgba(74,222,128,0.10); color: #4ade80 !important;
        padding: 2px 8px; border-radius: 20px; font-size: 11.5px; font-weight: 600;
        display: inline-block; margin-top: 4px;
    }

    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 100px;
        font-size: 11.5px; font-weight: 600; margin-left: 10px; vertical-align: middle;
        border: 1px solid transparent;
    }
    .badge-tp { background: rgba(74,222,128,0.12); color: #4ade80; border-color: rgba(74,222,128,0.25); }
    .badge-fp { background: rgba(250,178,25,0.14); color: #ffd88a; border-color: rgba(250,178,25,0.3); }
    .badge-fn { background: rgba(251,113,133,0.14); color: #fb7185; border-color: rgba(251,113,133,0.3); }
    .badge-tn { background: rgba(255,255,255,0.08); color: #c3c2b7; border-color: rgba(255,255,255,0.15); }

    .swatch {
        display: inline-block; width: 10px; height: 10px; border-radius: 3px;
        margin-right: 8px; vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

with open("dashboard_data.json") as f:
    data = json.load(f)

summary = data["summary"]
clusters = data["clusters"]
graph_data = data["graph"]

BADGE_LABELS = {
    "TP": ("badge-tp", "confirmed catch"),
    "FP": ("badge-fp", "false alarm"),
    "FN": ("badge-fn", "missed"),
    "TN": ("badge-tn", "correctly ignored"),
}

# Used by both the graph AND the cluster-header swatch below
def ring_color(ring_id):
    if ring_id is None:
        return "#7D8797"
    if ring_id.startswith("promo_ring"):
        return "#fb7185"
    if ring_id.startswith("return_ring"):
        return "#38bdf8"
    if ring_id.startswith("mule_ring"):
        return "#4ade80"
    if ring_id.startswith("legit_cluster"):
        return "#fbbf24"
    return "white"

def fmt_rs(n):
    """Indian short currency notation — ₹15.8L instead of ₹1,580,800, so it
    actually fits in a metric tile instead of getting cut off with '...'."""
    if abs(n) >= 100000:
        return f"₹{n/100000:.1f}L"
    return f"₹{n:,.0f}"

top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("""
    <div class="status-bar"><span class="status-dot"></span>snapshot · training/evaluation run</div>
    """, unsafe_allow_html=True)
with top_right:
    st.download_button(
        "Export report (CSV)",
        data=pd.DataFrame(clusters)[["size", "confidence", "density", "avg_weight", "true_label", "outcome"]].to_csv(index=False),
        file_name="flagged_clusters.csv",
        mime="text/csv",
    )

st.markdown("""
<div class="hero">
  <div class="hero-icon">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="5" r="1.6" stroke="#2dd4bf" stroke-width="1.4"/><circle cx="5" cy="12" r="1.6" stroke="#2dd4bf" stroke-width="1.4"/><circle cx="19" cy="12" r="1.6" stroke="#2dd4bf" stroke-width="1.4"/><circle cx="9" cy="19" r="1.6" stroke="#2dd4bf" stroke-width="1.4"/><circle cx="16" cy="19" r="1.6" stroke="#2dd4bf" stroke-width="1.4"/><path d="M12 6.5L5.4 11M12 6.5L18.6 11M5.6 13.2L9 17.7M18.4 13.2L16 17.7M9.7 19L15.3 19" stroke="#2dd4bf" stroke-width="1.1" opacity="0.7"/></svg>
  </div>
  <div>
    <h1>Abuse-Ring Sentinel</h1>
    <p>Graph-based fraud ring detection — flagged clusters from one training/evaluation run</p>
  </div>
</div>
""", unsafe_allow_html=True)

## Summary metrics — two rows of 4, so nothing gets truncated
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Total accounts", f"{summary['total_accounts']:,}")
r1c2.metric("Ring members", f"{summary['total_ring_members']:,}", f"{summary['fraud_rate']:.1%} of accounts")
r1c3.metric("Flagged clusters", summary["flagged_count"], f"{summary['total_communities']} scored")
r1c4.metric("Detector cost", fmt_rs(summary["total_detector_cost_rs"]))

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.metric("Precision", f"{summary['mean_precision']:.0%}", "5-fold CV")
r2c2.metric("Recall", f"{summary['mean_recall']:.0%}", "5-fold CV")
r2c3.metric("Fraud prevented", fmt_rs(summary["fraud_prevented_rs"]))
r2c4.metric("Net value", fmt_rs(summary["net_value_rs"]), "fraud prevented − cost")

st.divider()

# Cluster selector
cluster_labels = [
    f"#{i+1} — {c['size']} accounts, {c['confidence']:.0%} confidence, {c['outcome']}"
    for i, c in enumerate(clusters)
]

selected_idx = st.selectbox("Select a flagged cluster", range(len(clusters)), format_func=lambda i: cluster_labels[i])
cluster = clusters[selected_idx]

badge_class, badge_text = BADGE_LABELS[cluster["outcome"]]
pattern_color = ring_color(cluster["true_label"])

left, right = st.columns([2, 1])

with left:
    with st.container(border=True):
        st.markdown(
            f"### <span class='swatch' style='background:{pattern_color}'></span>"
            f"Cluster of {cluster['size']} accounts "
            f"<span class='badge {badge_class}'>{badge_text}</span>",
            unsafe_allow_html=True,
        )
        st.code(cluster["explanation"], language=None)

        # Rebuild just this cluster's mini-graph from the exported edge list
        members = set(cluster["members"])
        G_sub = nx.Graph()
        G_sub.add_nodes_from(members)
        for e in graph_data["edges"]:
            if e["source"] in members and e["target"] in members:
                G_sub.add_edge(e["source"], e["target"], weight=e["weight"])

        node_ring_lookup = {n["id"]: n["ring_id"] for n in graph_data["nodes"]}

        # Add the actual resource (device/VPA/address) that's driving this
        # cluster as its own node — same fact explain_community() already
        # printed in text, now shown directly in the graph instead of only
        # implied by edge color. Included in the SAME layout computation (not
        # positioned after the fact) so the physics simulation pulls it
        # naturally toward the accounts that actually share it.
        resource_node = cluster["primary_resource_value"]
        resource_members = [m for m in cluster["primary_resource_members"] if m in members]
        G_layout = G_sub.copy()
        G_layout.add_node(resource_node)
        for m in resource_members:
            G_layout.add_edge(resource_node, m, weight=3)

        # Same 3D force-directed layout as the notebook's Plotly graph cell —
        # edges act like springs, unrelated nodes push apart.
        pos_3d = nx.spring_layout(G_layout, weight="weight", seed=42, dim=3, k=0.6)

        # Account-to-account edges (gray) and resource-to-account edges (teal)
        # get their own traces so they read as two different kinds of connection.
        def edge_coords(edges):
            ex, ey, ez = [], [], []
            for a, b in edges:
                x0, y0, z0 = pos_3d[a]
                x1, y1, z1 = pos_3d[b]
                ex += [x0, x1, None]; ey += [y0, y1, None]; ez += [z0, z1, None]
            return ex, ey, ez

        acc_ex, acc_ey, acc_ez = edge_coords(G_sub.edges())
        edge_trace = go.Scatter3d(
            x=acc_ex, y=acc_ey, z=acc_ez,
            mode="lines",
            line=dict(color="#4c5563", width=2),
            hoverinfo="none",
        )

        res_ex, res_ey, res_ez = edge_coords([(resource_node, m) for m in resource_members])
        resource_edge_trace = go.Scatter3d(
            x=res_ex, y=res_ey, z=res_ez,
            mode="lines",
            line=dict(color="rgba(45,212,191,0.45)", width=2),
            hoverinfo="none",
        )

        node_ids = list(G_sub.nodes)
        node_colors = [ring_color(node_ring_lookup.get(n)) for n in node_ids]
        node_x = [pos_3d[n][0] for n in node_ids]
        node_y = [pos_3d[n][1] for n in node_ids]
        node_z = [pos_3d[n][2] for n in node_ids]
        # Short, real account labels (not placeholder names) — strip the
        # "acc_" prefix so "acc_ring16_3" reads as "ring16_3" without
        # inventing a fake short name for it.
        node_labels = [n.removeprefix("acc_") for n in node_ids]
        node_hover = [f"{n}<br>ring: {node_ring_lookup.get(n)}" for n in node_ids]

        # "Glow": Plotly has no native glow filter — this is the standard trick
        # (the same one the mockup's SVG used via radialGradient): a larger,
        # soft, low-opacity marker layered directly behind the solid one.
        glow_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers",
            marker=dict(size=22, color=node_colors, opacity=0.18),
            hoverinfo="skip",
        )

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers+text",
            marker=dict(size=6, color=node_colors, line=dict(color="#0A0D13", width=1)),
            text=node_labels,
            textposition="top center",
            textfont=dict(color=node_colors, size=10, family="JetBrains Mono, monospace"),
            hovertext=node_hover,
            hoverinfo="text",
        )

        # The resource node itself — teal, larger glow, its own label — same
        # visual language as the mockup's "shared device" node.
        rx, ry, rz = pos_3d[resource_node]
        resource_glow_trace = go.Scatter3d(
            x=[rx], y=[ry], z=[rz], mode="markers",
            marker=dict(size=28, color="#2dd4bf", opacity=0.22), hoverinfo="skip",
        )
        resource_trace = go.Scatter3d(
            x=[rx], y=[ry], z=[rz],
            mode="markers+text",
            marker=dict(size=9, color="#2dd4bf", line=dict(color="#0A0D13", width=1.5)),
            text=[resource_node],
            textposition="top center",
            textfont=dict(color="#5eead4", size=11, family="JetBrains Mono, monospace"),
            hovertext=[f"{resource_node}<br>{cluster['primary_resource_type']} — shared by {len(resource_members)} accounts"],
            hoverinfo="text",
        )

        fig3d = go.Figure(data=[
            edge_trace, resource_edge_trace, glow_trace, node_trace,
            resource_glow_trace, resource_trace,
        ])
        fig3d.update_layout(
            paper_bgcolor="#0A0D13",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                bgcolor="#0A0D13",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False,
            height=420,
        )
        st.plotly_chart(fig3d, width="stretch")

with right:
    with st.container(border=True):
        st.metric("Confidence", f"{cluster['confidence']:.0%}")
        st.metric("Density", f"{cluster['density']:.0%}")
        st.metric("Avg edge weight", f"{cluster['avg_weight']:.1f}")
        st.metric("Timing spread", f"{cluster['timestamp_std_hours']/24:.0f} days")
        st.metric("Ground truth", cluster["true_label"] or "no dominant label")
        st.metric("Actually a ring?", "Yes" if cluster["actually_ring"] else "No")

st.divider()
st.subheader("All flagged clusters")

df_flagged = pd.DataFrame(clusters)[["size", "confidence", "density", "avg_weight", "true_label", "outcome"]].copy()
# ProgressColumn's format string applies to the raw value directly — it does NOT
# multiply a 0-1 fraction into a percentage for you. Scale to 0-100 ourselves.
df_flagged["confidence"] = df_flagged["confidence"] * 100
df_flagged["density"] = df_flagged["density"] * 100

st.dataframe(
    df_flagged,
    width="stretch",
    column_config={
        "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=100, format="%.0f%%"),
        "density": st.column_config.ProgressColumn("density", min_value=0, max_value=100, format="%.0f%%"),
    },
    hide_index=True,
)

st.caption(
    "This is a fixed snapshot from one training/evaluation run of the ringFraudDetect "
    "notebook — it does not run live scoring. See the README for methodology and honest limitations."
)
