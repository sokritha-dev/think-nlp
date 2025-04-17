import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python render_report.py <csv_file>")
    sys.exit(1)

csv_path = Path(sys.argv[1])
if not csv_path.exists():
    print(f"❌ File not found: {csv_path}")
    sys.exit(1)

# Load Locust CSV
df = pd.read_csv(csv_path)

# Clean and prepare data
df = df[df["Name"].notna() & (df["Name"] != "Aggregated")]
df["Label"] = df["Type"] + " " + df["Name"]
df["Failure %"] = (df["Failure Count"] / df["Request Count"]) * 100

# Create a Plotly chart layout
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=[
        "Requests per Second",
        "Median Response Time (ms)",
        "Failure Rate (%)",
    ],
    vertical_spacing=0.2,
)

# Chart 1: RPS
fig.add_trace(
    go.Bar(
        x=df["Label"],
        y=df["Requests/s"],
        name="Requests/s",
        marker_color="steelblue",
        text=df["Requests/s"],
        textposition="outside",
    ),
    row=1,
    col=1,
)

# Chart 2: Median Response Time
fig.add_trace(
    go.Bar(
        x=df["Label"],
        y=df["Median Response Time"],
        name="Median Response Time",
        marker_color="darkorange",
        text=df["Median Response Time"],
        textposition="outside",
    ),
    row=2,
    col=1,
)

# Chart 3: Failure %
fig.add_trace(
    go.Bar(
        x=df["Label"],
        y=df["Failure %"],
        name="Failure %",
        marker_color="crimson",
        text=[f"{val:.2f}%" for val in df["Failure %"]],
        textposition="outside",
    ),
    row=3,
    col=1,
)

# Final layout tweaks
fig.update_layout(
    height=900,
    title_text="📊 Upload API Load Test - Visual Summary",
    showlegend=False,
    margin=dict(t=60, b=40),
)

# Save to HTML
output_file = Path("reports/upload_test_charts.html")
fig.write_html(str(output_file), include_plotlyjs="cdn")
print(f"✅ Chart-based report saved to {output_file}")
