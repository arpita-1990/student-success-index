from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "NIRF_State University"
UG3_CSV = DATA_DIR / "State_Public_UG3yr_All_50.csv"
UG4_CSV = DATA_DIR / "State_Public_UG4yr_All_50.csv"
OUT_HTML = ROOT / "state_public_university_india_heatmap.html"
OUT_METRICS_CSV = DATA_DIR / "state_public_university_state_metrics.csv"
OUT_LOOKUP_CSV = DATA_DIR / "state_public_university_state_lookup.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"


def norm(text: object) -> str:
    return (
        str(text)
        .strip()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("`", "'")
        .replace("&", "and")
        .replace("  ", " ")
        .lower()
    )


STATE_BY_INST_RAW = {
    "Jadavpur University": "West Bengal",
    "Anna University": "Tamil Nadu",
    "Panjab University": "Chandigarh",
    "Andhra University, Visakhapatnam": "Andhra Pradesh",
    "Kerala University": "Kerala",
    "Cochin University of Science and Technology": "Kerala",
    "Osmania University": "Telangana",
    "University of Kashmir": "Jammu and Kashmir",
    "Gauhati University": "Assam",
    "Bharathiar University": "Tamil Nadu",
    "Savitribai Phule Pune University": "Maharashtra",
    "Mumbai University": "Maharashtra",
    "Delhi Technological University": "Delhi",
    "Alagappa University": "Tamil Nadu",
    "Calcutta University": "West Bengal",
    "Bharathidasan University": "Tamil Nadu",
    "Mahatma Gandhi University, Kottayam": "Kerala",
    "University of Madras": "Tamil Nadu",
    "IIIT-Delhi (Indraprastha Institute of Information Technology Delhi)": "Delhi",
    "Mysore University": "Karnataka",
    "University of Jammu": "Jammu and Kashmir",
    "Guru Gobind Singh Indraprastha University": "Delhi",
    "Madan Mohan Malaviya University of Technology": "Uttar Pradesh",
    "Acharya Nagarjuna University": "Andhra Pradesh",
    "Gujarat University": "Gujarat",
    "Bangalore University": "Karnataka",
    "University of Lucknow": "Uttar Pradesh",
    "Punjab Agricultural University, Ludhiana": "Punjab",
    "King George`s Medical University": "Uttar Pradesh",
    "Dibrugarh University": "Assam",
    "Madurai Kamaraj University": "Tamil Nadu",
    "Guru Jambheshwar University of Science and Technology, Hissar": "Haryana",
    "Manonmaniam Sundaranar University, Tirunelveli": "Tamil Nadu",
    "Annamalai University": "Tamil Nadu",
    "Kurukshetra University": "Haryana",
    "Sher-e-Kashmir University of Agricultural Science & Technology of Kashmir, Srinagar": "Jammu and Kashmir",
    "University of Agricultural Sciences, Bangalore": "Karnataka",
    "Calicut University, Thenhipalem, Malapuram": "Kerala",
    "Netaji Subhas University of Technology (NSUT)": "Delhi",
    "Periyar University": "Tamil Nadu",
    "CHAUDHARY CHARAN SINGH UNIVERSITY MEERUT": "Uttar Pradesh",
    "Tamil Nadu Agricultural University": "Tamil Nadu",
    "COEP Technological University": "Maharashtra",
    "G.B. Pant Universtiy of Agriculture and Technology, Pantnagar": "Uttarakhand",
    "Shivaji University": "Maharashtra",
    "Maharshi Dayanand University": "Haryana",
    "Sri Venkateswara University": "Andhra Pradesh",
    "Utkal University": "Odisha",
    "Devi Ahilya Vishwavidyalaya": "Madhya Pradesh",
    "Visvesvaraya Technological University": "Karnataka",
}

STATE_BY_INST = {norm(k): v for k, v in STATE_BY_INST_RAW.items()}
GEO_FIX = {"Odisha": "Orissa", "Uttarakhand": "Uttaranchal"}


def to_num(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    s = str(value).strip().replace(",", "")
    if s in {"", "-", "nan", "None"}:
        return float("nan")
    return pd.to_numeric(s, errors="coerce")


def get_state(name: str) -> str | None:
    key = norm(name)
    if key in STATE_BY_INST:
        return STATE_BY_INST[key]
    key = key.replace(" of science and technology ", " of science and technology, ")
    return STATE_BY_INST.get(key)


def extract_latest_records(csv_path: Path, program_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=4, header=None)
    bases = [2, 16, 30]
    rows: list[dict] = []

    for _, row in df.iterrows():
        inst = str(row[0]).strip()
        if not inst or inst == "nan":
            continue
        state = get_state(inst)
        if not state:
            continue

        picked = None
        for base in reversed(bases):
            year = row[base]
            grads = to_num(row[base + 5])
            grad_rate = to_num(row[base + 6])
            non_grad_rate = to_num(row[base + 7])
            placed_pct = to_num(row[base + 9])
            higher_ed_pct = to_num(row[base + 11])
            if pd.notna(grads) and grads > 0 and (pd.notna(grad_rate) or pd.notna(non_grad_rate)):
                if pd.isna(non_grad_rate) and pd.notna(grad_rate):
                    non_grad_rate = 100 - grad_rate
                picked = {
                    "institution": inst,
                    "state": state,
                    "program": program_label,
                    "year": str(year),
                    "graduates": float(grads),
                    "grad_rate": float(grad_rate) if pd.notna(grad_rate) else None,
                    "non_grad_pct": float(non_grad_rate) if pd.notna(non_grad_rate) else None,
                    "placement_pct": float(placed_pct) if pd.notna(placed_pct) else None,
                    "higher_ed_pct": float(higher_ed_pct) if pd.notna(higher_ed_pct) else None,
                }
                break

        if picked:
            rows.append(picked)

    return pd.DataFrame(rows)


def weighted_average(group: pd.DataFrame, metric: str) -> float | None:
    valid = group.dropna(subset=[metric, "graduates"])
    if valid.empty:
        return None
    total_w = valid["graduates"].sum()
    if not total_w:
        return None
    return round((valid[metric] * valid["graduates"]).sum() / total_w, 2)


def aggregate_state_metrics(records: pd.DataFrame) -> pd.DataFrame:
    out = []
    for state, grp in records.groupby("state", sort=True):
        out.append(
            {
                "state": state,
                "map_state": GEO_FIX.get(state, state),
                "institution_count": int(grp["institution"].nunique()),
                "program_rows_used": int(len(grp)),
                "graduates_represented": int(grp["graduates"].sum()),
                "non_graduation_pct": weighted_average(grp, "non_grad_pct"),
                "placement_pct": weighted_average(grp, "placement_pct"),
                "higher_education_pct": weighted_average(grp, "higher_ed_pct"),
                "universities": "; ".join(sorted(grp["institution"].unique())),
            }
        )
    return pd.DataFrame(out).sort_values("state").reset_index(drop=True)


def lookup_table() -> pd.DataFrame:
    rows = []
    for inst, state in sorted(STATE_BY_INST_RAW.items(), key=lambda x: (x[1], x[0])):
        rows.append(
            {
                "institution": inst,
                "state": state,
                "research_basis": "Matched from official university name and city/state reference",
            }
        )
    return pd.DataFrame(rows)


def build_choropleth(metrics_df: pd.DataFrame) -> str:
    geojson = requests.get(GEOJSON_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json()
    all_map_states = sorted(feature["properties"]["NAME_1"] for feature in geojson["features"])

    metric_specs = [
        ("non_graduation_pct", "Non-graduation %", "YlOrRd", True),
        ("placement_pct", "Placement %", "Blues", False),
        ("higher_education_pct", "Higher education %", "Purples", False),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            geojson=geojson,
            featureidkey="properties.NAME_1",
            locations=all_map_states,
            z=[0] * len(all_map_states),
            locationmode="geojson-id",
            colorscale=[[0, "#E5E7EB"], [1, "#E5E7EB"]],
            showscale=False,
            marker_line_color="#FFFFFF",
            marker_line_width=0.8,
            hovertemplate="<b>%{location}</b><br>No mapped state public university outcome data available<extra></extra>",
            visible=True,
        )
    )

    for i, (col, label, scale, rev) in enumerate(metric_specs):
        custom = metrics_df[["state", "institution_count", "graduates_represented", "universities"]].values
        hover = (
            "<b>%{customdata[0]}</b><br>"
            + label
            + ": <b>%{z:.1f}%</b><br>"
            + "Institutions covered: %{customdata[1]}<br>"
            + "Graduates represented: %{customdata[2]:,}<br>"
            + "<extra></extra>"
        )
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                featureidkey="properties.NAME_1",
                locations=metrics_df["map_state"],
                z=metrics_df[col],
                locationmode="geojson-id",
                colorscale=scale,
                reversescale=rev,
                marker_line_color="#FFFFFF",
                marker_line_width=0.8,
                colorbar_title=label,
                customdata=custom,
                hovertemplate=hover,
                visible=(i == 0),
            )
        )

    fig.update_geos(
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
        center={"lat": 22.8, "lon": 79.0},
        lataxis_range=[5, 38.5],
        lonaxis_range=[67, 98],
    )

    visibility_maps = []
    for i in range(len(metric_specs)):
        visible = [True] + [False] * len(metric_specs)
        visible[i + 1] = True
        visibility_maps.append(visible)

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=10, r=10, t=70, b=10),
        height=760,
        title=dict(
            text="India heat map — state public university outcomes",
            x=0.02,
            font=dict(size=22, color="#1B2A4A"),
        ),
        annotations=[
            dict(
                text="Gray states/UTs = no available mapped university outcome data",
                x=0.02,
                y=1.02,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="#6B7280"),
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.02,
                y=1.10,
                showactive=True,
                bgcolor="#F3F4F6",
                bordercolor="#D1D5DB",
                buttons=[
                    dict(label=label, method="update", args=[{"visible": visibility_maps[i]}])
                    for i, (_, label, _, _) in enumerate(metric_specs)
                ],
            )
        ],
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": True})


def build_html(metrics_df: pd.DataFrame, chart_html: str) -> str:
    overall_non_grad = metrics_df["non_graduation_pct"].dropna().mean()
    overall_place = metrics_df["placement_pct"].dropna().mean()
    overall_he = metrics_df["higher_education_pct"].dropna().mean()
    coverage_states = int(len(metrics_df))
    coverage_insts = int(metrics_df["institution_count"].sum())

    table_df = metrics_df[[
        "state",
        "institution_count",
        "graduates_represented",
        "non_graduation_pct",
        "placement_pct",
        "higher_education_pct",
    ]].copy()
    table_df.columns = [
        "State / UT",
        "Institutions",
        "Graduates represented",
        "Non-graduation %",
        "Placement %",
        "Higher education %",
    ]
    for col in ["Non-graduation %", "Placement %", "Higher education %"]:
        table_df[col] = table_df[col].map(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
    table_html = table_df.to_html(index=False, classes="metric-table", border=0)

    no_data_count = 36 - coverage_states

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
<title>State Public University Heat Map — India</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background: #f4f7fb; color: #18212f; }}
  .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  .hero {{ background: linear-gradient(135deg, #1B2A4A, #23486A); color: white; border-radius: 16px; padding: 22px 24px; box-shadow: 0 10px 24px rgba(0,0,0,.12); }}
  .hero h1 {{ margin: 0 0 8px; color: #F5A623; font-size: 1.7rem; }}
  .hero p {{ margin: 6px 0; font-size: 0.95rem; line-height: 1.55; color: #E5EEF8; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin: 18px 0 22px; }}
  .card {{ background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(20,35,60,.08); }}
  .card .k {{ color: #6b7280; font-size: 0.8rem; margin-bottom: 5px; }}
  .card .v {{ color: #1B2A4A; font-size: 1.4rem; font-weight: 700; }}
  .section {{ background: white; border-radius: 14px; padding: 16px; box-shadow: 0 2px 12px rgba(20,35,60,.08); margin-top: 18px; }}
  .section h2 {{ margin: 0 0 6px; color: #1B2A4A; font-size: 1.1rem; }}
  .section p {{ margin: 0 0 10px; color: #5b6472; font-size: 0.9rem; line-height: 1.55; }}
  .metric-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 8px; }}
  .metric-table th {{ background: #1B2A4A; color: #F5A623; padding: 10px; text-align: left; }}
  .metric-table td {{ border: 1px solid #E5E7EB; padding: 9px 10px; }}
  .metric-table tr:nth-child(even) {{ background: #F9FAFB; }}
  .note {{ font-size: 0.82rem; color: #6B7280; margin-top: 10px; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} .wrap {{ padding: 14px; }} }}
</style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1>India heat map for State Public Universities</h1>
      <p>This view aggregates NIRF state public university outcome data by state and union territory.</p>
      <p><strong>Method used:</strong> latest reported UG 3-year and UG 4-year outcome cycle per university, combined with graduate-count weighting.</p>
      <p><strong>State mapping research basis:</strong> matched from official university names, city identifiers in the NIRF list, and public institutional references.</p>
    </div>

    <div class=\"grid\">
      <div class=\"card\"><div class=\"k\">States / UTs with data</div><div class=\"v\">{coverage_states}</div></div>
      <div class=\"card\"><div class=\"k\">States / UTs without data</div><div class=\"v\">{no_data_count}</div></div>
      <div class=\"card\"><div class=\"k\">Institutions mapped</div><div class=\"v\">{coverage_insts}</div></div>
      <div class=\"card\"><div class=\"k\">Avg placement %</div><div class=\"v\">{overall_place:.1f}</div></div>
    </div>

    <div class=\"section\">
      <h2>Interactive heat map</h2>
      <p>Use the buttons above the map to switch between non-graduation, placement, and higher education outcomes.</p>
      {chart_html}
      <div class=\"note\">Non-graduation is a risk indicator, while placement and higher education show positive transitions after graduation.</div>
    </div>

    <div class=\"section\">
      <h2>State-wise summary table</h2>
      <p>The table below shows the weighted averages used in the map.</p>
      {table_html}
      <div class=\"note\">Blank values mean the relevant outcome metric was not reported in the latest available cycle for that state’s institutions.</div>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    ug3 = extract_latest_records(UG3_CSV, "UG 3-Year")
    ug4 = extract_latest_records(UG4_CSV, "UG 4-Year")
    combined = pd.concat([ug3, ug4], ignore_index=True)

    metrics_df = aggregate_state_metrics(combined)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    lookup_table().to_csv(OUT_LOOKUP_CSV, index=False)

    chart_html = build_choropleth(metrics_df)
    OUT_HTML.write_text(build_html(metrics_df, chart_html), encoding="utf-8")

    print(f"Created: {OUT_HTML}")
    print(f"Created: {OUT_METRICS_CSV}")
    print(f"Created: {OUT_LOOKUP_CSV}")
    print(f"States covered: {len(metrics_df)}")
    print(metrics_df[["state", "non_graduation_pct", "placement_pct", "higher_education_pct"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
