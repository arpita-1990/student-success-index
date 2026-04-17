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


def aggregate_state_metrics(records: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = []
    for state, grp in records.groupby("state", sort=True):
        out.append(
            {
                "state": state,
                "map_state": GEO_FIX.get(state, state),
                f"{prefix}institution_count": int(grp["institution"].nunique()),
                f"{prefix}program_rows_used": int(len(grp)),
                f"{prefix}graduates_represented": int(grp["graduates"].sum()),
                f"{prefix}non_graduation_pct": weighted_average(grp, "non_grad_pct"),
                f"{prefix}placement_pct": weighted_average(grp, "placement_pct"),
                f"{prefix}higher_education_pct": weighted_average(grp, "higher_ed_pct"),
                f"{prefix}universities": "; ".join(sorted(grp["institution"].unique())),
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


def build_choropleth(ug3_metrics_df: pd.DataFrame, ug4_metrics_df: pd.DataFrame) -> str:
    geojson = requests.get(GEOJSON_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json()
    all_map_states = sorted(feature["properties"]["NAME_1"] for feature in geojson["features"])

    ug3_rows = json.dumps(ug3_metrics_df.where(pd.notna(ug3_metrics_df), None).to_dict(orient="records"))
    ug4_rows = json.dumps(ug4_metrics_df.where(pd.notna(ug4_metrics_df), None).to_dict(orient="records"))
    states_json = json.dumps(all_map_states)
    geojson_url_json = json.dumps(GEOJSON_URL)

    html = """
    <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>

    <div class=\"metric-row\">
      <div class=\"metric-header\">
        <div class=\"metric-pill metric-pill-blue\">Placement comparison</div>
        <h3>Placement percentage</h3>
        <p>Darker blue indicates better placement outcomes. Gray states and UTs indicate no available data.</p>
      </div>
      <div class=\"dual-map-grid\">
        <div class=\"map-card\"><div class=\"map-card-title\">UG 3-Year</div><div id=\"map-placement-ug3\" class=\"mini-map\"></div></div>
        <div class=\"map-card\"><div class=\"map-card-title\">UG 4-Year</div><div id=\"map-placement-ug4\" class=\"mini-map\"></div></div>
      </div>
    </div>

    <div class=\"metric-row\">
      <div class=\"metric-header\">
        <div class=\"metric-pill metric-pill-amber\">Risk comparison</div>
        <h3>Non-graduation percentage</h3>
        <p>This is the risk view. Gray states and UTs indicate no available data.</p>
      </div>
      <div class=\"dual-map-grid\">
        <div class=\"map-card\"><div class=\"map-card-title\">UG 3-Year</div><div id=\"map-nongrad-ug3\" class=\"mini-map\"></div></div>
        <div class=\"map-card\"><div class=\"map-card-title\">UG 4-Year</div><div id=\"map-nongrad-ug4\" class=\"mini-map\"></div></div>
      </div>
    </div>

    <div class=\"metric-row\">
      <div class=\"metric-header\">
        <div class=\"metric-pill metric-pill-purple\">Progression comparison</div>
        <h3>Higher education percentage</h3>
        <p>This shows the share of graduates moving into further studies. Gray states and UTs indicate no available data.</p>
      </div>
      <div class=\"dual-map-grid\">
        <div class=\"map-card\"><div class=\"map-card-title\">UG 3-Year</div><div id=\"map-he-ug3\" class=\"mini-map\"></div></div>
        <div class=\"map-card\"><div class=\"map-card-title\">UG 4-Year</div><div id=\"map-he-ug4\" class=\"mini-map\"></div></div>
      </div>
    </div>

    <script>
      const GEOJSON_URL = __GEOJSON_URL__;
      const ALL_MAP_STATES = __ALL_STATES__;
      const PROGRAMS = {
        ug3: { label: 'UG 3-Year', rows: __UG3_ROWS__ },
        ug4: { label: 'UG 4-Year', rows: __UG4_ROWS__ }
      };

      const METRIC_SPECS = [
        { key: 'placement_pct', label: 'Placement %', colorscale: 'Blues', reversescale: false, ids: { ug3: 'map-placement-ug3', ug4: 'map-placement-ug4' } },
        { key: 'non_graduation_pct', label: 'Non-graduation %', colorscale: 'YlOrRd', reversescale: true, ids: { ug3: 'map-nongrad-ug3', ug4: 'map-nongrad-ug4' } },
        { key: 'higher_education_pct', label: 'Higher education %', colorscale: 'Purples', reversescale: false, ids: { ug3: 'map-he-ug3', ug4: 'map-he-ug4' } }
      ];

      function baseLayout() {
        return {
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
          margin: { l: 8, r: 34, t: 4, b: 8 },
          height: 420,
          geo: {
            visible: false,
            bgcolor: 'rgba(0,0,0,0)',
            projection: { type: 'mercator' },
            center: { lat: 22.8, lon: 76.8 },
            lataxis: { range: [5, 38.5] },
            lonaxis: { range: [66.5, 100] }
          }
        };
      }

      function buildMapTraces(rows, spec, zmin, zmax, geojson) {
        const noDataTrace = {
          type: 'choropleth',
          geojson,
          featureidkey: 'properties.NAME_1',
          locations: ALL_MAP_STATES,
          z: ALL_MAP_STATES.map(() => 0),
          locationmode: 'geojson-id',
          colorscale: [[0, '#E5E7EB'], [1, '#E5E7EB']],
          showscale: false,
          marker: { line: { color: '#FFFFFF', width: 0.8 } },
          hovertemplate: '<b>%{location}</b><br>No mapped data available<extra></extra>'
        };

        const metricRows = rows.filter(row => row[spec.key] !== null && row[spec.key] !== undefined);
        const metricTrace = {
          type: 'choropleth',
          geojson,
          featureidkey: 'properties.NAME_1',
          locations: metricRows.map(row => row.map_state),
          z: metricRows.map(row => row[spec.key]),
          locationmode: 'geojson-id',
          colorscale: spec.colorscale,
          reversescale: spec.reversescale,
          zmin,
          zmax,
          marker: { line: { color: '#FFFFFF', width: 0.8 } },
          colorbar: { title: spec.label, thickness: 10, len: 0.62, x: 1.03, y: 0.5 },
          customdata: metricRows.map(row => [row.state, row.institution_count || row.ug3_institution_count || row.ug4_institution_count, row.graduates_represented || row.ug3_graduates_represented || row.ug4_graduates_represented]),
          hovertemplate: '<b>%{customdata[0]}</b><br>' + spec.label + ': <b>%{z:.1f}%</b><br>Institutions covered: %{customdata[1]}<br>Graduates represented: %{customdata[2]:,}<extra></extra>'
        };

        return [noDataTrace, metricTrace];
      }

      function renderAllMaps(geojson) {
        METRIC_SPECS.forEach(spec => {
          const values = [];
          Object.values(PROGRAMS).forEach(program => {
            program.rows.forEach(row => {
              if (row[spec.key] !== null && row[spec.key] !== undefined) values.push(row[spec.key]);
            });
          });
          const zmin = values.length ? Math.min(...values) : 0;
          const zmax = values.length ? Math.max(...values) : 100;

          Object.entries(PROGRAMS).forEach(([programKey, program]) => {
            Plotly.newPlot(
              spec.ids[programKey],
              buildMapTraces(program.rows, spec, zmin, zmax, geojson),
              baseLayout(),
              { responsive: true, displayModeBar: false }
            );
          });
        });
      }

      fetch(GEOJSON_URL)
        .then(resp => resp.json())
        .then(geojson => renderAllMaps(geojson))
        .catch(err => {
          document.querySelectorAll('.mini-map').forEach(el => {
            el.innerHTML = '<div style="padding:24px;color:#B91C1C">Map failed to load. Please refresh the page.</div>';
          });
          console.error(err);
        });
    </script>
    """

    return (
        html.replace("__GEOJSON_URL__", geojson_url_json)
        .replace("__ALL_STATES__", states_json)
        .replace("__UG3_ROWS__", ug3_rows)
        .replace("__UG4_ROWS__", ug4_rows)
    )


def build_html(metrics_df: pd.DataFrame, ug3_metrics_df: pd.DataFrame, ug4_metrics_df: pd.DataFrame, chart_html: str) -> str:
    coverage_states_total = int(len(metrics_df))
    coverage_states_ug3 = int(len(ug3_metrics_df))
    coverage_states_ug4 = int(len(ug4_metrics_df))
    coverage_insts = len(STATE_BY_INST_RAW)
    no_data_count = 36 - coverage_states_total

    table_df = metrics_df[[
        "state",
        "ug3_non_graduation_pct",
        "ug4_non_graduation_pct",
        "ug3_placement_pct",
        "ug4_placement_pct",
        "ug3_higher_education_pct",
        "ug4_higher_education_pct",
    ]].copy()
    table_df.columns = [
        "State / UT",
        "UG3 Non-grad %",
        "UG4 Non-grad %",
        "UG3 Placement %",
        "UG4 Placement %",
        "UG3 Higher Ed %",
        "UG4 Higher Ed %",
    ]
    for col in table_df.columns[1:]:
        table_df[col] = table_df[col].map(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
    table_html = table_df.to_html(index=False, classes="metric-table", border=0)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
<title>State Public University Heat Map — India</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif; background: #f4f7fb; color: #18212f; }}
  .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
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
  .metric-row {{ margin-top: 18px; padding-top: 8px; border-top: 1px solid #EEF2F7; }}
  .metric-row:first-of-type {{ margin-top: 8px; padding-top: 0; border-top: 0; }}
  .metric-header h3 {{ margin: 4px 0 4px; color: #17324F; font-size: 1rem; }}
  .metric-header p {{ margin: 0 0 12px; color: #6B7280; font-size: 0.84rem; }}
  .metric-pill {{ display:inline-flex; align-items:center; padding:5px 10px; border-radius:999px; font-size:0.74rem; font-weight:700; letter-spacing:.02em; }}
  .metric-pill-blue {{ background:#DBEAFE; color:#1D4ED8; }}
  .metric-pill-amber {{ background:#FEF3C7; color:#B45309; }}
  .metric-pill-purple {{ background:#EDE9FE; color:#6D28D9; }}
  .dual-map-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; }}
  .map-card {{ background:linear-gradient(180deg, #FBFDFF, #F8FAFC); border:1px solid #E5E7EB; border-radius:14px; padding:12px; box-shadow: 0 6px 18px rgba(20,35,60,.05); }}
  .map-card-title {{ color:#1B2A4A; font-size:0.95rem; font-weight:800; margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #EDF2F7; }}
  .mini-map {{ height:420px; }}
  .metric-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 8px; }}
  .metric-table th {{ background: #1B2A4A; color: #F5A623; padding: 10px; text-align: left; }}
  .metric-table td {{ border: 1px solid #E5E7EB; padding: 9px 10px; }}
  .metric-table tr:nth-child(even) {{ background: #F9FAFB; }}
  .note {{ font-size: 0.82rem; color: #6B7280; margin-top: 10px; }}
  @media (max-width: 1000px) {{ .dual-map-grid, .grid {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 640px) {{ .wrap {{ padding: 14px; }} }}
</style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1>India heat map for State Public Universities</h1>
      <p>This page compares <strong>UG 3-Year</strong> and <strong>UG 4-Year</strong> state outcomes side by side for placement, non-graduation, and higher education.</p>
      <p><strong>Method used:</strong> latest reported cycle per university, aggregated state-wise using graduate-count weighting.</p>
      <p><strong>State mapping research basis:</strong> matched from official university names, city identifiers in the NIRF list, and public institutional references.</p>
    </div>

    <div class=\"grid\">
      <div class=\"card\"><div class=\"k\">States / UTs with UG3 data</div><div class=\"v\">{coverage_states_ug3}</div></div>
      <div class=\"card\"><div class=\"k\">States / UTs with UG4 data</div><div class=\"v\">{coverage_states_ug4}</div></div>
      <div class=\"card\"><div class=\"k\">Institutions mapped</div><div class=\"v\">{coverage_insts}</div></div>
      <div class=\"card\"><div class=\"k\">States / UTs without usable data</div><div class=\"v\">{no_data_count}</div></div>
    </div>

    <div class=\"section\">
      <h2>Side-by-side UG 3 and UG 4 maps</h2>
      <p>Each row compares the same metric across UG 3-Year and UG 4-Year programmes for the full India map.</p>
      {chart_html}
      <div class=\"note\">Gray states and UTs indicate no available mapped university outcome data for that programme.</div>
    </div>

    <div class=\"section\">
      <h2>State-wise summary table</h2>
      <p>The table below shows the state-level percentages used in the side-by-side maps.</p>
      {table_html}
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    ug3 = extract_latest_records(UG3_CSV, "UG 3-Year")
    ug4 = extract_latest_records(UG4_CSV, "UG 4-Year")

    ug3_metrics = aggregate_state_metrics(ug3)
    ug4_metrics = aggregate_state_metrics(ug4)

    ug3_export = ug3_metrics.rename(
        columns={c: f"ug3_{c}" for c in ug3_metrics.columns if c not in {"state", "map_state"}}
    )
    ug4_export = ug4_metrics.rename(
        columns={c: f"ug4_{c}" for c in ug4_metrics.columns if c not in {"state", "map_state"}}
    )
    metrics_df = pd.merge(ug3_export, ug4_export, on=["state", "map_state"], how="outer").sort_values("state").reset_index(drop=True)

    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    lookup_table().to_csv(OUT_LOOKUP_CSV, index=False)

    chart_html = build_choropleth(ug3_metrics, ug4_metrics)
    OUT_HTML.write_text(build_html(metrics_df, ug3_metrics, ug4_metrics, chart_html), encoding="utf-8")

    print(f"Created: {OUT_HTML}")
    print(f"Created: {OUT_METRICS_CSV}")
    print(f"Created: {OUT_LOOKUP_CSV}")
    print(f"UG3 states covered: {len(ug3_metrics)}")
    print(f"UG4 states covered: {len(ug4_metrics)}")
    print(metrics_df[["state", "ug3_placement_pct", "ug4_placement_pct"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
