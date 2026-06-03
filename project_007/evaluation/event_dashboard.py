"""
PROJECT 007 — P4.5 Event Review Dashboard
Exports an HTML report showing per-event details with video context.

Usage:
    python -m evaluation.event_dashboard [--dataset-dir dataset]
"""

import json
import time
from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


def _html_header(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #00d4ff; font-size: 28px; margin-bottom: 10px; }}
h2 {{ color: #ff6b35; font-size: 22px; margin: 30px 0 15px; border-bottom: 1px solid #333; padding-bottom: 8px; }}
h3 {{ color: #aaa; font-size: 16px; margin-bottom: 8px; }}
.header {{ background: linear-gradient(135deg, #0d1117, #161b22); padding: 30px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #21262d; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
.summary-card {{ background: #161b22; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #21262d; }}
.summary-card .value {{ font-size: 36px; font-weight: bold; color: #58a6ff; }}
.summary-card .label {{ font-size: 13px; color: #8b949e; margin-top: 5px; }}
.card-tp .value {{ color: #3fb950; }}
.card-fp .value {{ color: #f85149; }}
.card-fn .value {{ color: #d29922; }}
.card-tn .value {{ color: #58a6ff; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #0d1117; border-radius: 8px; overflow: hidden; }}
th {{ background: #161b22; color: #58a6ff; padding: 12px; text-align: left; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; font-size: 13px; }}
tr:hover {{ background: #161b22; }}
.tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.tag-tp {{ background: #1b4332; color: #3fb950; }}
.tag-fp {{ background: #4a1919; color: #f85149; }}
.tag-fn {{ background: #3d2e00; color: #d29922; }}
.tag-tn {{ background: #0c2d48; color: #58a6ff; }}
.tag-rule {{ background: #1c1c3a; color: #bc8cff; margin: 1px; }}
.bar {{ height: 8px; border-radius: 4px; background: #21262d; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 4px; }}
.section {{ background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 20px; margin: 20px 0; }}
.feature-list {{ list-style: none; }}
.feature-list li {{ padding: 4px 0; font-size: 13px; }}
.feature-list li span {{ color: #58a6ff; font-weight: bold; margin-left: 5px; }}
.metrics-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }}
.metric {{ background: #161b22; padding: 8px 16px; border-radius: 6px; font-size: 13px; }}
.metric b {{ color: #58a6ff; }}
.footer {{ text-align: center; color: #484f58; padding: 30px; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
"""


def _html_footer() -> str:
    return f"""
<div class="footer">
    PROJECT 007 — P4.5 Event Review Dashboard — Generated {time.strftime("%Y-%m-%d %H:%M:%S")}
</div>
</div>
</body>
</html>"""


def generate_event_dashboard(
    error_analysis_path: str = "evaluation/reports/error_analysis.json",
    class_metrics_path: str = "evaluation/reports/class_metrics.json",
    feature_analysis_path: str = "evaluation/reports/feature_analysis.json",
    output_path: str = "evaluation/reports/event_dashboard.html",
) -> str:
    """
    Generate an HTML event review dashboard from analysis reports.
    """
    html = _html_header("PROJECT 007 — Event Review Dashboard")

    # Header
    html += """
<div class="header">
    <h1>PROJECT 007 — Event Review Dashboard</h1>
    <h3>P4.5 Scientific Validation Framework</h3>
</div>
"""

    # Load error analysis
    error_data = {}
    ea_path = Path(error_analysis_path)
    if ea_path.exists():
        with open(ea_path, "r") as f:
            error_data = json.load(f)

    # Summary cards
    counts = error_data.get("counts", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    metrics = error_data.get("metrics", {})

    html += '<h2>Summary</h2>'
    html += '<div class="summary-grid">'
    html += f'<div class="summary-card card-tp"><div class="value">{counts.get("TP", 0)}</div><div class="label">True Positives</div></div>'
    html += f'<div class="summary-card card-fp"><div class="value">{counts.get("FP", 0)}</div><div class="label">False Positives</div></div>'
    html += f'<div class="summary-card card-tn"><div class="value">{counts.get("TN", 0)}</div><div class="label">True Negatives</div></div>'
    html += f'<div class="summary-card card-fn"><div class="value">{counts.get("FN", 0)}</div><div class="label">False Negatives</div></div>'
    html += '</div>'

    # Metrics row
    html += '<div class="metrics-row">'
    for key, label in [("precision", "Precision"), ("recall", "Recall"),
                        ("f1", "F1"), ("false_positive_rate", "FPR"),
                        ("false_negative_rate", "FNR")]:
        val = metrics.get(key, 0)
        html += f'<div class="metric"><b>{label}:</b> {val:.4f}</div>'
    html += '</div>'

    # FP Rule Attribution
    fp_rules = error_data.get("false_positive_rule_attribution", {})
    if fp_rules:
        html += '<h2>False Positive Rule Attribution</h2>'
        html += '<div class="section"><table>'
        html += '<tr><th>Rule</th><th>FP Count</th><th>Bar</th></tr>'
        max_count = max(fp_rules.values()) if fp_rules else 1
        for rule, count in fp_rules.items():
            pct = int(100 * count / max_count)
            html += f'<tr><td><span class="tag tag-rule">{rule}</span></td>'
            html += f'<td>{count}</td>'
            html += f'<td><div class="bar"><div class="bar-fill" style="width:{pct}%;background:#f85149;"></div></div></td></tr>'
        html += '</table></div>'

    # FN distribution
    fn_dist = error_data.get("false_negative_gt_distribution", {})
    if fn_dist:
        html += '<h2>False Negative Ground Truth Distribution</h2>'
        html += '<div class="section"><table>'
        html += '<tr><th>Ground Truth Label</th><th>Missed Count</th></tr>'
        for label, count in sorted(fn_dist.items(), key=lambda x: -x[1]):
            html += f'<tr><td>{label}</td><td>{count}</td></tr>'
        html += '</table></div>'

    # Per-Class Metrics
    cm_path = Path(class_metrics_path)
    if cm_path.exists():
        with open(cm_path, "r") as f:
            class_data = json.load(f)

        per_class = class_data.get("per_class", {})
        if per_class:
            html += '<h2>Per-Class Metrics</h2>'
            html += '<div class="section"><table>'
            html += '<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>'
            for cls, m in per_class.items():
                html += f'<tr><td>{cls}</td>'
                html += f'<td>{m.get("precision", 0):.4f}</td>'
                html += f'<td>{m.get("recall", 0):.4f}</td>'
                html += f'<td>{m.get("f1", 0):.4f}</td>'
                html += f'<td>{m.get("support", 0)}</td></tr>'
            html += '</table></div>'

        # Confusion matrix
        cm = class_data.get("confusion_matrix", {})
        if cm:
            classes = class_data.get("classes", [])
            html += '<h2>Confusion Matrix</h2>'
            html += '<div class="section"><table>'
            html += '<tr><th>GT \\ Pred</th>'
            for c in classes:
                html += f'<th>{c[:12]}</th>'
            html += '</tr>'
            for gt_cls in classes:
                html += f'<tr><td><b>{gt_cls}</b></td>'
                for pred_cls in classes:
                    val = cm.get(gt_cls, {}).get(pred_cls, 0)
                    style = ' style="color:#3fb950;font-weight:bold;"' if gt_cls == pred_cls and val > 0 else ''
                    html += f'<td{style}>{val}</td>'
                html += '</tr>'
            html += '</table></div>'

    # Feature Analysis
    fa_path = Path(feature_analysis_path)
    if fa_path.exists():
        with open(fa_path, "r") as f:
            feat_data = json.load(f)

        html += '<h2>Feature Importance</h2>'

        for method in ["random_forest", "xgboost", "shap"]:
            method_data = feat_data.get(method, {})
            importance = method_data.get("importance", [])
            if importance:
                html += f'<div class="section"><h3>{method.replace("_", " ").title()}</h3>'
                html += '<table><tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Bar</th></tr>'
                max_imp = importance[0][1] if importance else 1
                for rank, (feat, imp) in enumerate(importance[:15], 1):
                    pct = int(100 * imp / max_imp) if max_imp > 0 else 0
                    html += f'<tr><td>{rank}</td><td>{feat}</td><td>{imp:.6f}</td>'
                    html += f'<td><div class="bar"><div class="bar-fill" style="width:{pct}%;background:#58a6ff;"></div></div></td></tr>'
                html += '</table></div>'

        # Consensus
        consensus = feat_data.get("consensus_ranking", [])
        if consensus:
            html += '<div class="section"><h3>Consensus Ranking (avg rank across methods)</h3>'
            html += '<table><tr><th>Rank</th><th>Feature</th><th>Avg Rank</th></tr>'
            for rank, (feat, avg) in enumerate(consensus[:10], 1):
                html += f'<tr><td>{rank}</td><td>{feat}</td><td>{avg:.1f}</td></tr>'
            html += '</table></div>'

    # Sample Errors
    sample_fps = error_data.get("sample_errors", {}).get("false_positives", [])
    if sample_fps:
        html += '<h2>Sample False Positives</h2>'
        html += '<div class="section"><table>'
        html += '<tr><th>Video</th><th>Frame</th><th>Time</th><th>Score</th><th>State</th><th>Rules</th></tr>'
        for e in sample_fps[:25]:
            rules_html = " ".join(f'<span class="tag tag-rule">{r}</span>' for r in e.get("contributing_rules", []))
            html += f'<tr><td>{e.get("video_id", "")}</td>'
            html += f'<td>{e.get("frame_id", "")}</td>'
            html += f'<td>{e.get("timestamp_sec", "")}s</td>'
            html += f'<td>{e.get("fused_score", 0):.4f}</td>'
            html += f'<td>{e.get("state", "")}</td>'
            html += f'<td>{rules_html}</td></tr>'
        html += '</table></div>'

    sample_fns = error_data.get("sample_errors", {}).get("false_negatives", [])
    if sample_fns:
        html += '<h2>Sample False Negatives</h2>'
        html += '<div class="section"><table>'
        html += '<tr><th>Video</th><th>Frame</th><th>Time</th><th>Ground Truth</th><th>Score</th></tr>'
        for e in sample_fns[:25]:
            html += f'<tr><td>{e.get("video_id", "")}</td>'
            html += f'<td>{e.get("frame_id", "")}</td>'
            html += f'<td>{e.get("timestamp_sec", "")}s</td>'
            html += f'<td>{e.get("ground_truth", "")}</td>'
            html += f'<td>{e.get("fused_score", 0):.4f}</td></tr>'
        html += '</table></div>'

    html += _html_footer()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Event dashboard saved to {out}")
    return str(out)


if __name__ == "__main__":
    generate_event_dashboard()
