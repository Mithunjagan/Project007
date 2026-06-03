"""
PROJECT 007 — P4.5 Final Scientific Validation Report
Generates a comprehensive HTML report answering key validation questions.

Usage:
    python -m evaluation.validation_report
"""

import json
import time
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def _load_json(path: str) -> dict:
    """Load a JSON file or return empty dict."""
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}


def generate_validation_report(
    output_path: str = "evaluation/reports/project007_validation_report.html",
) -> str:
    """
    Generate the final scientific validation report from all P4.5 analysis outputs.
    """
    # Load all reports
    error_analysis = _load_json("evaluation/reports/error_analysis.json")
    class_metrics = _load_json("evaluation/reports/class_metrics.json")
    feature_analysis = _load_json("evaluation/reports/feature_analysis.json")
    threshold_sens = _load_json("evaluation/reports/threshold_sensitivity.json")
    robustness = _load_json("evaluation/reports/robustness_eval.json")
    leakage = _load_json("evaluation/reports/leakage_check.json")
    p4_comparison = _load_json("evaluation/reports/p4_comparison.json")
    ml_baseline = _load_json("evaluation/reports/ml_baseline.json")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PROJECT 007 — Scientific Validation Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 30px; }}
h1 {{ color: #00d4ff; font-size: 32px; margin-bottom: 5px; }}
h2 {{ color: #ff6b35; font-size: 24px; margin: 40px 0 15px; border-bottom: 2px solid #21262d; padding-bottom: 8px; }}
h3 {{ color: #58a6ff; font-size: 18px; margin: 20px 0 10px; }}
.header {{ background: linear-gradient(135deg, #0d1117, #1a1e29); padding: 40px; border-radius: 16px; margin-bottom: 30px; border: 1px solid #21262d; }}
.subtitle {{ color: #8b949e; font-size: 16px; margin-top: 5px; }}
.timestamp {{ color: #484f58; font-size: 12px; margin-top: 10px; }}
.answer-box {{ background: #161b22; border-left: 4px solid #58a6ff; padding: 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
.answer-box.good {{ border-left-color: #3fb950; }}
.answer-box.warn {{ border-left-color: #d29922; }}
.answer-box.bad {{ border-left-color: #f85149; }}
.answer-box .question {{ color: #c9d1d9; font-weight: bold; font-size: 15px; margin-bottom: 8px; }}
.answer-box .answer {{ color: #e0e0e0; font-size: 14px; }}
.answer-box .value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
.answer-box .value.green {{ color: #3fb950; }}
.answer-box .value.yellow {{ color: #d29922; }}
.answer-box .value.red {{ color: #f85149; }}
.section {{ background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 25px; margin: 15px 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th {{ background: #161b22; color: #58a6ff; padding: 12px; text-align: left; font-size: 13px; text-transform: uppercase; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; font-size: 13px; }}
tr:hover {{ background: #161b22; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.tag-pass {{ background: #1b4332; color: #3fb950; }}
.tag-fail {{ background: #4a1919; color: #f85149; }}
.tag-warn {{ background: #3d2e00; color: #d29922; }}
.bar {{ height: 8px; border-radius: 4px; background: #21262d; overflow: hidden; display: inline-block; width: 100px; vertical-align: middle; }}
.bar-fill {{ height: 100%; border-radius: 4px; }}
.verdict {{ font-size: 20px; font-weight: bold; padding: 15px 25px; border-radius: 8px; text-align: center; margin: 20px 0; }}
.verdict-pass {{ background: #1b4332; color: #3fb950; border: 1px solid #3fb950; }}
.verdict-warn {{ background: #3d2e00; color: #d29922; border: 1px solid #d29922; }}
.verdict-fail {{ background: #4a1919; color: #f85149; border: 1px solid #f85149; }}
.footer {{ text-align: center; color: #484f58; padding: 40px; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>PROJECT 007 — Scientific Validation Report</h1>
    <div class="subtitle">P4.5 Framework — Comprehensive System Evaluation</div>
    <div class="timestamp">Generated: {timestamp}</div>
</div>

<h2>1. Key Questions Answered</h2>
"""

    # Q1: False Positive Rate
    fpr = error_analysis.get("metrics", {}).get("false_positive_rate", None)
    if fpr is not None:
        cls = "good" if fpr < 0.05 else ("warn" if fpr < 0.15 else "bad")
        val_cls = "green" if fpr < 0.05 else ("yellow" if fpr < 0.15 else "red")
        html += f"""
<div class="answer-box {cls}">
    <div class="question">What is the false positive rate?</div>
    <div class="value {val_cls}">{fpr:.4f} ({fpr*100:.2f}%)</div>
    <div class="answer">{'Excellent — below 5% threshold.' if fpr < 0.05 else 'Needs improvement.' if fpr < 0.15 else 'Too high — calibration required.'}</div>
</div>"""
    else:
        html += '<div class="answer-box warn"><div class="question">What is the false positive rate?</div><div class="answer">No error analysis data available. Run: <code>python -m evaluation.error_analysis</code></div></div>'

    # Q2: False Negative Rate
    fnr = error_analysis.get("metrics", {}).get("false_negative_rate", None)
    if fnr is not None:
        cls = "good" if fnr < 0.10 else ("warn" if fnr < 0.25 else "bad")
        val_cls = "green" if fnr < 0.10 else ("yellow" if fnr < 0.25 else "red")
        html += f"""
<div class="answer-box {cls}">
    <div class="question">What is the false negative rate?</div>
    <div class="value {val_cls}">{fnr:.4f} ({fnr*100:.2f}%)</div>
    <div class="answer">{'Good — most anomalies detected.' if fnr < 0.10 else 'Some anomalies missed.' if fnr < 0.25 else 'Too many missed detections.'}</div>
</div>"""
    else:
        html += '<div class="answer-box warn"><div class="question">What is the false negative rate?</div><div class="answer">No data available.</div></div>'

    # Q3: Which rules cause most errors?
    fp_rules = error_analysis.get("false_positive_rule_attribution", {})
    if fp_rules:
        top_rule = list(fp_rules.items())[0] if fp_rules else ("none", 0)
        html += f"""
<div class="answer-box warn">
    <div class="question">Which rules cause the most errors?</div>
    <div class="value yellow">{top_rule[0]}</div>
    <div class="answer">Top FP-causing rule with {top_rule[1]} false positive attributions. Full breakdown:<br>"""
        for rule, count in list(fp_rules.items())[:5]:
            html += f"&nbsp;&nbsp;• {rule}: {count}<br>"
        html += "</div></div>"
    else:
        html += '<div class="answer-box warn"><div class="question">Which rules cause most errors?</div><div class="answer">No data available.</div></div>'

    # Q4: Which features matter most?
    consensus = feature_analysis.get("consensus_ranking", [])
    if consensus:
        html += f"""
<div class="answer-box good">
    <div class="question">Which features matter most?</div>
    <div class="answer">Consensus ranking across RF, XGBoost, and SHAP:<br>"""
        for rank, (feat, avg) in enumerate(consensus[:5], 1):
            html += f"&nbsp;&nbsp;{rank}. <b>{feat}</b> (avg rank: {avg:.1f})<br>"
        html += "</div></div>"
    else:
        html += '<div class="answer-box warn"><div class="question">Which features matter most?</div><div class="answer">No feature analysis data. Run: <code>python -m evaluation.feature_analysis</code></div></div>'

    # Q5: Is hybrid better than rules-only?
    if p4_comparison:
        rules_f1 = p4_comparison.get("rules_only", {}).get("f1", 0)
        hybrid_f1 = p4_comparison.get("hybrid", {}).get("f1", 0)
        ml_f1 = p4_comparison.get("ml_only", {}).get("f1", 0)
        better = hybrid_f1 > rules_f1
        cls = "good" if better else "warn"
        html += f"""
<div class="answer-box {cls}">
    <div class="question">Is hybrid actually better than rules-only?</div>
    <div class="value {'green' if better else 'yellow'}">{'YES' if better else 'NO'}</div>
    <div class="answer">
        Rules-only F1: {rules_f1:.4f}<br>
        ML-only F1: {ml_f1:.4f}<br>
        Hybrid F1: {hybrid_f1:.4f}<br>
        {'Hybrid improves F1 by ' + f'{(hybrid_f1-rules_f1)*100:.2f}' + ' percentage points.' if better else 'Rules-only performs equally or better. ML may not be adding value yet.'}
    </div>
</div>"""
    else:
        html += '<div class="answer-box warn"><div class="question">Is hybrid actually better than rules-only?</div><div class="answer">No comparison data. Run: <code>python -m evaluation.p4_comparison</code></div></div>'

    # Section 2: Per-Class Performance
    per_class = class_metrics.get("per_class", {})
    if per_class:
        html += """
<h2>2. Per-Class Performance</h2>
<div class="section">
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th><th>Assessment</th></tr>"""
        for cls, m in per_class.items():
            f1 = m.get("f1", 0)
            tag = "tag-pass" if f1 > 0.7 else ("tag-warn" if f1 > 0.4 else "tag-fail")
            label = "Good" if f1 > 0.7 else ("Fair" if f1 > 0.4 else "Poor")
            html += f"""<tr><td>{cls}</td>
<td>{m.get('precision', 0):.4f}</td><td>{m.get('recall', 0):.4f}</td>
<td>{f1:.4f}</td><td>{m.get('support', 0)}</td>
<td><span class="tag {tag}">{label}</span></td></tr>"""
        html += "</table></div>"

    # Section 3: Threshold Sensitivity
    thresh_results = threshold_sens.get("results", [])
    if thresh_results:
        html += """
<h2>3. Threshold Sensitivity</h2>
<div class="section">
<table>
<tr><th>Variation</th><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th></tr>"""
        for r in thresh_results:
            var = r.get("variation_pct", 0)
            highlight = ' style="background:#1a1e29;"' if var == 0 else ""
            html += f"""<tr{highlight}><td>{var:+d}%</td><td>{r.get('threshold', 0):.4f}</td>
<td>{r.get('precision', 0):.4f}</td><td>{r.get('recall', 0):.4f}</td><td>{r.get('f1', 0):.4f}</td></tr>"""
        html += "</table></div>"

    # Section 4: Robustness
    robustness_results = robustness.get("results", {})
    if robustness_results:
        html += """
<h2>4. Robustness Under Perturbations</h2>
<div class="section">
<table>
<tr><th>Perturbation</th><th>Precision</th><th>Recall</th><th>F1</th><th>F1 Degradation</th></tr>"""
        for perturb, r in robustness_results.items():
            deg = r.get("f1_degradation_pct", 0)
            deg_str = f"{deg:+.1f}%" if perturb != "none" else "—"
            tag = "tag-pass" if deg < 10 else ("tag-warn" if deg < 25 else "tag-fail")
            html += f"""<tr><td>{perturb}</td>
<td>{r.get('precision', 0):.4f}</td><td>{r.get('recall', 0):.4f}</td>
<td>{r.get('f1', 0):.4f}</td>
<td><span class="tag {tag}">{deg_str}</span></td></tr>"""
        html += "</table></div>"

    # Section 5: Leakage Check
    if leakage:
        has_leak = leakage.get("has_video_leakage", False) or leakage.get("has_frame_leakage", False)
        tag = "tag-fail" if has_leak else "tag-pass"
        label = "FAIL" if has_leak else "PASS"
        html += f"""
<h2>5. Dataset Leakage Check</h2>
<div class="section">
<p>Video-level leakage: <span class="tag {tag}">{label}</span></p>
<p>Frame-level leakage: {leakage.get('frame_leakage_count', 0)} overlapping frames</p>
<p>Train videos: {leakage.get('train_videos', 0)} | Test videos: {leakage.get('test_videos', 0)}</p>
</div>"""

    # Section 6: ML Baseline
    comparison = ml_baseline.get("comparison", {})
    if comparison:
        html += """
<h2>6. ML Model Comparison</h2>
<div class="section">
<table>
<tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Inference (ms)</th><th>Train (s)</th></tr>"""
        for model, m in comparison.items():
            html += f"""<tr><td>{model}</td>
<td>{m.get('accuracy', 0):.4f}</td><td>{m.get('macro_f1', 0):.4f}</td>
<td>{m.get('inference_latency_ms', 0):.4f}</td><td>{m.get('train_time_sec', 0):.2f}</td></tr>"""
        html += "</table>"
        rec = ml_baseline.get("recommended_model", "")
        if rec:
            html += f'<p style="margin-top:10px;">Recommended model: <b>{rec}</b></p>'
        html += "</div>"

    # Overall Verdict
    html += '<h2>7. Overall Verdict</h2>'
    precision = error_analysis.get("metrics", {}).get("precision", 0)
    recall = error_analysis.get("metrics", {}).get("recall", 0)
    f1 = error_analysis.get("metrics", {}).get("f1", 0)

    if f1 > 0.7 and fpr is not None and fpr < 0.10:
        html += '<div class="verdict verdict-pass">SYSTEM VALIDATED — Ready for controlled deployment</div>'
    elif f1 > 0.4:
        html += '<div class="verdict verdict-warn">SYSTEM NEEDS IMPROVEMENT — Calibration and more data required</div>'
    elif error_analysis:
        html += '<div class="verdict verdict-fail">SYSTEM NOT READY — Significant errors detected</div>'
    else:
        html += '<div class="verdict verdict-warn">INSUFFICIENT DATA — Run all analysis modules first</div>'

    # Instructions
    html += """
<h2>8. How to Generate Full Report</h2>
<div class="section">
<pre style="color:#58a6ff;background:#161b22;padding:15px;border-radius:8px;font-size:13px;overflow-x:auto;">
# Step 1: Build training dataset
python -m training.dataset_builder

# Step 2: Train ML models
python -m models.evaluator

# Step 3: Run all analyses
python -m evaluation.error_analysis
python -m evaluation.class_metrics
python -m evaluation.feature_analysis --dataset dataset/training_dataset.parquet
python -m evaluation.threshold_sensitivity
python -m evaluation.robustness_eval
python -m evaluation.leakage_check --dataset dataset/training_dataset.parquet
python -m evaluation.p4_comparison

# Step 4: Generate reports
python -m evaluation.event_dashboard
python -m evaluation.validation_report
</pre>
</div>"""

    html += f"""
<div class="footer">
    PROJECT 007 — P4.5 Scientific Validation Report — {timestamp}
</div>
</div>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Validation report saved to {out}")
    return str(out)


if __name__ == "__main__":
    generate_validation_report()
