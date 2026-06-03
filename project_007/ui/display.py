"""
PROJECT 007 — Visual Debug Overlay
Renders per-person annotations AND a system telemetry HUD.

P1: Added proxy rules panel, behavior score bar, and recording indicator.
"""

import cv2

from config import (
    COLOR_NORMAL,
    COLOR_WARNING,
    COLOR_DANGER,
    ARM_VELOCITY_HIGH,
    FALL_SCORE_HIGH,
    FRAME_AGE_WARN_MS,
    FRAME_AGE_DANGER_MS,
    POSE_QUEUE_MAX,
    QUEUE_WARN_RATIO,
    QUEUE_DANGER_RATIO,
    MAX_RESULT_AGE_MS,
)
from pipeline.events import RuleEvent, BehaviorScore
from utils.logger import get_logger

logger = get_logger(__name__)

# Colours for the telemetry HUD
_GREEN  = (0, 255, 0)
_ORANGE = (0, 165, 255)
_RED    = (0, 0, 255)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)


class DebugOverlay:
    """
    Draws multiple overlay layers:
    1. Per-person annotations (bboxes, tracks)
    2. Telemetry HUD (top-right)
    3. Active Rules Panel (bottom-left)
    4. Behavior Score Bar (bottom-center)
    """

    def __init__(self):
        self._realtime_lost_flash: bool = False  # toggle for flashing
        self._recording_flash: bool = False
        logger.info("DebugOverlay initialised (P1 telemetry HUD)")

    def render(
        self, frame, detections, all_motion, telemetry: dict,
        fused_evidence: list = None,
        is_recording: bool = False
    ):
        """Annotate *frame* and return the result."""
        overlay = frame.copy()

        # ── 1. Per-person annotations ──
        for det in detections:
            motion = all_motion.get(det.track_id, {})
            fall_score = motion.get("fall_score", 0.0)
            arm_vel = motion.get("arm_velocity", 0.0)

            color = self._pick_person_color(arm_vel, fall_score)
            x1, y1, x2, y2 = det.bbox

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            self._label(overlay, f"ID:{det.track_id}", (x1, y1 - 10), color)

        # ── 2. Telemetry HUD (top-right) ──
        self._render_telemetry_hud(overlay, telemetry)

        # ── 3. REALTIME LOST warning ──
        frame_age = telemetry.get("frame_age_ms", 0.0)
        if frame_age > MAX_RESULT_AGE_MS:
            self._render_realtime_lost(overlay)

        # ── 4. Thread stall warnings ──
        warnings = telemetry.get("warnings", [])
        if warnings:
            self._render_warnings(overlay, warnings)

        # ── 5. P2 Fusion HUD ──
        if fused_evidence:
            self._render_fused_evidence(overlay, fused_evidence)
            self._render_top_center_threat_warnings(overlay, fused_evidence)

        # ── 7. P1 Recording Indicator ──
        if is_recording:
            self._render_recording_indicator(overlay)

        # ── 8. P1.5 Scene Instability Meter ──
        flow_mag = telemetry.get("global_flow_magnitude", 0.0)
        self._render_instability_meter(overlay, flow_mag)

        return overlay

    def _render_telemetry_hud(self, frame, t: dict):
        h, w = frame.shape[:2]
        fps           = t.get("fps", 0.0)
        frame_age     = t.get("frame_age_ms", 0.0)
        det_lat       = t.get("detection_latency_ms", 0.0)
        pose_lat      = t.get("pose_latency_ms", 0.0)
        q_depth       = int(t.get("queue_depth", 0))
        q_wait        = t.get("queue_wait_ms", 0.0)
        drops         = int(t.get("dropped_crops", 0))
        tracks        = int(t.get("active_tracks", 0))
        cpu           = t.get("cpu_util", 0.0)
        gpu           = t.get("gpu_util", 0.0)
        vram          = t.get("vram_gb", 0.0)
        persons       = int(t.get("person_count", 0))

        lines = [
            (f"FPS: {fps:.1f}",          self._health_color(fps, 15, 10, invert=True)),
            (f"FRAME AGE: {frame_age:.0f}ms", self._age_color(frame_age)),
            (f"YOLO: {det_lat:.0f}ms",   self._latency_color(det_lat, 100, 200)),
            (f"POSE: {pose_lat:.0f}ms",  self._latency_color(pose_lat, 80, 150)),
            (f"Q-WAIT: {q_wait:.0f}ms",  self._latency_color(q_wait, 50, 100)),
            (f"QUEUE: {q_depth}/{POSE_QUEUE_MAX}", self._queue_color(q_depth)),
            (f"DROP: {drops}",           _GREEN if drops == 0 else _ORANGE),
            (f"TRACKS: {tracks}",        _WHITE),
            (f"PERSONS: {persons}",      _WHITE),
            (f"CPU: {cpu:.0f}%",         self._health_color(100 - cpu, 30, 15, invert=True)),
        ]

        if gpu > 0 or vram > 0:
            lines.append((f"GPU: {gpu:.0f}%",   self._health_color(100 - gpu, 30, 15, invert=True)))
            lines.append((f"VRAM: {vram:.1f}GB", _WHITE))

        panel_w = 200
        panel_h = 22 * len(lines) + 10
        x0 = w - panel_w - 10
        y0 = 10

        sub = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        if sub.size > 0:
            cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), _BLACK, cv2.FILLED)
            cv2.addWeighted(sub, 0.3, frame[y0:y0 + panel_h, x0:x0 + panel_w], 0.7, 0, frame[y0:y0 + panel_h, x0:x0 + panel_w])

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (text, color) in enumerate(lines):
            ty = y0 + 20 + i * 22
            cv2.putText(frame, text, (x0 + 8, ty), font, 0.45, color, 1, cv2.LINE_AA)

    def _render_top_center_threat_warnings(self, frame, fused_evidence: list):
        """Renders large top-center warnings for direct camera threats or critical states."""
        h, w = frame.shape[:2]
        threats = []
        for fe in fused_evidence:
            if fe.state == "CRITICAL":
                threats.append("CRITICAL THREAT DETECTED")
            else:
                for rule_type in fe.contributing_rules:
                    if rule_type in ["CAMERA_SHAKE", "CAMERA_BLOCKAGE"]:
                        threats.append("CAMERA TAMPERING")
                    elif rule_type == "LENS_OCCLUSION":
                        threats.append("LENS OCCLUSION")
                    elif rule_type in ["CAMERA_RUSH", "PROXIMITY_INTRUSION"]:
                        threats.append("PROXIMITY ALERT")
                    elif rule_type == "ABNORMAL_SINGLE_SUBJECT_ENERGY":
                        threats.append("HIGH-ENERGY SINGLE SUBJECT")
        
        # Deduplicate
        threats = list(dict.fromkeys(threats))

        y0 = 30
        for threat in threats:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(threat, font, 1.0, 3)
            cx = (w - tw) // 2
            
            # Flash effect
            if int(cv2.getTickCount() / cv2.getTickFrequency() * 4) % 2 == 0:
                cv2.rectangle(frame, (cx - 10, y0 - th - 10), (cx + tw + 10, y0 + 10), _RED, cv2.FILLED)
                cv2.putText(frame, threat, (cx, y0), font, 1.0, _WHITE, 3, cv2.LINE_AA)
            
            y0 += (th + 30)

    def _render_instability_meter(self, frame, flow_mag: float):
        """Renders a small vertical bar for scene instability on the left edge."""
        h, w = frame.shape[:2]
        bar_w = 15
        bar_h = 100
        x0 = 10
        y0 = h // 2 - bar_h // 2
        
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), cv2.FILLED)
        
        # Scale flow_mag against TAMPER_SHAKE_MAGNITUDE (15.0)
        ratio = flow_mag / 15.0
        
        fill_h = min(bar_h, int(bar_h * ratio))
        if fill_h > 0:
            color = _RED if ratio > 0.8 else _ORANGE if ratio > 0.4 else _GREEN
            cv2.rectangle(frame, (x0, y0 + bar_h - fill_h), (x0 + bar_w, y0 + bar_h), color, cv2.FILLED)
            
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), _WHITE, 1)
        cv2.putText(frame, "INSTB", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA)
    def _render_fused_evidence(self, frame, fused_evidence: list):
        """Render active P2 Fusion States in the bottom left and bottom center."""
        if not fused_evidence:
            return

        h, w = frame.shape[:2]
        x0 = 10
        y0 = h - 150

        cv2.putText(frame, "CONTEXTUAL THREAT FUSION:", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)
        y0 += 20

        highest_score = 0.0
        is_anomalous = False

        for i, fe in enumerate(fused_evidence[:3]):  # Show max 3 active tracking groups
            state_col = _GREEN
            if fe.state == "SUSPICIOUS": state_col = _ORANGE
            elif fe.state == "HIGH_RISK": state_col = _RED
            elif fe.state == "CRITICAL": state_col = _RED

            text = f"T-IDs {list(fe.track_ids)}: {fe.state} [{fe.evidence_score:.2f}]"
            cv2.putText(frame, text, (x0 + 10, y0 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_col, 1, cv2.LINE_AA)
            
            # Show contributing rules beneath
            if fe.contributing_rules:
                rule_txt = " + ".join(fe.contributing_rules[:2])
                cv2.putText(frame, f"  -> {rule_txt}", (x0 + 10, y0 + (i * 20) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _WHITE, 1, cv2.LINE_AA)
            
            y0 += 15
            
            if fe.evidence_score > highest_score:
                highest_score = fe.evidence_score
            if fe.state in ["HIGH_RISK", "CRITICAL"]:
                is_anomalous = True

        # Render global trend bar (replaces old behavior score)
        bar_w = 400
        bar_h = 20
        bx0 = (w - bar_w) // 2
        by0 = h - 50

        # Background
        cv2.rectangle(frame, (bx0, by0), (bx0 + bar_w, by0 + bar_h), (50, 50, 50), cv2.FILLED)
        
        # Fill
        fill_w = int(bar_w * min(1.0, highest_score))
        color = _GREEN if highest_score < 0.35 else (_ORANGE if highest_score < 0.6 else _RED)

        if fill_w > 0:
            cv2.rectangle(frame, (bx0, by0), (bx0 + fill_w, by0 + bar_h), color, cv2.FILLED)

        # Border
        cv2.rectangle(frame, (bx0, by0), (bx0 + bar_w, by0 + bar_h), _WHITE, 1)

        # Text
        cv2.putText(frame, f"MAX RISK SCORE: {highest_score:.2f}", (bx0, by0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)

        if is_anomalous:
            text = "OBSERVED ANOMALOUS ACTIVITY"
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, text, (bx0 + (bar_w - tw) // 2, by0 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2, cv2.LINE_AA)

    def _render_recording_indicator(self, frame):
        """Flash a red recording indicator."""
        self._recording_flash = not self._recording_flash
        if not self._recording_flash:
            return

        h, w = frame.shape[:2]
        text = "REC"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (w - 70, h - 30), font, 0.7, _RED, 2, cv2.LINE_AA)
        cv2.circle(frame, (w - 90, h - 35), 8, _RED, cv2.FILLED)

    def _render_realtime_lost(self, frame):
        self._realtime_lost_flash = not self._realtime_lost_flash
        if not self._realtime_lost_flash:
            return

        h, w = frame.shape[:2]
        text = "REALTIME LOST"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cx, cy = (w - tw) // 2, h // 2
        cv2.rectangle(frame, (cx - 20, cy - th - 15), (cx + tw + 20, cy + 15), _RED, cv2.FILLED)
        cv2.putText(frame, text, (cx, cy), font, scale, _WHITE, thickness, cv2.LINE_AA)

    def _render_warnings(self, frame, warnings: list[str]):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, msg in enumerate(warnings):
            y = h - 30 - i * 25
            (tw, th), _ = cv2.getTextSize(msg, font, 0.6, 2)
            cv2.rectangle(frame, (10, y - th - 5), (tw + 20, y + 5), _RED, cv2.FILLED)
            cv2.putText(frame, msg, (15, y), font, 0.6, _WHITE, 2, cv2.LINE_AA)

    @staticmethod
    def _pick_person_color(arm_velocity: float, fall_score: float):
        if fall_score >= FALL_SCORE_HIGH:
            return COLOR_DANGER
        if arm_velocity >= ARM_VELOCITY_HIGH:
            return COLOR_WARNING
        return COLOR_NORMAL

    @staticmethod
    def _age_color(age_ms: float):
        if age_ms >= FRAME_AGE_DANGER_MS: return _RED
        if age_ms >= FRAME_AGE_WARN_MS: return _ORANGE
        return _GREEN

    @staticmethod
    def _latency_color(latency_ms: float, warn: float, danger: float):
        if latency_ms >= danger: return _RED
        if latency_ms >= warn: return _ORANGE
        return _GREEN

    @staticmethod
    def _queue_color(depth: int):
        ratio = depth / max(POSE_QUEUE_MAX, 1)
        if ratio >= QUEUE_DANGER_RATIO: return _RED
        if ratio >= QUEUE_WARN_RATIO: return _ORANGE
        return _GREEN

    @staticmethod
    def _health_color(value: float, warn: float, danger: float, invert: bool = False):
        if invert:
            if value <= danger: return _RED
            if value <= warn: return _ORANGE
            return _GREEN
        if value >= danger: return _RED
        if value >= warn: return _ORANGE
        return _GREEN

    @staticmethod
    def _label(frame, text, position, color, scale=0.50, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = int(position[0]), int(position[1])
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), _BLACK, cv2.FILLED)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
