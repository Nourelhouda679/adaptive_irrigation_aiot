import random
import psutil
import numpy as np

# ── thresholds ─────────────────────────────────────────────────
MAX_EDGE_CPU        = 80     # % — إذا تجاوز CPU هذا الحد → السحابة
MAX_NETWORK_LATENCY = 120    # ms — إذا تجاوز هذا الحد → الحافة (تجنب التأخير)
MIN_WARMUP_SAMPLES  = 15     # عدد القراءات اللازمة قبل الحساب التكيفي

# ── adaptive threshold state ───────────────────────────────────
_score_history: list[float] = []


def update_adaptive_threshold(new_conf: float) -> float:
    """
    يحسب EDGE_THRESHOLD تكيفياً بناءً على آخر 20 قيمة confidence.

    - أقل من MIN_WARMUP_SAMPLES قراءة: قيمة تدريجية محافظة
    - بعد الـ warmup: mean - 0.5 * std (مقيّدة بين 0.50 و 0.90)
    """
    _score_history.append(new_conf)
    if len(_score_history) > 20:
        _score_history.pop(0)

    n = len(_score_history)

    if n < MIN_WARMUP_SAMPLES:
        # warm-up تدريجي: يبدأ من 0.70 وينخفض تدريجياً مع تراكم البيانات
        ratio = n / MIN_WARMUP_SAMPLES
        return round(0.70 - 0.05 * ratio, 3)

    mean   = np.mean(_score_history)
    std    = np.std(_score_history)
    thresh = mean - 0.5 * std
    return float(np.clip(thresh, 0.50, 0.90))


# ── helpers ────────────────────────────────────────────────────

def simulate_network_latency() -> int:
    """محاكاة تأخير الشبكة بتوزيع واقعي."""
    roll = random.random()
    if roll < 0.70:
        return random.randint(20, 80)     # شبكة جيدة   — 70% من الوقت
    elif roll < 0.90:
        return random.randint(80, 150)    # متوسطة      — 20%
    else:
        return random.randint(150, 400)   # بطيئة       — 10%


def get_edge_cpu_usage() -> float:
    """قراءة CPU الحقيقية للجهاز المحلي (edge)."""
    return psutil.cpu_percent(interval=0.1)


# ── main router ────────────────────────────────────────────────

def dynamic_router(edge_confidence: float) -> dict:
    """
    يقرر تشغيل الاستدلال على الحافة أو السحابة بمنطق edge-first.

    منطق الأولويات (بالترتيب):
    1. ثقة الحافة أقل من الحد التكيفي  → سحابة (النموذج غير واثق)
    2. CPU الحافة مثقل                  → سحابة (موارد غير كافية)
    3. الشبكة بطيئة (> 120ms)           → حافة  (تجنب التأخير)
    4. كل شيء مناسب                     → حافة  (edge-first افتراضياً)

    Parameters
    ----------
    edge_confidence : float — درجة ثقة محرك القواعد (0–1)

    Returns
    -------
    dict : use_edge, network_latency, edge_cpu, reason, adaptive_thresh
    """
    network_latency = simulate_network_latency()
    edge_cpu        = get_edge_cpu_usage()
    adaptive_thresh = update_adaptive_threshold(edge_confidence)

    # ── منطق التوجيه edge-first ────────────────────────────────

    if edge_confidence < adaptive_thresh:
        use_edge = False
        reason   = (
            f"Low edge confidence ({edge_confidence:.2f} < "
            f"adaptive thresh {adaptive_thresh:.2f}) → Cloud"
        )

    elif edge_cpu >= MAX_EDGE_CPU:
        use_edge = False
        reason   = (
            f"Edge CPU overloaded ({edge_cpu:.1f}% ≥ {MAX_EDGE_CPU}%) → Cloud"
        )

    elif network_latency > MAX_NETWORK_LATENCY:
        # شبكة بطيئة → الحافة أفضل لتجنب التأخير
        use_edge = True
        reason   = (
            f"Slow network ({network_latency} ms > {MAX_NETWORK_LATENCY} ms) "
            f"→ Edge to avoid latency"
        )

    else:
        # كل الظروف مناسبة → حافة افتراضياً (edge-first)
        use_edge = True
        reason   = (
            f"Edge-first: conf={edge_confidence:.2f} ≥ thresh={adaptive_thresh:.2f}, "
            f"cpu={edge_cpu:.1f}%, latency={network_latency} ms"
        )

    return {
        "use_edge"        : use_edge,
        "network_latency" : network_latency,
        "edge_cpu"        : edge_cpu,
        "reason"          : reason,
        "adaptive_thresh" : round(adaptive_thresh, 3),
    }
