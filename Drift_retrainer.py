import threading
import numpy as np
import pandas as pd
from collections import deque
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


class DriftRetrainer:
    """
    Adaptive Drift Retrainer — Sliding Window Buffer.

    المنطق:
    1. كل قراءة IoT جديدة عالية الثقة (>= min_label_conf) تُضاف
       إلى buffer متحرك (آخر buffer_size قراءة).
    2. إذا انخفض متوسط الثقة في النافذة الأخيرة عن threshold،
       يُطلق إعادة تدريب في الخلفية على buffer فقط (بدون dataset).
    3. SMOTE يُطبَّق داخل الـ retrain لمعالجة اختلال الفئات.
    4. النموذج القديم يبقى حتى ينتهي الـ retrain (zero downtime).
    """

    def __init__(
        self,
        model,
        feature_cols      : list,
        target_encoder,
        buffer_size       : int   = 500,   # أقصى عدد قراءات في الـ buffer
        window_size       : int   = 20,    # نافذة الكشف عن الـ drift
        threshold         : float = 0.75,  # حد الثقة لاكتشاف الـ drift
        min_label_conf    : float = 0.80,  # أدنى ثقة لقبول قراءة في الـ buffer
        min_buffer_retrain: int   = 50,    # أدنى عدد قراءات لبدء الـ retrain
        min_retrain_gap   : int   = 30,    # أدنى قراءات بين كل retrain وآخر
    ):
        self._model             = model
        self._feature_cols      = feature_cols
        self._target_encoder    = target_encoder
        self._buffer_size       = buffer_size
        self._window            = window_size
        self._threshold         = threshold
        self._min_label_conf    = min_label_conf
        self._min_buffer_retrain= min_buffer_retrain
        self._min_gap           = min_retrain_gap

        # sliding window buffer — يحتفظ فقط بآخر buffer_size قراءة
        self._buffer_X : deque = deque(maxlen=buffer_size)
        self._buffer_y : deque = deque(maxlen=buffer_size)

        # نافذة متابعة الثقة
        self._scores   : deque = deque(maxlen=window_size)

        self._lock           = threading.Lock()
        self._retraining     = False
        self._retrain_count  = 0
        self._readings_since = 0
        self.last_status     = "✅ No drift detected"

    # ── public API ────────────────────────────────────────────

    def observe(self, confidence_pct: float):
        """
        أضف confidence score جديد (0–100).
        يُطلق إعادة التدريب تلقائياً عند اكتشاف drift.
        """
        self._readings_since += 1
        self._scores.append(confidence_pct)

        if (
            len(self._scores) >= self._window
            and not self._retraining
            and self._readings_since >= self._min_gap
            and len(self._buffer_X) >= self._min_buffer_retrain
        ):
            avg = np.mean(list(self._scores))
            if avg < self._threshold * 100:
                self._trigger_retrain()

    def add_sample(self, x_encoded: pd.Series, y_label_encoded: int):
        """
        أضف قراءة IoT جديدة موثوقة إلى الـ buffer.
        يُستدعى من الخارج فقط عند confidence >= min_label_conf.

        Parameters
        ----------
        x_encoded       : pd.Series — القراءة بعد الترميز (مطابقة feature_cols)
        y_label_encoded : int       — الفئة المُشفَّرة (0/1/2)
        """
        self._buffer_X.append(x_encoded.values)
        self._buffer_y.append(y_label_encoded)

    def get_model(self):
        """أعد النموذج الحالي (قد يكون مُعاداً تدريبه)."""
        with self._lock:
            return self._model

    def get_status(self) -> dict:
        scores_list = list(self._scores)
        return {
            "status"         : self.last_status,
            "retrain_count"  : self._retrain_count,
            "window_avg"     : round(np.mean(scores_list), 1) if scores_list else 100.0,
            "is_retraining"  : self._retraining,
            "buffer_size"    : len(self._buffer_X),
            "drift_detected" : (
                np.mean(scores_list) < self._threshold * 100
                if len(scores_list) >= self._window else False
            ),
        }

    # ── internal ──────────────────────────────────────────────

    def _trigger_retrain(self):
        self._retraining     = True
        self._readings_since = 0
        self.last_status     = "⚠️ Drift detected — retraining on recent data..."
        print(
            f"[DriftRetrainer] Drift detected "
            f"(avg={np.mean(list(self._scores)):.1f}%) — "
            f"retraining on {len(self._buffer_X)} buffered samples"
        )
        t = threading.Thread(target=self._retrain_worker, daemon=True)
        t.start()

    def _retrain_worker(self):
        try:
            # ── بناء DataFrame من الـ buffer ──────────────────
            X_buf = pd.DataFrame(
                list(self._buffer_X),
                columns=self._feature_cols,
            )
            y_buf = np.array(list(self._buffer_y))

            # ── SMOTE لمعالجة اختلال الفئات ───────────────────
            unique, counts = np.unique(y_buf, return_counts=True)
            # SMOTE يحتاج على الأقل فئتين وكل فئة >= 2 عينات
            can_smote = (
                len(unique) >= 2
                and counts.min() >= 2
                and len(X_buf) >= 10
            )

            if can_smote:
                k = max(1, min(5, counts.min() - 1))
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_res, y_res = smote.fit_resample(X_buf, y_buf)
            else:
                X_res, y_res = X_buf, y_buf

            # ── تدريب النموذج الجديد ───────────────────────────
            new_model = XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            new_model.fit(X_res, y_res)

            # ── استبدال النموذج القديم بأمان (thread-safe) ────
            with self._lock:
                self._model = new_model
                self._retrain_count += 1

            self.last_status = (
                f"✅ Retrained on {len(X_buf)} recent samples "
                f"(#{self._retrain_count})"
            )
            print(
                f"[DriftRetrainer] Retrain #{self._retrain_count} complete "
                f"— trained on {len(X_res)} samples "
                f"({'with' if can_smote else 'without'} SMOTE)"
            )

        except Exception as e:
            self.last_status = f"❌ Retraining failed: {e}"
            print(f"[DriftRetrainer] Error: {e}")
        finally:
            self._retraining = False
