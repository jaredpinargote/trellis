import time
import threading
import numpy as np
from collections import deque

class Telemetry:
    """Thread-safe request telemetry collector."""
    def __init__(self, max_latencies=10000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.latencies: deque[float] = deque(maxlen=max_latencies)
        self.label_counts = {}

    def record_request(self, latency_ms, label, is_cached=False, is_error=False):
        with self._lock:
            self.total_requests += 1
            if is_error:
                self.total_errors += 1
            if is_cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            self.latencies.append(latency_ms)
            self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def snapshot(self, model_version="unknown", ood_threshold=0.0):
        with self._lock:
            uptime = time.time() - self.start_time
            lats = np.array(self.latencies) if self.latencies else np.array([0])
            return {
                "uptime_seconds": round(uptime, 1),
                "uptime_human": _format_uptime(uptime),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(self.total_errors / max(self.total_requests, 1), 4),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(self.cache_hits / max(self.total_requests, 1), 4),
                "latency": {
                    "mean_ms": round(float(np.mean(lats)), 2),
                    "p50_ms": round(float(np.median(lats)), 2),
                    "p95_ms": round(float(np.percentile(lats, 95)), 2),
                    "p99_ms": round(float(np.percentile(lats, 99)), 2),
                    "max_ms": round(float(np.max(lats)), 2),
                    "samples": len(self.latencies),
                },
                "label_distribution": dict(sorted(
                    self.label_counts.items(), key=lambda x: -x[1]
                )),
                "requests_per_second": round(
                    self.total_requests / max(uptime, 1), 1
                ),
                "model_version": model_version,
                "ood_threshold": ood_threshold,
            }


def _format_uptime(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
