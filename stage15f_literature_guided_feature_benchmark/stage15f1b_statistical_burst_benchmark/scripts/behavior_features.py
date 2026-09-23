from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import feature_names


FEATURE_NAMES = feature_names()


def _stats(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


@dataclass
class BehaviorAccumulator:
    flow_uid: str
    timestamps: list[float] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    directions: list[int] = field(default_factory=list)
    payload_lengths: list[float] = field(default_factory=list)
    first_endpoint: tuple[str, str] | None = None

    def add(
        self,
        timestamp: float,
        frame_length: int,
        payload_length: int | None,
        source: tuple[str, str] | None,
        destination: tuple[str, str] | None,
        explicit_direction: int | None = None,
    ) -> None:
        if explicit_direction is None:
            if self.first_endpoint is None and source is not None:
                self.first_endpoint = source
            if source is None or destination is None or self.first_endpoint is None:
                direction = 0
            elif source == self.first_endpoint:
                direction = 1
            elif destination == self.first_endpoint:
                direction = -1
            else:
                direction = 0
        else:
            direction = int(np.sign(explicit_direction))
        self.timestamps.append(float(timestamp))
        self.lengths.append(float(max(frame_length, 0)))
        self.directions.append(direction)
        self.payload_lengths.append(float(max(payload_length, 0)) if payload_length is not None else float("nan"))

    def vector(self, max_packets: int | None = None) -> np.ndarray:
        n = len(self.timestamps) if max_packets is None else min(len(self.timestamps), max_packets)
        if n <= 0:
            raise RuntimeError(f"empty target flow: {self.flow_uid}")
        timestamps = np.asarray(self.timestamps[:n], dtype=np.float64)
        lengths = np.asarray(self.lengths[:n], dtype=np.float64)
        directions = np.asarray(self.directions[:n], dtype=np.int8)
        payloads = np.asarray(self.payload_lengths[:n], dtype=np.float64)
        iats = np.maximum(0.0, np.diff(timestamps))
        length_q25, length_q50, length_q75, length_q90 = np.quantile(lengths, [0.25, 0.50, 0.75, 0.90])
        if iats.size:
            iat_q50, iat_q90 = np.quantile(iats, [0.50, 0.90])
        else:
            iat_q50 = iat_q90 = float("nan")
        duration = float(max(0.0, timestamps[-1] - timestamps[0]))
        forward = directions > 0
        reverse = directions < 0
        nonzero = directions[directions != 0]
        direction_changes = float(np.sum(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0.0
        length_mean, length_std = _stats(lengths)
        iat_mean, iat_std = _stats(iats)
        fwd_lengths, rev_lengths = lengths[forward], lengths[reverse]
        fwd_length_mean, fwd_length_std = _stats(fwd_lengths)
        rev_length_mean, rev_length_std = _stats(rev_lengths)
        fwd_times, rev_times = timestamps[forward], timestamps[reverse]
        fwd_iat_mean, fwd_iat_std = _stats(np.maximum(0.0, np.diff(fwd_times)))
        rev_iat_mean, rev_iat_std = _stats(np.maximum(0.0, np.diff(rev_times)))

        burst_directions: list[int] = []
        burst_counts: list[int] = []
        burst_bytes: list[float] = []
        burst_starts: list[float] = []
        burst_ends: list[float] = []
        for timestamp, length, direction in zip(timestamps, lengths, directions):
            if not burst_directions or direction != burst_directions[-1]:
                burst_directions.append(int(direction))
                burst_counts.append(1)
                burst_bytes.append(float(length))
                burst_starts.append(float(timestamp))
                burst_ends.append(float(timestamp))
            else:
                burst_counts[-1] += 1
                burst_bytes[-1] += float(length)
                burst_ends[-1] = float(timestamp)
        burst_counts_array = np.asarray(burst_counts, dtype=np.float64)
        burst_bytes_array = np.asarray(burst_bytes, dtype=np.float64)
        burst_durations = np.maximum(0.0, np.asarray(burst_ends) - np.asarray(burst_starts))
        inter_burst = np.maximum(0.0, np.asarray(burst_starts[1:]) - np.asarray(burst_ends[:-1]))
        burst_count_mean, burst_count_std = _stats(burst_counts_array)
        burst_bytes_mean, burst_bytes_std = _stats(burst_bytes_array)
        burst_duration_mean, burst_duration_std = _stats(burst_durations)
        inter_burst_mean, inter_burst_std = _stats(inter_burst)
        payload_bytes = float(payloads.sum()) if np.isfinite(payloads).all() else float("nan")
        fwd_bytes = float(lengths[forward].sum())
        rev_bytes = float(lengths[reverse].sum())

        values = {
            "packet_count": float(n),
            "total_bytes": float(lengths.sum()),
            "duration_seconds": duration,
            "packet_length_mean": length_mean,
            "packet_length_std": length_std,
            "iat_mean_seconds": iat_mean,
            "iat_std_seconds": iat_std,
            "forward_packet_count": float(forward.sum()),
            "reverse_packet_count": float(reverse.sum()),
            "forward_reverse_packet_ratio": float((forward.sum() + 1.0) / (reverse.sum() + 1.0)),
            "direction_changes": direction_changes,
            "payload_bytes": payload_bytes,
            "packets_per_second": float(n / duration) if duration > 0 else float("nan"),
            "bytes_per_second": float(lengths.sum() / duration) if duration > 0 else float("nan"),
            "packet_length_min": float(lengths.min()),
            "packet_length_max": float(lengths.max()),
            "packet_length_median": float(length_q50),
            "packet_length_p25": float(length_q25),
            "packet_length_p75": float(length_q75),
            "packet_length_p90": float(length_q90),
            "forward_bytes": fwd_bytes,
            "reverse_bytes": rev_bytes,
            "forward_length_mean": fwd_length_mean,
            "forward_length_std": fwd_length_std,
            "reverse_length_mean": rev_length_mean,
            "reverse_length_std": rev_length_std,
            "forward_reverse_byte_ratio": float((fwd_bytes + 1.0) / (rev_bytes + 1.0)),
            "iat_min_seconds": float(iats.min()) if iats.size else float("nan"),
            "iat_max_seconds": float(iats.max()) if iats.size else float("nan"),
            "iat_median_seconds": float(iat_q50),
            "iat_p90_seconds": float(iat_q90),
            "forward_iat_mean_seconds": fwd_iat_mean,
            "forward_iat_std_seconds": fwd_iat_std,
            "reverse_iat_mean_seconds": rev_iat_mean,
            "reverse_iat_std_seconds": rev_iat_std,
            "burst_count": float(len(burst_counts)),
            "burst_packet_count_mean": burst_count_mean,
            "burst_packet_count_std": burst_count_std,
            "burst_packet_count_max": float(burst_counts_array.max()),
            "burst_bytes_mean": burst_bytes_mean,
            "burst_bytes_std": burst_bytes_std,
            "burst_bytes_max": float(burst_bytes_array.max()),
            "burst_duration_mean": burst_duration_mean,
            "burst_duration_std": burst_duration_std,
            "burst_duration_max": float(burst_durations.max()),
            "forward_burst_count": float(sum(direction > 0 for direction in burst_directions)),
            "reverse_burst_count": float(sum(direction < 0 for direction in burst_directions)),
            "inter_burst_iat_mean": inter_burst_mean,
            "inter_burst_iat_std": inter_burst_std,
        }
        if set(values) != set(FEATURE_NAMES):
            raise RuntimeError(f"feature schema mismatch: missing={set(FEATURE_NAMES)-set(values)}, extra={set(values)-set(FEATURE_NAMES)}")
        return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64)
