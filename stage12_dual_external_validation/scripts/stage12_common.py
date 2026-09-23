#!/usr/bin/env python3
"""Shared frozen definitions for Stage 12 external validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


STAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_ROOT.parent
CONFIG_PATH = STAGE_ROOT / "configs" / "stage12_config.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames are required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class CaptureLabel:
    canonical_class: str
    application_label: str
    official_category: str
    official_joint_label: str
    domain_state: str
    label_source: str


def _vpn_service(name: str) -> str:
    if "bittorrent" in name or "torrent" in name:
        return "P2P"
    if name.startswith("email"):
        return "Email"
    if any(term in name for term in ("ftps", "sftp", "scp", "skype_file")):
        return "File-Transfer"
    if any(term in name for term in ("netflix", "spotify", "vimeo", "youtube")):
        return "Streaming"
    if "audio" in name or "voipbuster" in name or "video" in name:
        return "VoIP"
    if "chat" in name:
        return "Chat"
    raise ValueError(f"No official ISCX-VPN traffic category mapping for {name}")


def classify_iscx_vpn(path: Path, pcap_root: Path) -> CaptureLabel:
    stem = path.stem.lower()
    domain = "VPN" if stem.startswith("vpn_") else "NonVPN"
    name = stem[4:] if stem.startswith("vpn_") else stem
    app_rules = (
        ("voipbuster", "VoIPBuster"),
        ("bittorrent", "BitTorrent"),
        ("facebook", "Facebook"),
        ("hangout", "Hangouts"),
        ("netflix", "Netflix"),
        ("spotify", "Spotify"),
        ("youtube", "YouTube"),
        ("vimeo", "Vimeo"),
        ("gmail", "Gmail"),
        ("skype", "Skype"),
        ("email", "Email"),
        ("ftps", "FTPS"),
        ("sftp", "SFTP"),
        ("scp", "SCP"),
        ("aim", "AIM"),
        ("icq", "ICQ"),
    )
    application = next((label for token, label in app_rules if name.startswith(token)), None)
    if application is None:
        raise ValueError(f"No unambiguous ISCX-VPN application mapping for {path.name}")
    service = _vpn_service(name)
    return CaptureLabel(
        canonical_class=application,
        application_label=application,
        official_category=service,
        official_joint_label=f"{domain}-{service}",
        domain_state=domain,
        label_source="official capture filename + UNB application/category description",
    )


def _tor_category(path: Path, pcap_root: Path) -> tuple[str, str]:
    rel = path.relative_to(pcap_root).as_posix()
    lower = path.name.lower()
    is_tor = rel.startswith("Tor/")
    if is_tor:
        prefix_rules = {
            "audio_": "Audio",
            "browsing_": "Browsing",
            "chat_": "Chat",
            "file-transfer_": "File-Transfer",
            "mail_": "Email",
            "p2p_": "P2P",
            "video_": "Video",
            "voip_": "VoIP",
        }
        category = next((value for prefix, value in prefix_rules.items() if lower.startswith(prefix)), None)
        if category is None:
            if lower.startswith("tor_spotify"):
                category = "Audio"
            elif lower.startswith(("torfacebook", "torgoogle", "tortwitter")):
                category = "Browsing"
            elif lower.startswith(("torrent", "tor_p2p")):
                category = "P2P"
            elif lower.startswith(("torvimeo", "toryoutube")):
                category = "Video"
        if category is None:
            raise ValueError(f"No official Tor category mapping for {rel}")
        return "Tor", category
    if any(token in lower for token in ("email_imap", "pop_filetransfer", "thunderbird")):
        category = "Email"
    elif any(token in lower for token in ("ftp_filetransfer", "sftp_filetransfer", "skype_transfer")):
        category = "File-Transfer"
    elif "spotify" in lower:
        category = "Audio"
    elif any(token in lower for token in ("browsing", "ssl")):
        category = "Browsing"
    elif "chat" in lower:
        category = "Chat"
    elif any(token in lower for token in ("p2p", "torrent")):
        category = "P2P"
    elif any(token in lower for token in ("vimeo", "youtube")):
        category = "Video"
    elif any(token in lower for token in ("audio", "voice")):
        category = "VoIP"
    else:
        raise ValueError(f"No official NonTor category mapping for {rel}")
    return "NonTor", category


def classify_iscx_tor(path: Path, pcap_root: Path) -> CaptureLabel:
    domain, category = _tor_category(path, pcap_root)
    lower = path.stem.lower()
    app_rules = (
        ("spotify", "Spotify"),
        ("facebook", "Facebook"),
        ("hangout", "Hangouts"),
        ("youtube", "YouTube"),
        ("vimeo", "Vimeo"),
        ("skype", "Skype"),
        ("thunderbird", "Email"),
        ("email", "Email"),
        ("pop_", "Email"),
        ("sftp", "SFTP"),
        ("ftp", "FTP"),
        ("torrent", "P2P"),
        ("p2p", "P2P"),
        ("vuze", "P2P"),
        ("multiplespeed", "P2P"),
        ("browsing", "Browsing"),
        ("ssl", "Browsing"),
        ("google", "Google"),
        ("twitter", "Twitter"),
        ("aim", "AIM"),
        ("icq", "ICQ"),
    )
    application = next((label for token, label in app_rules if token in lower), None)
    if application is None:
        raise ValueError(f"No unambiguous ISCXTor2016 application mapping for {path.name}")
    return CaptureLabel(
        canonical_class=application,
        application_label=application,
        official_category=category,
        official_joint_label=f"{domain}-{category}",
        domain_state=domain,
        label_source="official capture filename + Scenario-B category taxonomy",
    )


def classify_capture(dataset: str, path: Path, pcap_root: Path) -> CaptureLabel:
    if dataset == "iscx_vpn":
        return classify_iscx_vpn(path, pcap_root)
    if dataset == "iscx_tor":
        return classify_iscx_tor(path, pcap_root)
    raise ValueError(dataset)


def discover_pcaps(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}
    )


def openness_counts(class_count: int) -> dict[str, int]:
    low = max(1, math.ceil(0.05 * class_count))
    medium = max(low + 1, math.ceil(0.15 * class_count))
    high = max(medium + 1, math.ceil(0.25 * class_count))
    proposed = {"low": low, "medium": medium, "high": high}
    return {name: count for name, count in proposed.items() if class_count - count >= 4}


def frozen_unknown_protocol(classes: Iterable[str], seed: int = 42) -> dict:
    ordered = sorted(set(classes))
    counts = openness_counts(len(ordered))
    if len(counts) < 2:
        raise RuntimeError(
            f"SETTING_REDUCED_DUE_TO_CLASS_COUNT cannot retain at least two settings: K={len(ordered)}"
        )
    permutation = np.random.default_rng(seed).permutation(ordered).tolist()
    settings = {
        name: {
            "unknown_count": count,
            "unknown_classes": permutation[:count],
            "known_classes": sorted(set(ordered) - set(permutation[:count])),
        }
        for name, count in counts.items()
    }
    return {
        "protocol_seed": seed,
        "eligible_classes_canonical_sorted": ordered,
        "permutation": permutation,
        "settings": settings,
        "nested": all(
            set(settings[right]["unknown_classes"]).issuperset(settings[left]["unknown_classes"])
            for left, right in zip(settings, list(settings)[1:])
        ),
        "setting_reduced_due_to_class_count": len(settings) < 3,
    }


def quantile_higher(values: np.ndarray, q: float = 0.95) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def array_digest(data: np.ndarray, target: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (np.ascontiguousarray(data), np.ascontiguousarray(target)):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def git_identity(path: Path) -> dict[str, str]:
    import subprocess

    def call(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"

    return {"commit": call("rev-parse", "HEAD"), "status": call("status", "--short")}


def load_config() -> dict:
    return read_json(CONFIG_PATH)

