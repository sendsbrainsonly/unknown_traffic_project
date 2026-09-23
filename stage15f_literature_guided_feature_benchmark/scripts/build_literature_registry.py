#!/usr/bin/env python3
"""Convert the reviewed primary-source notes into the Stage 15F registry CSV."""

from __future__ import annotations

import re
from pathlib import Path

from common import ROOT, sha256_file, write_csv, write_json


FIELDS = (
    "raw input",
    "granularity",
    "packet window",
    "bytes/packet",
    "header / payload",
    "boundary",
    "direction",
    "IAT",
    "length",
    "statistics",
    "burst",
    "pretraining",
    "encoder",
    "objective",
    "tasks / datasets",
    "leakage risks",
)


def extract_link(line: str) -> str:
    return ";".join(re.findall(r"<([^>]+)>", line))


def main() -> None:
    source = ROOT / "literature_primary_source_notes.md"
    text = source.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE))
    rows = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else text.index("## Cross-paper", start)
        section = text[start:end]
        method = match.group(2).strip()
        title_match = re.search(r"^- Accurate title: \*\*(.+?)\*\*$", section, flags=re.MULTILINE)
        year_match = re.search(r"^- Year / venue: (.+)$", section, flags=re.MULTILINE)
        paper_match = re.search(r"^- Paper: (.+)$", section, flags=re.MULTILINE)
        code_match = re.search(r"^- Official code(?:/project page)?: (.+)$", section, flags=re.MULTILINE)
        table = {}
        for field_name, finding in re.findall(r"^\| ([^|]+?) \| (.+?) \|$", section, flags=re.MULTILINE):
            if field_name in {"Field", "---"}:
                continue
            table[field_name.strip()] = finding.strip()
        missing = [name for name in FIELDS if name not in table]
        if missing:
            raise RuntimeError(f"{method}: missing registry fields {missing}")
        difference_match = re.search(r"^Paper/code difference: (.+?)(?=\n\n|\Z)", section, flags=re.MULTILINE | re.DOTALL)
        code_line = code_match.group(1).strip() if code_match else "UNKNOWN No author-maintained official implementation verified."
        row = {
            "method": method,
            "full_title": title_match.group(1) if title_match else "UNKNOWN",
            "year_venue": year_match.group(1).strip() if year_match else "UNKNOWN",
            "paper_link": extract_link(paper_match.group(1)) if paper_match else "UNKNOWN",
            "official_code_link": extract_link(code_line) or "UNKNOWN",
            "official_code_status": "UNKNOWN" if "UNKNOWN" in code_line or "not verified" in code_line.lower() else ("PLACEHOLDER_ONLY" if "coming soon" in section.lower() else "VERIFIED_AUTHOR_MAINTAINED"),
            "raw_input": table["raw input"],
            "granularity": table["granularity"],
            "packet_window": table["packet window"],
            "bytes_per_packet": table["bytes/packet"],
            "header_payload": table["header / payload"],
            "boundary": table["boundary"],
            "direction": table["direction"],
            "iat": table["IAT"],
            "packet_length": table["length"],
            "flow_statistics": table["statistics"],
            "burst": table["burst"],
            "pretraining": table["pretraining"],
            "encoder": table["encoder"],
            "pretraining_or_training_objective": table["objective"],
            "tasks_datasets": table["tasks / datasets"],
            "data_leakage_risks": table["leakage risks"],
            "paper_code_difference": " ".join(difference_match.group(1).split()) if difference_match else "UNKNOWN",
            "evidence_policy": "PAPER_CONFIRMED/CODE_CONFIRMED/INFERRED/UNKNOWN tags are embedded per field",
        }
        rows.append(row)
    if len(rows) != 19:
        raise RuntimeError(f"expected 19 works, found {len(rows)}")
    output = ROOT / "literature_feature_registry.csv"
    write_csv(output, rows, list(rows[0]))
    write_json(
        ROOT / "literature_registry_audit.json",
        {
            "status": "PASS",
            "work_count": len(rows),
            "required_field_count": len(FIELDS),
            "source_notes_sha256": sha256_file(source),
            "registry_sha256": sha256_file(output),
            "unknown_official_code": [row["method"] for row in rows if row["official_code_status"] == "UNKNOWN"],
            "placeholder_official_code": [row["method"] for row in rows if row["official_code_status"] == "PLACEHOLDER_ONLY"],
        },
    )
    print(f"works={len(rows)} registry={output}")


if __name__ == "__main__":
    main()
