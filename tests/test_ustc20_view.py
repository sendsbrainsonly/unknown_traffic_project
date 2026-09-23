from pathlib import Path

from src.preprocessing.flow_split import discover_classes
from src.preprocessing.ustc20_view import EXPECTED_CLASSES, build_ustc20_view


def _make_source(root: Path) -> None:
    for split, classes in EXPECTED_CLASSES.items():
        for class_name in sorted(classes):
            if class_name == "SMB":
                for shard in ("SMB-1.pcap", "SMB-2.pcap"):
                    path = root / split / shard
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"pcap")
            elif class_name == "Weibo":
                for shard in ("Weibo-1.pcap", "Weibo-2.pcap"):
                    path = root / split / "Weibo" / shard
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"pcap")
            else:
                path = root / split / f"{class_name}.pcap"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"pcap")


def test_build_view_merges_smb_without_touching_source(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "view"
    _make_source(source)

    manifest = build_ustc20_view(source, output)
    discovered = discover_classes(str(output))
    names = {(split, class_name) for split, class_name, _ in discovered}

    assert manifest["class_count"] == 20
    assert manifest["source_pcap_count"] == 22
    assert len(discovered) == 20
    assert ("Benign", "SMB") in names
    assert ("Benign", "SMB-1") not in names
    smb = next(pcaps for split, name, pcaps in discovered if name == "SMB")
    assert [stem for stem, _ in smb] == ["SMB-1", "SMB-2"]
    assert all(Path(path).is_symlink() for _, path in smb)
    assert (source / "Benign" / "SMB-1.pcap").read_bytes() == b"pcap"
