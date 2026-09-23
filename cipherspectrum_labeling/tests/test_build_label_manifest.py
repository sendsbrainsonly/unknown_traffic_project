from pathlib import Path
import unittest

from cipherspectrum_labeling.scripts.build_label_manifest import (
    canonicalize_label,
    capture_group_status,
    derive_label_sets,
    parse_filename,
    stable_flow_id,
)


class FilenameParsingTests(unittest.TestCase):
    def test_parse_capture_and_flow_fields(self) -> None:
        name = (
            "traffic_2024-01-18_adblockplus.org_aes-128_chromium_2."
            "pcap.TCP_10-0-0-2_54321_1-2-3-4_443.pcap"
        )
        result = parse_filename(name)
        self.assertEqual(result["parse_status"], "OK")
        self.assertEqual(
            result["capture_id"],
            "traffic_2024-01-18_adblockplus.org_aes-128_chromium_2",
        )
        self.assertEqual(result["split_group_id"], result["capture_id"])
        self.assertEqual(result["visited_domain"], "adblockplus.org")
        self.assertEqual(result["observed_cipher"], "aes-128")
        self.assertEqual(result["src_ip"], "10.0.0.2")
        self.assertEqual(result["dst_ip"], "1.2.3.4")

    def test_parse_failure_is_explicit(self) -> None:
        result = parse_filename("unexpected.pcap")
        self.assertEqual(result["parse_status"], "FILENAME_PARSE_ERROR")
        self.assertEqual(result["split_group_id"], "")


class LabelPolicyTests(unittest.TestCase):
    def test_anomalous_directory_is_normalized_but_marked(self) -> None:
        label, source, status = canonicalize_label("aes-256-gcm", "chacha20")
        self.assertEqual(label, "getpocket.com")
        self.assertEqual(source, "ANOMALOUS_DIRECTORY_NORMALIZATION")
        self.assertEqual(status, "LOCAL_EXTRA_NORMALIZED_DIRECTORY")

    def test_normal_directory_is_preserved(self) -> None:
        label, source, _status = canonicalize_label("aes-128-gcm", "typekit.net")
        self.assertEqual(label, "typekit.net")
        self.assertEqual(source, "DIRECTORY_NAME")

    def test_group_cipher_mismatch_is_not_hidden(self) -> None:
        self.assertEqual(
            capture_group_status("chacha20-poly1305", "aes-256"),
            "MISPLACED_GROUP",
        )
        self.assertEqual(capture_group_status("mix", "aes-256"), "MIX_EXPECTED")

    def test_derive_official_intersection_and_normalized_union(self) -> None:
        class FakePath:
            def __init__(self, name: str) -> None:
                self.name = name

            def is_dir(self) -> bool:
                return True

        class FakeRoot:
            def __init__(self, names: list[str]) -> None:
                self.names = names

            def iterdir(self):
                return iter(FakePath(name) for name in self.names)

        roots = {
            "aes-128-gcm": FakeRoot(["common", "getpocket.com"]),
            "aes-256-gcm": FakeRoot(["common", "chacha20"]),
            "chacha20-poly1305": FakeRoot(["common", "getpocket.com"]),
            "mix": FakeRoot(["common", "getpocket.com"]),
        }
        official, local = derive_label_sets(roots)  # type: ignore[arg-type]
        self.assertEqual(official, ["common"])
        self.assertEqual(local, ["common", "getpocket.com"])

    def test_flow_id_is_stable_and_path_sensitive(self) -> None:
        first = stable_flow_id("a/b/c.pcap")
        self.assertEqual(first, stable_flow_id("a/b/c.pcap"))
        self.assertNotEqual(first, stable_flow_id("a/b/d.pcap"))


if __name__ == "__main__":
    unittest.main()
