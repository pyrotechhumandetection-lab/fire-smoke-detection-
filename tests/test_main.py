"""Fast checks of the alert rules and credential handling (no model, no camera).

    python -m unittest discover -s tests -v
"""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402

FIRE = main.Detection("fire", 0.87, (100, 100, 300, 300))
SMOKE = main.Detection("smoke", 0.66, (50, 20, 400, 200))


class AlertRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.snapshots = Path(self.tmp.name) / "snapshots"  # does not exist yet: must be created
        self.alerts = main.AlertManager({"consecutive_frames": 3, "cooldown_s": 10, "jpeg_quality": 90},
                                        self.snapshots)
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.t0 = datetime(2026, 9, 19, 10, 32, 15)
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def feed(self, detections, second):
        with redirect_stdout(self.out):
            return self.alerts.update(detections, self.frame, now=float(second),
                                      when=self.t0 + timedelta(seconds=second))

    def test_needs_three_consecutive_frames(self):
        self.assertEqual(self.feed([FIRE], 0), [])
        self.assertEqual(self.feed([FIRE], 1), [])
        fired = self.feed([FIRE], 2)
        self.assertEqual([f[0] for f in fired], ["fire"])
        self.assertEqual(fired[0][2].name, "fire_20260919_103217.jpg")
        self.assertTrue(fired[0][2].is_file())
        self.assertIn("[ALERT] FIRE DETECTED at 2026-09-19 10:32:17 (conf 0.87) -> saved ", self.out.getvalue())

    def test_a_miss_resets_the_count(self):
        self.feed([FIRE], 0)
        self.feed([FIRE], 1)
        self.feed([], 2)  # gap
        self.assertEqual(self.feed([FIRE], 3), [])
        self.assertEqual(self.feed([FIRE], 4), [])
        self.assertEqual(len(self.feed([FIRE], 5)), 1)

    def test_cooldown_blocks_repeats(self):
        for s in range(3):
            self.feed([FIRE], s)                               # alert at t=2
        for s in range(3, 12):
            self.assertEqual(self.feed([FIRE], s), [], s)      # t=3..11 < 10 s later
        self.assertEqual(len(self.feed([FIRE], 12)), 1)        # t=12: cooldown over
        self.assertEqual(self.alerts.counts["fire"], 2)
        self.assertEqual(len(list(self.snapshots.glob("fire_*.jpg"))), 2)

    def test_fire_and_smoke_are_independent(self):
        self.feed([FIRE, SMOKE], 0)
        self.feed([FIRE], 1)
        self.assertEqual([f[0] for f in self.feed([FIRE, SMOKE], 2)], ["fire"])  # smoke streak was broken
        self.feed([SMOKE], 3)
        self.assertEqual([f[0] for f in self.feed([SMOKE], 4)], ["smoke"])
        self.assertEqual(self.alerts.active, frozenset({"smoke"}))

    def test_reset_streaks(self):
        self.feed([FIRE], 0)
        self.feed([FIRE], 1)
        self.alerts.reset_streaks()  # e.g. stream reconnected
        self.assertEqual(self.feed([FIRE], 2), [])

    def test_same_second_does_not_overwrite(self):
        self.alerts.cooldown = 0
        for _ in range(3):
            self.feed([FIRE], 0)
        self.feed([FIRE], 0)
        names = sorted(p.name for p in self.snapshots.glob("*.jpg"))
        self.assertEqual(names, ["fire_20260919_103215.jpg", "fire_20260919_103215_1.jpg"])


class Credentials(unittest.TestCase):
    def test_special_characters_are_encoded(self):
        url = main.add_credentials("rtsp://192.168.1.64:554/Streaming/Channels/102", "admin", "p@ss:w#rd/1%")
        parts = urlsplit(url)
        self.assertEqual(parts.hostname, "192.168.1.64")
        self.assertEqual(parts.port, 554)
        self.assertEqual(parts.path, "/Streaming/Channels/102")
        self.assertEqual(unquote(parts.password), "p@ss:w#rd/1%")

    def test_query_string_kept(self):
        url = main.add_credentials("rtsp://10.0.0.5/cam/realmonitor?channel=1&subtype=1", "user", "pw")
        self.assertEqual(url, "rtsp://user:pw@10.0.0.5/cam/realmonitor?channel=1&subtype=1")

    def test_existing_credentials_and_files_untouched(self):
        self.assertEqual(main.add_credentials("rtsp://a:b@host/x", "admin", "pw"), "rtsp://a:b@host/x")
        self.assertEqual(main.add_credentials(r"D:\videos\fire.mp4", "admin", "pw"), r"D:\videos\fire.mp4")
        self.assertEqual(main.add_credentials("rtsp://host/x", "", ""), "rtsp://host/x")

    def test_password_masked_in_logs(self):
        url = main.add_credentials("rtsp://192.168.1.64:554/live", "admin", "S3cret!")
        self.assertEqual(main.mask_url(url), "rtsp://admin:****@192.168.1.64:554/live")
        self.assertNotIn("S3cret", main.mask_url(url))


class ClassNames(unittest.TestCase):
    def test_canonical_labels(self):
        self.assertEqual(main.canonical_label("Fire"), "fire")
        self.assertEqual(main.canonical_label("smoke"), "smoke")
        self.assertEqual(main.canonical_label("Smoke"), "smoke")
        self.assertIsNone(main.canonical_label("default"))


if __name__ == "__main__":
    unittest.main()
