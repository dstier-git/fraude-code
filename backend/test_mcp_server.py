import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mcp_server
import planning


class FilesystemToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.workspace_patch = patch.object(mcp_server, "WORKSPACE", self.workspace)
        self.workspace_patch.start()

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temp_dir.cleanup()

    def test_reads_utf8_file(self) -> None:
        (self.workspace / "hello.txt").write_text("héllo", encoding="utf-8")

        self.assertEqual(mcp_server.read_file("hello.txt"), "héllo")

    def test_creates_file_and_parent_directories(self) -> None:
        result = mcp_server.write_file("nested/new.txt", "first\nsecond")

        self.assertEqual(result, "Successfully wrote 2 lines to nested/new.txt.")
        self.assertEqual(
            (self.workspace / "nested/new.txt").read_text(encoding="utf-8"),
            "first\nsecond",
        )

    def test_rejects_existing_file(self) -> None:
        target = self.workspace / "existing.txt"
        target.write_text("original", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            mcp_server.write_file("existing.txt", "replacement")

        self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_rejects_paths_outside_workspace(self) -> None:
        with self.assertRaises(ValueError):
            mcp_server.safe_path("../outside.txt")

        with self.assertRaises(ValueError):
            mcp_server.safe_path("/tmp/outside.txt")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_rejects_symlink_escape(self) -> None:
        outside = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: outside.rmdir())
        (self.workspace / "escape").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(ValueError):
            mcp_server.safe_path("escape/file.txt")


class PlanningToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.workspace_patch = patch.object(mcp_server, "WORKSPACE", self.workspace)
        self.workspace_patch.start()

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temp_dir.cleanup()

    def test_plan_identity_and_path_are_stable(self) -> None:
        plan = planning.Plan(title="Example", status="draft", content="Body")

        self.assertEqual(plan.id, plan.id)
        self.assertEqual(plan.path, plan.path)
        self.assertEqual(plan.path, f".fraude/plans/{plan.id}.json")

    def test_create_and_get_plan_round_trip(self) -> None:
        result = mcp_server.create_plan("Example", "Body")
        plan_files = list((self.workspace / ".fraude" / "plans").glob("*.json"))

        self.assertEqual(len(plan_files), 1)
        plan_id = plan_files[0].stem
        self.assertEqual(
            result,
            f"Successfully created plan Example at .fraude/plans/{plan_id}.json.",
        )
        self.assertEqual(mcp_server.get_plan(plan_id), "Body")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_rejects_plan_directory_symlink_escape(self) -> None:
        outside = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: outside.rmdir())
        (self.workspace / ".fraude").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(ValueError):
            mcp_server.create_plan("Example", "Body")


if __name__ == "__main__":
    unittest.main()
