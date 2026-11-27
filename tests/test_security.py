"""Persona name and path validation: the boundary that keeps writes inside
PANMIND_DIR. Every case here is a concrete attack a persona_name argument
from an untrusted MCP caller could carry."""

import pytest

from panmind.security import (
    InvalidInputError,
    PathTraversalError,
    persona_path,
    validate_persona_name,
)


class TestValidatePersonaName:
    @pytest.mark.parametrize(
        "name", ["developer", "dev_user", "coder-1", "A", "a" * 50]
    )
    def test_accepts_whitelisted_names(self, name):
        assert validate_persona_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "../../x",
            "..",
            ".",
            "../etc/passwd",
            "a/b",
            "a\\b",
            "",
            "   ",
            " developer",
            "developer ",
            "a" * 51,
            "dev.user",
            "dev/../user",
            "~root",
            "dev\0user",
        ],
    )
    def test_rejects_traversal_and_unsafe_names(self, name):
        with pytest.raises(InvalidInputError):
            validate_persona_name(name)

    def test_rejects_non_string(self):
        with pytest.raises(InvalidInputError):
            validate_persona_name(None)  # type: ignore[arg-type]


class TestPersonaPath:
    def test_stays_under_base_dir(self, tmp_path):
        base = tmp_path / "personas"
        base.mkdir()
        result = persona_path(base, "developer", ".yaml")
        assert result == (base / "developer.yaml").resolve()
        assert result.is_relative_to(base.resolve())

    def test_supports_multi_part_suffix(self, tmp_path):
        base = tmp_path / "personas"
        base.mkdir()
        result = persona_path(base, "developer", "_user.yaml")
        assert result.name == "developer_user.yaml"

    def test_rejects_traversal_in_persona(self, tmp_path):
        base = tmp_path / "personas"
        base.mkdir()
        with pytest.raises(InvalidInputError):
            persona_path(base, "../../etc/passwd", ".json")

    def test_rejects_traversal_in_suffix(self, tmp_path):
        base = tmp_path / "personas"
        base.mkdir()
        with pytest.raises(InvalidInputError):
            persona_path(base, "developer", "/../../evil.yaml")

    def test_base_dir_need_not_exist_yet(self, tmp_path):
        base = tmp_path / "not-yet-created"
        result = persona_path(base, "developer", ".yaml")
        assert result.is_relative_to(base.resolve())

    def test_symlinked_base_dir_still_contains_result(self, tmp_path):
        real_base = tmp_path / "real"
        real_base.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_base)
        result = persona_path(link, "developer", ".yaml")
        assert result.is_relative_to(real_base.resolve())

    def test_escape_is_unreachable_but_would_raise_path_traversal(self, tmp_path):
        # validate_persona_name already blocks every case that could make
        # persona_path escape base_dir, so this documents the second,
        # independent guarantee (resolve + relative_to) rather than
        # exercising a live bypass.
        base = tmp_path / "personas"
        base.mkdir()
        with pytest.raises((InvalidInputError, PathTraversalError)):
            persona_path(base, "..", ".yaml")


class TestReadImportSource:
    """The file is read through descriptors opened without following links,
    so a path that changes between the containment check and the read is
    refused rather than followed."""

    @pytest.fixture
    def layout(self, tmp_path):
        root, outside = tmp_path / "root", tmp_path / "outside"
        (root / "sub").mkdir(parents=True)
        outside.mkdir()
        (root / "sub" / "f.md").write_text("inside")
        (outside / "f.md").write_text("outside")
        return root, outside

    def test_reads_name_and_text(self, layout):
        from panmind.security import read_import_source

        root, _outside = layout
        assert read_import_source("sub/f.md", root) == ("f.md", "inside")

    def test_component_swapped_for_a_link_after_the_check_is_refused(
            self, layout, monkeypatch):
        import os
        import shutil

        from panmind.security import read_import_source

        root, outside = layout
        real_open = os.open

        def racing_open(path, flags, *args, **kwargs):
            if path == "sub":
                shutil.rmtree(root / "sub")
                (root / "sub").symlink_to(outside)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", racing_open)
        with pytest.raises(PathTraversalError):
            read_import_source("sub/f.md", root)

    def test_fifo_is_refused_without_blocking(self, layout):
        import os

        from panmind.security import read_import_source

        root, _outside = layout
        os.mkfifo(root / "pipe.md")
        with pytest.raises(InvalidInputError, match="regular file"):
            read_import_source("pipe.md", root)
