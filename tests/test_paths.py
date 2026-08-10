"""Data-root resolution must be portable between local disk and Colab."""

from pathlib import Path

from src.data.paths import (
    PANTS_DATA_ROOT_ENV,
    PROJECT_ROOT,
    get_case_paths,
    get_data_root,
    get_relative_case_paths,
)


def test_explicit_argument_wins_over_environment(monkeypatch):
    monkeypatch.setenv(PANTS_DATA_ROOT_ENV, "/from/environment")
    assert get_data_root("/from/argument") == Path("/from/argument")


def test_environment_variable_used_when_no_argument(monkeypatch):
    monkeypatch.setenv(PANTS_DATA_ROOT_ENV, "/content/drive/MyDrive/PanTS/data")
    assert get_data_root() == Path("/content/drive/MyDrive/PanTS/data")


def test_falls_back_to_project_root(monkeypatch):
    monkeypatch.delenv(PANTS_DATA_ROOT_ENV, raising=False)
    assert get_data_root() == PROJECT_ROOT / "PanTS" / "data"


def test_environment_is_reread_after_import(monkeypatch):
    """
    The Colab requirement: a process that sets PANTS_DATA_ROOT *after*
    importing this module must still pick up the new value. Resolving the
    environment once at import time would silently break notebooks.
    """
    monkeypatch.setenv(PANTS_DATA_ROOT_ENV, "/first/root")
    assert get_data_root() == Path("/first/root")

    monkeypatch.setenv(PANTS_DATA_ROOT_ENV, "/second/root")
    assert get_data_root() == Path("/second/root")


def test_case_paths_follow_the_resolved_root(monkeypatch):
    monkeypatch.setenv(PANTS_DATA_ROOT_ENV, "/content/drive/MyDrive/PanTS/data")
    paths = get_case_paths("PanTS_00000123")

    assert paths["ct"] == Path("/content/drive/MyDrive/PanTS/data/ImageTr/PanTS_00000123/ct.nii.gz")
    assert paths["combined"] == Path(
        "/content/drive/MyDrive/PanTS/data/LabelTr/PanTS_00000123/combined_labels.nii.gz"
    )


def test_test_split_maps_to_te_directories(monkeypatch):
    monkeypatch.setenv(PANTS_DATA_ROOT_ENV, "/data")
    paths = get_case_paths("PanTS_00009001", "test")
    assert paths["ct"] == Path("/data/ImageTe/PanTS_00009001/ct.nii.gz")


def test_relative_paths_contain_no_machine_specific_prefix():
    relative = get_relative_case_paths("PanTS_00000123")
    assert relative["ct"] == "ImageTr/PanTS_00000123/ct.nii.gz"
    assert relative["label"] == "LabelTr/PanTS_00000123/combined_labels.nii.gz"
    for value in relative.values():
        assert not value.startswith("/")
