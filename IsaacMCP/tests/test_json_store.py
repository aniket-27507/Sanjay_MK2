"""Tests for JSON store."""

from __future__ import annotations

import pytest

from isaac_mcp.storage.json_store import JsonStore


@pytest.fixture
def store(tmp_path):
    return JsonStore(str(tmp_path / "test.json"))


def test_load_missing_file(store):
    data = store.load()
    assert data == {}


def test_save_and_load(store):
    store.save({"key": "value", "num": 42})
    data = store.load()
    assert data["key"] == "value"
    assert data["num"] == 42


def test_append_entry(store):
    store.append_entry("items", {"name": "first"})
    store.append_entry("items", {"name": "second"})

    data = store.load()
    assert len(data["items"]) == 2
    assert data["items"][0]["name"] == "first"
    assert data["items"][1]["name"] == "second"


def test_append_to_nonexistent_key(store):
    store.save({"other": "data"})
    store.append_entry("new_list", {"a": 1})
    data = store.load()
    assert data["new_list"] == [{"a": 1}]
    assert data["other"] == "data"


def test_exists(store):
    assert not store.exists()
    store.save({})
    assert store.exists()


def test_overwrite_on_save(store):
    store.save({"v": 1})
    store.save({"v": 2})
    assert store.load()["v"] == 2


def test_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "test.json"
    s = JsonStore(str(nested))
    s.save({"nested": True})
    assert s.load()["nested"] is True
