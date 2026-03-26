"""Tests for JSONL parser."""

import json
from pathlib import Path

import pytest

from moderapi.parser import parse_jsonl, DETOXIFY_TO_PERSPECTIVE, PERSPECTIVE_TO_DETOXIFY


def test_parse_valid_jsonl(sample_jsonl: Path):
    records = list(parse_jsonl(sample_jsonl))
    assert len(records) == 10
    assert records[0].text == "Sample text number 0"
    assert "TOXICITY" in records[0].perspective_scores


def test_parse_empty_file(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    records = list(parse_jsonl(p))
    assert records == []


def test_parse_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n{invalid\n")
    records = list(parse_jsonl(p))
    assert records == []


def test_parse_missing_text(tmp_path: Path):
    p = tmp_path / "notext.jsonl"
    p.write_text(json.dumps({"scores": {"TOXICITY": 0.5}}) + "\n")
    records = list(parse_jsonl(p))
    assert records == []


def test_detoxify_mapping():
    assert len(DETOXIFY_TO_PERSPECTIVE) == 6
    assert DETOXIFY_TO_PERSPECTIVE["obscene"] == "PROFANITY"
    assert PERSPECTIVE_TO_DETOXIFY["PROFANITY"] == "obscene"
