"""Regression tests for the parsed local-corpus provisions index."""

import json

from axiom_encode.harness import validator_pipeline as vp


def _write_provisions(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_records_parse_once_per_file_and_preserve_order(tmp_path, monkeypatch):
    provisions = tmp_path / "sample.jsonl"
    _write_provisions(
        provisions,
        [
            {"citation_path": "us/statute/x/1", "body": "first"},
            {"citation_path": "us/statute/x/2", "body": "other"},
            {"citation_path": "us/statute/x/1", "body": "second"},
        ],
    )

    parse_counts = {"n": 0}
    original_read_text = type(provisions).read_text

    def counting_read_text(self, *args, **kwargs):
        if self == provisions:
            parse_counts["n"] += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(provisions), "read_text", counting_read_text)
    vp._LOCAL_CORPUS_PROVISION_INDEX.clear()

    first = vp._read_local_corpus_provision_records(provisions, "us/statute/x/1")
    second = vp._read_local_corpus_provision_records(provisions, "us/statute/x/2")
    repeat = vp._read_local_corpus_provision_records(provisions, "us/statute/x/1")

    assert [record["body"] for record in first] == ["first", "second"]
    assert [record["body"] for record in second] == ["other"]
    assert repeat == first
    assert parse_counts["n"] == 1


def test_index_refreshes_when_file_identity_changes(tmp_path):
    provisions = tmp_path / "sample.jsonl"
    _write_provisions(provisions, [{"citation_path": "us/statute/x/1", "body": "old"}])
    vp._LOCAL_CORPUS_PROVISION_INDEX.clear()

    before = vp._read_local_corpus_provision_records(provisions, "us/statute/x/1")
    assert [record["body"] for record in before] == ["old"]

    _write_provisions(
        provisions,
        [{"citation_path": "us/statute/x/1", "body": "replaced-with-longer-body"}],
    )
    after = vp._read_local_corpus_provision_records(provisions, "us/statute/x/1")
    assert [record["body"] for record in after] == ["replaced-with-longer-body"]


def test_missing_file_returns_empty(tmp_path):
    vp._LOCAL_CORPUS_PROVISION_INDEX.clear()
    absent = tmp_path / "absent.jsonl"
    assert vp._read_local_corpus_provision_records(absent, "us/statute/x/1") == []


def test_supabase_fetch_stays_memoized():
    assert hasattr(vp._fetch_supabase_corpus_source_text, "cache_clear")
