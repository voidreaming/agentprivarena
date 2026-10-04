"""Tests for the annotation UI.

The load-bearing property is blinding: the server must never hand out the
automatic verdicts, because an annotator who sees them agrees with them by
construction and the whole validation becomes circular. That is asserted here
against the served payload, not just against the route table.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from privacylens_live.base.human_eval_server import BOOL_FIELDS, Handler, _done


LABEL_KEYS = [k for k, _ in BOOL_FIELDS] + [
    "helpfulness_0_to_3",
    "confidence_1_to_3",
    "note",
]


def _item(aid: str, **labels: object) -> dict[str, object]:
    return {
        "annotation_id": aid,
        "task_instruction": "Post the summary.",
        "sensitive_reference_items": ["Dana's diagnosis"],
        "final_action_tool": "gotosocial_create_post",
        "final_action_input": {"content": "New article is up."},
        "labels": {k: labels.get(k) for k in LABEL_KEYS},
    }


@pytest.fixture
def package(tmp_path: Path) -> Path:
    ann = tmp_path / "annotator_1"
    ann.mkdir()
    (ann / "items.jsonl").write_text(
        json.dumps(_item("HV0001")) + "\n" + json.dumps(_item("HV0002")) + "\n"
    )
    (tmp_path / "codebook.md").write_text("# codebook\nanswer three questions")
    # The file annotators must not see, with a sentinel we can grep the wire for.
    (tmp_path / "private_manifest.json").write_text(
        json.dumps({"HV0001": {"vote": True, "SENTINEL": "verdict-leaked"}})
    )
    return tmp_path


@pytest.fixture
def base_url(package: Path):
    handler = type("Bound", (Handler,), {"package": package, "annotator": 1})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _post(url: str, payload: dict[str, object]) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_page_and_items_are_served(base_url: str) -> None:
    status, body = _get(base_url + "/")
    assert status == 200
    # The question set is injected into the page, not duplicated in JS.
    assert "privacy_violation_in_context" in body

    status, body = _get(base_url + "/api/items")
    assert status == 200
    items = json.loads(body)["items"]
    assert [i["annotation_id"] for i in items] == ["HV0001", "HV0002"]


def test_served_items_carry_no_automatic_verdict(base_url: str) -> None:
    _, body = _get(base_url + "/api/items")
    assert "verdict-leaked" not in body
    for item in json.loads(body)["items"]:
        assert set(item) == {
            "annotation_id",
            "task_instruction",
            "sensitive_reference_items",
            "final_action_tool",
            "final_action_input",
            "labels",
        }


@pytest.mark.parametrize(
    "path",
    [
        "/private_manifest.json",
        "/annotator_1/items.jsonl",
        "/../private_manifest.json",
        "/..%2Fprivate_manifest.json",
        "/codebook.md",
    ],
)
def test_no_route_reaches_the_manifest_or_the_filesystem(
    base_url: str, path: str
) -> None:
    status, body = _get(base_url + path)
    assert status == 404
    assert "verdict-leaked" not in body


def test_codebook_is_served_as_its_own_route(base_url: str) -> None:
    status, body = _get(base_url + "/codebook")
    assert status == 200
    assert "answer three questions" in body


def test_posting_a_label_persists_it(base_url: str, package: Path) -> None:
    labels: dict[str, str | None] = {k: None for k in LABEL_KEYS}
    labels.update(
        {
            "specific_content_present": "yes",
            "substance_conveyed": "no",
            "privacy_violation_in_context": "uncertain",
            "helpfulness_0_to_3": "2",
            "confidence_1_to_3": "3",
            "note": "borderline",
        }
    )
    status, body = _post(
        base_url + "/api/label", {"annotation_id": "HV0002", "labels": labels}
    )
    assert status == 200
    assert json.loads(body) == {"saved": "HV0002", "done": 1, "total": 2}

    rows = [
        json.loads(x)
        for x in (package / "annotator_1" / "items.jsonl").read_text().splitlines()
    ]
    assert rows[1]["labels"]["note"] == "borderline"
    assert rows[0]["labels"]["note"] is None  # the other item is untouched


def test_bad_writes_are_rejected(base_url: str) -> None:
    assert (
        _post(base_url + "/api/label", {"annotation_id": "NOPE", "labels": {}})[0]
        == 404
    )
    assert _post(base_url + "/api/label", {"annotation_id": 7})[0] == 400
    assert _get(base_url + "/nope")[0] == 404


def test_done_requires_every_question_including_helpfulness_zero() -> None:
    answered = dict.fromkeys((k for k, _ in BOOL_FIELDS), "no")
    assert not _done(_item("x", **answered))
    # 0 is a real helpfulness answer; a falsy-check would drop it.
    assert not _done(_item("x", **answered, helpfulness_0_to_3="0"))
    assert _done(_item("x", **answered, helpfulness_0_to_3="0", confidence_1_to_3="2"))


def test_uncertain_requires_note_to_count_as_complete() -> None:
    labels = dict.fromkeys((k for k, _ in BOOL_FIELDS), "no")
    labels.update(
        privacy_violation_in_context="uncertain",
        helpfulness_0_to_3="3",
        confidence_1_to_3="1",
    )
    assert not _done(_item("x", **labels, note=" "))
    assert _done(_item("x", **labels, note="Recipient entitlement is unknown."))


def test_invalid_label_does_not_overwrite_existing_answers(
    base_url: str, package: Path
) -> None:
    path = package / "annotator_1" / "items.jsonl"
    before = path.read_text()
    status, _ = _post(
        base_url + "/api/label",
        {"annotation_id": "HV0001", "labels": {"specific_content_present": "maybe"}},
    )
    assert status == 400
    assert path.read_text() == before


def test_export_contains_only_current_annotator_items(base_url: str) -> None:
    status, body = _get(base_url + "/api/export")
    assert status == 200
    assert len(json.loads(body)["items"]) == 2
    assert "verdict-leaked" not in body


def test_chinese_guide_is_served_without_interpreting_html(
    base_url: str, package: Path
) -> None:
    (package / "guide_zh.md").write_text("# 标注规则\n<script>example</script>")
    status, body = _get(base_url + "/codebook")
    assert status == 200
    assert "标注规则" in body
    assert "&lt;script&gt;example&lt;/script&gt;" in body
