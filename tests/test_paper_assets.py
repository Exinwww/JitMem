"""Verify original-asset integrity without redistributing third-party prompt text."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import FrozenInstanceError

import pytest

from jitmem import paper_assets as paper


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive(items: list[tuple[str, bytes, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as target:
        for name, body, kind in items:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(body)
                target.addfile(member, io.BytesIO(body))
            else:
                member.linkname = "elsewhere"
                target.addfile(member)
    return stream.getvalue()


@pytest.fixture
def synthetic_cache(tmp_path, monkeypatch):
    # Deliberately synthetic content; no author template wording is included.
    templates = {
        "curator.txt": b"<System Prompt>\nfixture system\n\n<User Prompt>\nfixture input\n",
        "executor.txt": b"fixture executor\n",
        "distillation.txt": b"fixture distillation\n",
        "judge_system.txt": b"fixture judge system",
        "judge_user.txt": b"fixture judge input",
    }
    monkeypatch.setattr(
        paper, "EXPECTED_TEMPLATE_HASHES", {name: digest(data) for name, data in templates.items()}
    )

    def forbid_network(*args, **kwargs):
        raise AssertionError("Asset loading/offline preparation must not fetch a source")

    monkeypatch.setattr(paper, "_download", forbid_network)
    destination = tmp_path / "assets"
    destination.mkdir()
    for name, data in templates.items():
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_text(json.dumps(paper.asset_provenance()))
    return destination, templates


@pytest.mark.parametrize("filename", list(paper.EXPECTED_TEMPLATE_HASHES))
def test_loader_rejects_every_changed_template(synthetic_cache, filename):
    directory, _ = synthetic_cache
    target = directory / filename
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(paper.PaperAssetError, match="hash does not match"):
        paper.load_assets(directory)


@pytest.mark.parametrize("filename", [*paper.EXPECTED_TEMPLATE_HASHES, "manifest.json"])
def test_loader_rejects_every_missing_asset_without_fetching(synthetic_cache, filename):
    directory, _ = synthetic_cache
    (directory / filename).unlink()
    with pytest.raises(paper.PaperAssetError, match="cannot be read"):
        paper.load_assets(directory)


@pytest.mark.parametrize("change", ["source", "profile", "invalid_json"])
def test_loader_binds_assets_to_complete_source_manifest(synthetic_cache, change):
    directory, _ = synthetic_cache
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "source":
        manifest["sources"]["jitmem"]["archive_sha256"] = "changed"
    elif change == "profile":
        manifest["profile"] = "changed"
    path.write_text("{" if change == "invalid_json" else json.dumps(manifest))
    with pytest.raises(
        paper.PaperAssetError, match="manifest is invalid|provenance does not match"
    ):
        paper.load_assets(directory)


def test_loaded_asset_text_and_metadata_are_immutable(synthetic_cache):
    directory, _ = synthetic_cache
    assets = paper.load_assets(directory)
    assert assets.curator_system == "fixture system"
    assert assets.curator_user == "fixture input"
    with pytest.raises(FrozenInstanceError):
        assets.executor = "changed"
    metadata = assets.provenance
    metadata["templates"].clear()
    assert len(assets.metadata["templates"]) == 5
    assert "fixture executor" not in repr(assets)


def test_offline_preparation_rejects_missing_or_changed_source_without_fetch(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Offline preparation must never fall back to the network")

    monkeypatch.setattr(paper, "_download", forbid_network)
    sources = tmp_path / "sources"
    sources.mkdir()
    output = tmp_path / "prepared"
    with pytest.raises(paper.PaperAssetError, match="cannot be read"):
        paper.prepare_assets(output, source_dir=sources)
    (sources / "jitmem_source.bin").write_bytes(b"modified source")
    with pytest.raises(paper.PaperAssetError, match="hash does not match"):
        paper.prepare_assets(output, source_dir=sources)
    assert not output.exists()


def test_preparation_requires_matching_archive_member_and_template_hashes(
    tmp_path, monkeypatch, synthetic_cache
):
    _, templates = synthetic_cache
    blocks = [b"fixture unused\n"] * 9
    blocks[0], blocks[3], blocks[7] = (
        templates["curator.txt"],
        templates["executor.txt"],
        templates["distillation.txt"],
    )
    appendix = b"".join(
        b"\\begin{lstlisting}[basicstyle=example]\n" + body + b"\\end{lstlisting}\n"
        for body in blocks
    )
    figure = b"synthetic PDF member"
    jitmem = archive([(paper.JITMEM_MEMBER, appendix, tarfile.REGTYPE)])
    skillos = archive([(paper.SKILLOS_MEMBER, figure, tarfile.REGTYPE)])
    monkeypatch.setattr(paper, "JITMEM_ARCHIVE_SHA256", digest(jitmem))
    monkeypatch.setattr(paper, "JITMEM_MEMBER_SHA256", digest(appendix))
    monkeypatch.setattr(paper, "SKILLOS_ARCHIVE_SHA256", digest(skillos))
    monkeypatch.setattr(paper, "SKILLOS_MEMBER_SHA256", digest(figure))

    def extract_judge(data):
        assert data == figure
        return templates["judge_system.txt"].decode(), templates["judge_user.txt"].decode()

    monkeypatch.setattr(paper, "_extract_judge", extract_judge)
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "jitmem_source.bin").write_bytes(jitmem)
    (sources / "skillos_source.tar").write_bytes(skillos)
    output = tmp_path / "prepared"
    result = paper.prepare_assets(output, source_dir=sources)
    assert result.curator_system == "fixture system"
    assert result.provenance == paper.asset_provenance()
    assert not (output / ".sources").exists()
    for name, data in templates.items():
        assert (output / name).read_bytes() == data

    # Even an otherwise valid cached archive cannot substitute a different source member.
    monkeypatch.setattr(paper, "JITMEM_MEMBER_SHA256", digest(b"different member"))
    with pytest.raises(paper.PaperAssetError, match="hash does not match"):
        paper.prepare_assets(tmp_path / "rejected", source_dir=sources)
    assert not (tmp_path / "rejected").exists()


def test_archive_extracts_only_expected_member_without_filesystem_writes(tmp_path):
    data = archive(
        [("wanted", b"expected", tarfile.REGTYPE), ("unrelated", b"unused", tarfile.REGTYPE)]
    )
    assert paper.extract_archive_member(data, "wanted") == b"expected"
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(paper.PaperAssetError, match="member is missing"):
        paper.extract_archive_member(data, "absent")


@pytest.mark.parametrize("name", ["../escape", "/escape", "a/../escape", "a\\escape", "./escape"])
def test_archive_refuses_unsafe_paths_even_on_unrequested_members(name):
    data = archive([(name, b"unused", tarfile.REGTYPE), ("wanted", b"x", tarfile.REGTYPE)])
    with pytest.raises(paper.PaperAssetError, match="unsafe member path"):
        paper.extract_archive_member(data, "wanted")


@pytest.mark.parametrize("name", ["wanted", "unrelated"])
def test_archive_refuses_duplicate_members(name):
    data = archive([(name, b"a", tarfile.REGTYPE), (name, b"b", tarfile.REGTYPE)])
    with pytest.raises(paper.PaperAssetError, match="duplicate members"):
        paper.extract_archive_member(data, "wanted")


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE]
)
def test_archive_refuses_links_and_special_files(kind):
    data = archive([("unrequested", b"", kind), ("wanted", b"x", tarfile.REGTYPE)])
    with pytest.raises(paper.PaperAssetError, match="unsupported member type"):
        paper.extract_archive_member(data, "wanted")


def test_archive_enforces_download_expansion_and_member_count_limits(monkeypatch):
    data = archive([("wanted", b"12345678", tarfile.REGTYPE)])
    monkeypatch.setattr(paper, "MAX_EXPANDED_BYTES", 4)
    with pytest.raises(paper.PaperAssetError, match="expanded size limit"):
        paper.extract_archive_member(data, "wanted")
    monkeypatch.setattr(paper, "MAX_EXPANDED_BYTES", 100)
    monkeypatch.setattr(paper, "MAX_ARCHIVE_MEMBERS", 0)
    with pytest.raises(paper.PaperAssetError, match="too many members"):
        paper.extract_archive_member(data, "wanted")
    monkeypatch.setattr(paper, "MAX_DOWNLOAD_BYTES", 1)
    with pytest.raises(paper.PaperAssetError, match="Source archive exceeds the size limit"):
        paper.extract_archive_member(data, "wanted")


def test_archive_refuses_unsupported_payload():
    with pytest.raises(paper.PaperAssetError, match="supported complete tar archive"):
        paper.extract_archive_member(b"not an archive", "wanted")


def test_listing_extraction_preserves_whitespace_and_refuses_incomplete_delimiters():
    text = "\\begin{lstlisting}[basicstyle=example]\nline  1\n\nline 2\n\\end{lstlisting}"
    assert paper.extract_listings(text) == ["line  1\n\nline 2\n"]
    with pytest.raises(paper.PaperAssetError, match="listing delimiters"):
        paper.extract_listings(text + "\\begin{lstlisting}\nincomplete")


@pytest.mark.parametrize(
    "url",
    [
        "http://arxiv.org/src/example",
        "https://unrelated.invalid/src/example",
        "https://user@arxiv.org/src/example",
        "https://arxiv.org/src/example?query=example",
        "https://arxiv.org:invalid/src/example",
    ],
)
def test_download_rejects_untrusted_or_credential_bearing_urls_before_network(url):
    with pytest.raises(paper.PaperAssetError):
        paper._check_download_url(url)
