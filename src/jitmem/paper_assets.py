"""Prepare and verify locally cached prompts from the two cited primary sources.

Third-party prompt text is downloaded into local assets, never embedded in this
module. Evaluation only calls ``load_assets``; preparation is an explicit step.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping

DEFAULT_ASSET_DIRECTORY = "outputs/paper_prompts/v1"
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
MAX_EXPANDED_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 256
MAX_TEMPLATE_BYTES = 32 * 1024

JITMEM_SOURCE_URL = "https://arxiv.org/src/2609.27334v1"
JITMEM_ARCHIVE_SHA256 = "7fbe36724756652232976b1b3de5e93710b0a8c36392cda5329135bb5b4ed667"
JITMEM_MEMBER = "sections/appendix_prompt.tex"
JITMEM_MEMBER_SHA256 = "f0822e51102dfb5fbc4c30f6bb61d11585963d0f2295f167189a53c2badc9a8b"
SKILLOS_SOURCE_URL = "https://arxiv.org/src/2605.06614v1"
SKILLOS_ARCHIVE_SHA256 = "3a3f45d849aefdf58905ecb2ee836ea0035e827ae4af740e0960ac458c80c076"
SKILLOS_MEMBER = "figures/alfworld_judge.pdf"
SKILLOS_MEMBER_SHA256 = "12109bc451eee6e163944e1ca4f6510381a38ea9b38809ab42c96a362ae370e1"

EXPECTED_TEMPLATE_HASHES: Mapping[str, str] = {
    "curator.txt": "5194cf0991f6e6a9faf80c85b9ebe8d30f7727844432776936d63f84d2b17c04",
    "executor.txt": "1eccc28bab43f335fdfcad9a322fc687f519fb750131ed95cddcf8d5b2642b61",
    "distillation.txt": "778916de8c3e8244632d4ebf1cc2d0b4817f3010b01a0f4e317a392444e970b0",
    "judge_system.txt": "02fbbbb8a63a1caf090b7e9dfa5c0994862c07851096892ae56fde3ea3388db6",
    "judge_user.txt": "fda96911b4daf6f1cc0f331043f9b68557e197fb1081161224b03ff6ca0b8ba5",
}


class PaperAssetError(ValueError):
    """An original prompt asset is unavailable or does not match its pinned source."""


@dataclass(frozen=True, slots=True, repr=False)
class PaperAssets:
    curator_system: str
    curator_user: str
    executor: str
    judge_system: str
    judge_user: str
    distillation: str
    _provenance_json: str = field(repr=False)

    @property
    def provenance(self) -> dict:
        """Return an independent public-safe copy, preserving asset immutability."""
        return json.loads(self._provenance_json)

    @property
    def metadata(self) -> dict:
        return self.provenance


def asset_provenance() -> dict:
    """Describe pinned source material without local paths or template contents."""
    return {
        "schema_version": 1,
        "profile": "paper-v1",
        "sources": {
            "jitmem": {
                "version": "2609.27334v1",
                "url": JITMEM_SOURCE_URL,
                "archive_sha256": JITMEM_ARCHIVE_SHA256,
                "member": JITMEM_MEMBER,
                "member_sha256": JITMEM_MEMBER_SHA256,
                "printed_pages": {"curator": 14, "executor": 15, "distillation": 17},
            },
            "skillos": {
                "version": "2605.06614v1",
                "url": SKILLOS_SOURCE_URL,
                "archive_sha256": SKILLOS_ARCHIVE_SHA256,
                "member": SKILLOS_MEMBER,
                "member_sha256": SKILLOS_MEMBER_SHA256,
                "printed_page": 22,
                "figure": 13,
                "extraction_dependency": "pypdf==6.10.0",
            },
        },
        "templates": dict(EXPECTED_TEMPLATE_HASHES),
        "license": {
            "name": "arXiv perpetual non-exclusive license",
            "url": "https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html",
            "distribution": "Third-party full templates remain local downloaded assets.",
        },
        "extraction": {
            "jitmem": "Verbatim lstlisting blocks 0, 3 and 7, including the final newline.",
            "curator_roles": "Remove displayed role markers and only exterior blank newlines.",
            "skillos": (
                "Extract Figure 13 PDF text; split its system/input boxes; reflow headings and "
                "bullets; decode displayed backslash-n in the input. Words are preserved, but "
                "the author's original runtime whitespace is unavailable."
            ),
        },
        "undisclosed_serialization": [
            "Full-storage curator modification and correctness-label syntax/position.",
            "ALFWorld raw trajectory, action history and admissible-action serialization.",
            "Executor/distillation API message role wrapping.",
            "Distillation task/trajectory input wrapper.",
        ],
    }


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify(data: bytes, expected: str, label: str) -> None:
    if _digest(data) != expected:
        raise PaperAssetError(f"Pinned content hash does not match: {label}")


def _read_bounded(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as error:
        raise PaperAssetError(f"Required local asset cannot be read: {path.name}") from error
    if len(data) > limit:
        raise PaperAssetError(f"Local asset exceeds the size limit: {path.name}")
    return data


def extract_archive_member(archive: bytes, member_name: str) -> bytes:
    """Read one regular member without writing or expanding an archive to disk."""
    if len(archive) > MAX_DOWNLOAD_BYTES:
        raise PaperAssetError("Source archive exceeds the size limit")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source:
            found: tarfile.TarInfo | None = None
            seen: set[str] = set()
            expanded = 0
            for index, member in enumerate(source):
                if index >= MAX_ARCHIVE_MEMBERS:
                    raise PaperAssetError("Source archive has too many members")
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or not path.parts
                    or str(path) != member.name.rstrip("/")
                ):
                    raise PaperAssetError("Source archive contains an unsafe member path")
                if member.name in seen:
                    raise PaperAssetError("Source archive contains duplicate members")
                seen.add(member.name)
                if not member.isfile() and not member.isdir():
                    raise PaperAssetError("Source archive contains an unsupported member type")
                expanded += member.size
                if member.size > MAX_DOWNLOAD_BYTES or expanded > MAX_EXPANDED_BYTES:
                    raise PaperAssetError("Source archive exceeds the expanded size limit")
                if member.name == member_name:
                    if not member.isfile():
                        raise PaperAssetError("Expected source member is not a regular file")
                    found = member
            if found is None:
                raise PaperAssetError("Expected source member is missing")
            stream = source.extractfile(found)
            if stream is None:
                raise PaperAssetError("Expected source member cannot be read")
            with stream:
                result = stream.read(MAX_DOWNLOAD_BYTES + 1)
            if len(result) != found.size:
                raise PaperAssetError("Expected source member has an invalid size")
            return result
    except (tarfile.TarError, OSError, EOFError) as error:
        raise PaperAssetError("Source is not a supported complete tar archive") from error


def extract_listings(text: str) -> list[str]:
    """Return listing bodies exactly; do not normalize source whitespace or TeX."""
    blocks = re.findall(
        r"\\begin\{lstlisting\}(?:\[[^\n]*\])?\n(.*?)\\end\{lstlisting\}",
        text,
        re.DOTALL,
    )
    if not blocks or len(blocks) != text.count(r"\begin{lstlisting}"):
        raise PaperAssetError("Source has missing or unsupported listing delimiters")
    return blocks


def _extract_judge(figure: bytes) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise PaperAssetError("Preparation requires pypdf==6.10.0; evaluation does not") from error
    try:
        reader = PdfReader(io.BytesIO(figure))
        if len(reader.pages) != 1:
            raise PaperAssetError("Expected a single-page SkillOS judge figure")
        raw = reader.pages[0].extract_text()
        if not isinstance(raw, str) or raw.count("# Inputs") != 1:
            raise PaperAssetError("SkillOS judge input separator is unavailable")
        system, user = raw.split("# Inputs", 1)
        if "\nSystem Instruction" not in user:
            raise PaperAssetError("SkillOS judge figure box labels are unavailable")
        user = ("# Inputs" + user).split("\nSystem Instruction", 1)[0].strip()
        system = (
            system.strip()
            .replace(" # Task  ", "\n\n# Task\n")
            .replace(' ## What "success" means  ', '\n\n## What "success" means\n')
            .replace(" ## Strictness  ", "\n\n## Strictness\n")
            .replace(" # Output  ", "\n\n# Output\n")
        )
        system = (
            system.replace("  -", "\n- ")
            .replace(" - Ignore", "\n- Ignore")
            .replace("  ", " ")
            .strip()
        )
        user = (
            user.replace("# Inputs  ## Task description \\n", "# Inputs\n\n## Task description\n")
            .replace("  ## Trajectory  ", "\n\n## Trajectory\n")
            .replace("\\n", "\n")
            .strip()
        )
    except PaperAssetError:
        raise
    except Exception as error:
        raise PaperAssetError("SkillOS judge figure could not be extracted") from error
    _verify(system.encode(), EXPECTED_TEMPLATE_HASHES["judge_system.txt"], "judge system")
    _verify(user.encode(), EXPECTED_TEMPLATE_HASHES["judge_user.txt"], "judge input")
    return system, user


def _check_download_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise PaperAssetError("Paper source URL is malformed") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PaperAssetError("Paper preparation accepts only HTTPS arXiv source URLs")


class _SourceRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        _check_download_url(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


def _download(url: str) -> bytes:
    _check_download_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "JitMem-paper-preparation/1.0"})
    opener = urllib.request.build_opener(_SourceRedirectHandler())
    try:
        with opener.open(request, timeout=60) as response:
            _check_download_url(response.geturl())
            data = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise PaperAssetError("Pinned public paper source could not be downloaded") from error
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise PaperAssetError("Paper download exceeds the size limit")
    return data


def _atomic_write(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".paper-assets-", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_assets(
    output_dir: str | Path = DEFAULT_ASSET_DIRECTORY,
    source_dir: str | Path | None = None,
) -> PaperAssets:
    """Download or read pinned archives, extract all assets, and validate the result.

    ``source_dir`` selects offline cached archives named ``jitmem_source.bin`` and
    ``skillos_source.tar``. With this argument, missing files never trigger a fetch.
    """
    output = Path(output_dir)
    cache = Path(source_dir) if source_dir is not None else output / ".sources"
    inputs = (
        ("jitmem_source.bin", JITMEM_SOURCE_URL, JITMEM_ARCHIVE_SHA256),
        ("skillos_source.tar", SKILLOS_SOURCE_URL, SKILLOS_ARCHIVE_SHA256),
    )
    archives: dict[str, bytes] = {}
    for filename, url, expected in inputs:
        path = cache / filename
        data = (
            _read_bounded(path, MAX_DOWNLOAD_BYTES)
            if source_dir is not None or path.exists()
            else _download(url)
        )
        _verify(data, expected, filename)
        archives[filename] = data
    appendix = extract_archive_member(archives["jitmem_source.bin"], JITMEM_MEMBER)
    _verify(appendix, JITMEM_MEMBER_SHA256, "JitMem prompt appendix")
    try:
        blocks = extract_listings(appendix.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise PaperAssetError("JitMem appendix is not UTF-8 source text") from error
    if len(blocks) != 9:
        raise PaperAssetError("Pinned JitMem appendix listing count changed")
    figure = extract_archive_member(archives["skillos_source.tar"], SKILLOS_MEMBER)
    _verify(figure, SKILLOS_MEMBER_SHA256, "SkillOS judge figure")
    judge_system, judge_user = _extract_judge(figure)
    templates = {
        "curator.txt": blocks[0].encode(),
        "executor.txt": blocks[3].encode(),
        "distillation.txt": blocks[7].encode(),
        "judge_system.txt": judge_system.encode(),
        "judge_user.txt": judge_user.encode(),
    }
    for filename, data in templates.items():
        _verify(data, EXPECTED_TEMPLATE_HASHES[filename], filename)
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in templates.items():
        _atomic_write(output / filename, data)
    manifest = json.dumps(asset_provenance(), indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(output / "manifest.json", manifest)
    if source_dir is None:
        cache.mkdir(parents=True, exist_ok=True)
        for filename, data in archives.items():
            _atomic_write(cache / filename, data)
    return load_assets(output)


def load_assets(directory: str | Path = DEFAULT_ASSET_DIRECTORY) -> PaperAssets:
    """Load local assets only, failing closed for any missing or changed content."""
    root = Path(directory)
    templates: dict[str, str] = {}
    for filename, expected in EXPECTED_TEMPLATE_HASHES.items():
        data = _read_bounded(root / filename, MAX_TEMPLATE_BYTES)
        _verify(data, expected, filename)
        try:
            templates[filename] = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PaperAssetError(f"Original template is not UTF-8: {filename}") from error
    try:
        manifest = json.loads(_read_bounded(root / "manifest.json", MAX_TEMPLATE_BYTES))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperAssetError("Local paper prompt manifest is invalid") from error
    provenance = asset_provenance()
    if manifest != provenance:
        raise PaperAssetError("Local paper prompt provenance does not match the pinned sources")
    curator = templates["curator.txt"]
    if not curator.startswith("<System Prompt>\n") or curator.count("<User Prompt>\n") != 1:
        raise PaperAssetError("Original curator role markers are invalid")
    system, user = curator.removeprefix("<System Prompt>\n").split("<User Prompt>\n", 1)
    return PaperAssets(
        curator_system=system.strip("\n"),
        curator_user=user.strip("\n"),
        executor=templates["executor.txt"].strip("\n"),
        judge_system=templates["judge_system.txt"],
        judge_user=templates["judge_user.txt"],
        distillation=templates["distillation.txt"].strip("\n"),
        _provenance_json=json.dumps(provenance, sort_keys=True),
    )
