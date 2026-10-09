#!/usr/bin/env python3
"""Prepare pinned original prompts into ignored local assets without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jitmem.paper_assets import (  # noqa: E402
    DEFAULT_ASSET_DIRECTORY,
    PaperAssetError,
    prepare_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_ASSET_DIRECTORY)
    parser.add_argument(
        "--source-dir",
        help="Offline directory containing jitmem_source.bin and skillos_source.tar",
    )
    args = parser.parse_args()
    try:
        assets = prepare_assets(output_dir=args.output_dir, source_dir=args.source_dir)
    except PaperAssetError as error:
        parser.exit(1, f"Paper prompt preparation failed: {error}\n")
    print(
        json.dumps(
            {
                "prepared_templates": len(assets.provenance["templates"]),
                "profile": assets.provenance["profile"],
                "template_hashes": assets.provenance["templates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
