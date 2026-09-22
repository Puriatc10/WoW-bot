"""Enforce the NON_CLAIMS single source of truth.

The four non-claim statements have exactly one human-editable source of truth:
``docs/non_claims.json``. Every other representation of those statements in the
repository must be byte-for-byte identical to that file. This module verifies
two of those representations:

* ``src/wow_bot/analysis/aggregate.py``, read through its ``NON_CLAIMS`` tuple.
* ``docs/SOAK_PROTOCOL.md``, read as plain text.

The ``"_comment"`` key of the JSON is documentation for human readers only and
is never treated as a non-claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from wow_bot.analysis.aggregate import NON_CLAIMS as AGG_NON_CLAIMS

REPO_ROOT = Path(__file__).resolve().parent.parent
NON_CLAIMS_JSON_PATH = REPO_ROOT / "docs" / "non_claims.json"
SOAK_PROTOCOL_PATH = REPO_ROOT / "docs" / "SOAK_PROTOCOL.md"

EXPECTED_ENTRY_COUNT = 4


def _load_non_claims() -> list[str]:
    """Return the four non-claim statements from the JSON source of truth.

    Reads only ``docs/non_claims.json``. The ``"_comment"`` key is ignored.
    """
    data = json.loads(NON_CLAIMS_JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(
            f"{NON_CLAIMS_JSON_PATH} must contain a JSON object, got {type(data).__name__}"
        )
    if set(data.keys()) != {"_comment", "non_claims"}:
        raise AssertionError(
            f"{NON_CLAIMS_JSON_PATH} must contain exactly the keys '_comment' and "
            f"'non_claims', got {sorted(data.keys())!r}"
        )

    non_claims = data["non_claims"]
    if not isinstance(non_claims, list):
        raise TypeError(
            f"{NON_CLAIMS_JSON_PATH} 'non_claims' must be a list, got "
            f"{type(non_claims).__name__}"
        )
    if len(non_claims) != EXPECTED_ENTRY_COUNT:
        raise AssertionError(
            f"{NON_CLAIMS_JSON_PATH} 'non_claims' must contain exactly "
            f"{EXPECTED_ENTRY_COUNT} entries, got {len(non_claims)}"
        )
    for index, entry in enumerate(non_claims):
        if not isinstance(entry, str):
            raise TypeError(
                f"{NON_CLAIMS_JSON_PATH} 'non_claims'[{index}] must be a string, got "
                f"{type(entry).__name__}"
            )
    return cast(list[str], non_claims)


def test_non_claims_consistency() -> None:
    """The JSON source of truth must match aggregate.py and SOAK_PROTOCOL.md exactly.

    Reads only ``docs/non_claims.json`` and ``docs/SOAK_PROTOCOL.md`` from disk,
    plus ``NON_CLAIMS`` imported from ``src/wow_bot/analysis/aggregate.py``.
    """
    non_claims = _load_non_claims()
    json_tuple = tuple(non_claims)

    # 1. The JSON source of truth must equal aggregate.py's NON_CLAIMS tuple.
    if json_tuple != AGG_NON_CLAIMS:
        for index in range(min(len(json_tuple), len(AGG_NON_CLAIMS))):
            json_entry = json_tuple[index]
            agg_entry = AGG_NON_CLAIMS[index]
            if json_entry != agg_entry:
                raise AssertionError(
                    f"{NON_CLAIMS_JSON_PATH} diverges from "
                    f"src/wow_bot/analysis/aggregate.py's NON_CLAIMS tuple: the first "
                    f"differing sentence is at index {index}, where the JSON has "
                    f"{json_entry!r} but aggregate.py has {agg_entry!r}"
                )
        raise AssertionError(
            f"{NON_CLAIMS_JSON_PATH} diverges from src/wow_bot/analysis/aggregate.py's "
            f"NON_CLAIMS tuple: the JSON has {len(json_tuple)} entries but aggregate.py "
            f"has {len(AGG_NON_CLAIMS)} entries"
        )

    # 2. Each entry is a distinct, non-empty, single-line sentence.
    for index, entry in enumerate(non_claims):
        if not entry:
            raise AssertionError(
                f"{NON_CLAIMS_JSON_PATH} 'non_claims'[{index}] must not be empty"
            )
        if not entry.endswith("."):
            raise AssertionError(
                f"{NON_CLAIMS_JSON_PATH} 'non_claims'[{index}] must end with a period, "
                f"got {entry!r}"
            )
        if entry != entry.strip():
            raise AssertionError(
                f"{NON_CLAIMS_JSON_PATH} 'non_claims'[{index}] must not have leading or "
                f"trailing whitespace, got {entry!r}"
            )
        if "\t" in entry or "\n" in entry:
            raise AssertionError(
                f"{NON_CLAIMS_JSON_PATH} 'non_claims'[{index}] must not contain a tab or "
                f"newline character, got {entry!r}"
            )

    if len(set(non_claims)) != EXPECTED_ENTRY_COUNT:
        duplicates = sorted({entry for entry in non_claims if non_claims.count(entry) > 1})
        raise AssertionError(
            f"{NON_CLAIMS_JSON_PATH} 'non_claims' entries must all be distinct, but the "
            f"following appear more than once: {duplicates!r}"
        )

    # 3. Every entry appears verbatim in docs/SOAK_PROTOCOL.md.
    protocol = SOAK_PROTOCOL_PATH.read_text(encoding="utf-8")
    for index, entry in enumerate(non_claims):
        if entry not in protocol:
            raise AssertionError(
                f"{SOAK_PROTOCOL_PATH} does not contain the non-claim at index {index} "
                f"verbatim; the missing sentence is {entry!r}"
            )
