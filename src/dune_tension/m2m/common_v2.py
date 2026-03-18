"""Compatibility wrapper around :mod:`dune_tension.m2m.common`.

This module is kept for legacy imports but now re-exports the single canonical
M2M client implementation.
"""

from __future__ import annotations

from dune_tension.m2m.common import (  # noqa: F401
    M2MError,
    ConnectToAPI,
    ConvertShortUUID,
    CreateComponent,
    EditAction,
    EditComponent,
    GetAction,
    GetComponent,
    GetListOfActions,
    GetListOfComponents,
    PerformAction,
)

__all__ = [
    "M2MError",
    "ConnectToAPI",
    "ConvertShortUUID",
    "CreateComponent",
    "GetComponent",
    "EditComponent",
    "GetListOfComponents",
    "PerformAction",
    "GetAction",
    "EditAction",
    "GetListOfActions",
]
