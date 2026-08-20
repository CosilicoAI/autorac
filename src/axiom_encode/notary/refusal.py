"""Typed refusal values shared by notary verification primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Refusal:
    """A deterministic refusal produced instead of a prose exception."""

    code: str
    path: str | None
    detail: str
