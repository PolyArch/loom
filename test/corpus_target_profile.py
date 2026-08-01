#!/usr/bin/env python3
"""Exact target-profile providers for repository corpus conformance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory


class TargetProfileDisposition(Enum):
    RUNNABLE = "runnable"
    INCOMPATIBLE_ISA = "incompatible-instruction-set"
    PROVIDER_UNAVAILABLE = "provider-unavailable"


@dataclass(frozen=True)
class ResolvedTargetProfile:
    disposition: TargetProfileDisposition
    compile_flags: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class _TargetProfileSpec:
    instruction_set_family: str
    provider_available: bool
    suites: frozenset[str] | None = None
    compile_flags: tuple[str, ...] = ()


_TARGET_PROFILES = {
    corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE: _TargetProfileSpec(
        instruction_set_family="riscv",
        provider_available=True,
    ),
    corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE: _TargetProfileSpec(
        instruction_set_family="riscv",
        provider_available=True,
        suites=frozenset({"cmsis-dsp"}),
        compile_flags=("-D__ARM_FP16_FORMAT_IEEE=1", "-D__fp16=_Float16"),
    ),
    "dsp": _TargetProfileSpec(
        instruction_set_family="arm",
        provider_available=False,
    ),
    "mve": _TargetProfileSpec(
        instruction_set_family="arm",
        provider_available=False,
    ),
    "neon": _TargetProfileSpec(
        instruction_set_family="arm",
        provider_available=False,
    ),
}


def _instruction_set_family(target_triple: str) -> str:
    architecture = target_triple.split("-", 1)[0]
    if architecture.startswith("riscv"):
        return "riscv"
    if architecture.startswith(("arm", "thumb", "aarch64")):
        return "arm"
    return architecture


def resolve_target_profile(
    suite: str, target_profile: str, target_triple: str
) -> ResolvedTargetProfile:
    spec = _TARGET_PROFILES.get(target_profile)
    if spec is None:
        return ResolvedTargetProfile(
            TargetProfileDisposition.PROVIDER_UNAVAILABLE,
            detail=f"target profile provider is unknown: {target_profile}",
        )

    selected_family = _instruction_set_family(target_triple)
    if selected_family != spec.instruction_set_family:
        return ResolvedTargetProfile(
            TargetProfileDisposition.INCOMPATIBLE_ISA,
            detail=(
                f"target profile {target_profile} requires "
                f"{spec.instruction_set_family}, but selected target "
                f"{target_triple} belongs to {selected_family}"
            ),
        )

    if spec.suites is not None and suite not in spec.suites:
        return ResolvedTargetProfile(
            TargetProfileDisposition.PROVIDER_UNAVAILABLE,
            detail=(
                f"target profile provider {target_profile} is unavailable "
                f"for suite {suite}"
            ),
        )
    if not spec.provider_available:
        return ResolvedTargetProfile(
            TargetProfileDisposition.PROVIDER_UNAVAILABLE,
            detail=f"target profile provider is unavailable: {target_profile}",
        )
    return ResolvedTargetProfile(
        TargetProfileDisposition.RUNNABLE,
        compile_flags=spec.compile_flags,
    )
