#!/usr/bin/env python3
"""Typed generated-public CMSIS-NN workload protocols."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class GeneratedCmsisNnProtocol:
    source: str
    protocol_symbol: str
    compiled_owner: Path
    authoritative_owner: Path


def _render_in_place_activation(
    symbol: str,
    element_type: str,
    values: str,
    expected: str,
) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum {{ kElementCount = 17 }};

static const {element_type} kInput[kElementCount] = {{
    {values},
}};

int main(void)
{{
    {element_type} data[kElementCount];
    for (size_t index = 0; index < kElementCount; ++index)
    {{
        data[index] = kInput[index];
    }}

    {symbol}(data, kElementCount);

    for (size_t index = 0; index < kElementCount; ++index)
    {{
        const {element_type} expected = {expected};
        if (data[index] != expected)
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_relu_q7() -> str:
    return _render_in_place_activation(
        "arm_relu_q7",
        "int8_t",
        "-128, -31, -7, -1, 0, 1, 2, 6, 7, 15, 31, 63, 95, 126, 127, -64, -2",
        "kInput[index] < 0 ? 0 : kInput[index]",
    )


def _render_relu_q15() -> str:
    return _render_in_place_activation(
        "arm_relu_q15",
        "int16_t",
        "-32768, -16384, -1025, -1, 0, 1, 2, 6, 7, "
        "255, 1024, 8191, 16384, 32766, 32767, -4096, -2",
        "kInput[index] < 0 ? 0 : kInput[index]",
    )


def _render_relu6_s8() -> str:
    return _render_in_place_activation(
        "arm_relu6_s8",
        "int8_t",
        "-128, -31, -7, -1, 0, 1, 2, 5, 6, 7, 15, 31, 63, 126, 127, -64, -2",
        "kInput[index] < 0 ? 0 : (kInput[index] > 6 ? 6 : kInput[index])",
    )


def _render_reshape_s8() -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum { kElementCount = 33 };

static const int8_t kInput[kElementCount] = {
    -128, -97, -64, -33, -16, -8, -4, -2, -1, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 31, 32, 47, 63,
    64, 79, 95, 111, 126, 127, -55, 42,
};

int main(void)
{
    int8_t output[kElementCount] = {0};
    arm_reshape_s8(kInput, output, kElementCount);
    for (size_t index = 0; index < kElementCount; ++index)
    {
        if (output[index] != kInput[index])
        {
            return 1;
        }
    }
    return 0;
}
"""


def _render_q7_to_q15_with_offset() -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

enum { kElementCount = 33 };
static const int16_t kOffset = 257;

static const int8_t kInput[kElementCount] = {
    -128, -97, -64, -33, -16, -8, -4, -2, -1, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 31, 32, 47, 63,
    64, 79, 95, 111, 126, 127, -55, 42,
};

int main(void)
{
    int16_t output[kElementCount] = {0};
    arm_q7_to_q15_with_offset(
        kInput, output, kElementCount, kOffset);
    for (size_t index = 0; index < kElementCount; ++index)
    {
        const int16_t expected = (int16_t)kInput[index] + kOffset;
        if (output[index] != expected)
        {
            return 1;
        }
    }
    return 0;
}
"""


_RENDERERS: dict[tuple[str, str], Callable[[], str]] = {
    ("arm_relu_q7", "void (int8_t *, uint16_t)"): _render_relu_q7,
    ("arm_relu_q15", "void (int16_t *, uint16_t)"): _render_relu_q15,
    ("arm_relu6_s8", "void (int8_t *, uint16_t)"): _render_relu6_s8,
    (
        "arm_reshape_s8",
        "void (const int8_t *, int8_t *, const uint32_t)",
    ): _render_reshape_s8,
    (
        "arm_q7_to_q15_with_offset",
        "void (const int8_t *, int16_t *, int32_t, int16_t)",
    ): _render_q7_to_q15_with_offset,
}


def _owned_path(raw_path: str, external_root: Path) -> Path:
    path = Path(raw_path)
    try:
        relative = path.relative_to(Path("externals/cmsis-nn"))
    except ValueError as exc:
        raise WorkloadProviderError(
            f"generated CMSIS-NN owner escapes its source root: {path}"
        ) from exc
    resolved = external_root / "cmsis-nn" / relative
    if not resolved.is_file():
        raise WorkloadProviderError(f"generated CMSIS-NN owner is unavailable: {path}")
    return resolved


def _declaration_owner(
    producer: corpus_inventory.CmsisNnGeneratedWorkloadProducer,
    external_root: Path,
) -> Path:
    declaration = re.compile(
        rf"\b{re.escape(producer.public_symbol)}\s*\([^;{{}}]*\)\s*;",
        re.DOTALL,
    )
    owners = [
        owner
        for owner in (
            _owned_path(definition, external_root)
            for definition in producer.definitions
        )
        if declaration.search(owner.read_text(encoding="utf-8"))
    ]
    if len(owners) != 1:
        raise WorkloadProviderError(
            "generated CMSIS-NN protocol must resolve one public declaration: "
            f"{producer.public_symbol}"
        )
    return owners[0]


def render_generated_cmsis_nn_protocol(
    workload: corpus_inventory.ProgramWorkload,
    external_root: Path,
) -> GeneratedCmsisNnProtocol:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisNnGeneratedWorkloadProducer):
        raise WorkloadProviderError(
            f"workload is not a generated CMSIS-NN protocol: {workload.identity}"
        )
    renderer = _renderer_for(workload)
    if renderer is None:
        raise WorkloadProviderError(
            f"generated CMSIS-NN protocol is unsupported: {workload.identity}"
        )
    if len(workload.sources) != 1:
        raise WorkloadProviderError(
            "generated CMSIS-NN protocol must select one implementation source: "
            f"{workload.identity}"
        )
    compiled_owner = _owned_path(workload.sources[0], external_root)
    return GeneratedCmsisNnProtocol(
        source=renderer(),
        protocol_symbol=producer.public_symbol,
        compiled_owner=compiled_owner,
        authoritative_owner=_declaration_owner(producer, external_root),
    )


def _renderer_for(
    workload: corpus_inventory.ProgramWorkload,
) -> Callable[[], str] | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisNnGeneratedWorkloadProducer):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    if (
        len(calls) != 1
        or producer.public_symbol != calls[0][0]
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or workload.oracle.kind != "generated-native-reference"
        or workload.oracle.path != producer.public_symbol
    ):
        return None
    return _RENDERERS.get(calls[0])


def supports_generated_cmsis_nn_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> bool:
    return _renderer_for(workload) is not None
