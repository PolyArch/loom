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
    compiled_owner: Path | None
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


def _render_relu_q7(_wrapper_symbol: str) -> str:
    return _render_in_place_activation(
        "arm_relu_q7",
        "int8_t",
        "-128, -31, -7, -1, 0, 1, 2, 6, 7, 15, 31, 63, 95, 126, 127, -64, -2",
        "kInput[index] < 0 ? 0 : kInput[index]",
    )


def _render_relu_q15(_wrapper_symbol: str) -> str:
    return _render_in_place_activation(
        "arm_relu_q15",
        "int16_t",
        "-32768, -16384, -1025, -1, 0, 1, 2, 6, 7, "
        "255, 1024, 8191, 16384, 32766, 32767, -4096, -2",
        "kInput[index] < 0 ? 0 : kInput[index]",
    )


def _render_relu6_s8(_wrapper_symbol: str) -> str:
    return _render_in_place_activation(
        "arm_relu6_s8",
        "int8_t",
        "-128, -31, -7, -1, 0, 1, 2, 5, 6, 7, 15, 31, 63, 126, 127, -64, -2",
        "kInput[index] < 0 ? 0 : (kInput[index] > 6 ? 6 : kInput[index])",
    )


def _render_reshape_s8(_wrapper_symbol: str) -> str:
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


def _render_q7_to_q15_with_offset(_wrapper_symbol: str) -> str:
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


def _render_header_memory(
    wrapper_symbol: str,
    operation: str,
    element_type: str,
    input_values: str,
) -> str:
    is_set = operation == "arm_memset_s8"
    wrapper_parameters = (
        f"{element_type} *output, const int8_t value, uint32_t byte_count"
        if is_set
        else f"{element_type} *output, const {element_type} *input, uint32_t byte_count"
    )
    operation_arguments = (
        "output, value, byte_count" if is_set else "output, input, byte_count"
    )
    main_arguments = (
        "output, kFillValue, kByteCount" if is_set else "output, kInput, kByteCount"
    )
    expected = "kFillValue" if is_set else "kInput[index]"
    input_declaration = ""
    if not is_set:
        input_declaration = f"""static const {element_type} kInput[kElementCount] = {{
    {input_values},
}};

"""
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

enum {{ kElementCount = 33 }};
enum {{ kByteCount = kElementCount * sizeof({element_type}) }};
static const int8_t kFillValue = -37;

{input_declaration}LOOM_NOINLINE void {wrapper_symbol}({wrapper_parameters})
{{
    {operation}({operation_arguments});
}}

int main(void)
{{
    {element_type} output[kElementCount] = {{0}};
    {wrapper_symbol}({main_arguments});
    for (size_t index = 0; index < kElementCount; ++index)
    {{
        if (output[index] != {expected})
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_memcpy_s8(wrapper_symbol: str) -> str:
    return _render_header_memory(
        wrapper_symbol,
        "arm_memcpy_s8",
        "int8_t",
        "-128, -97, -64, -33, -16, -8, -4, -2, -1, 0, 1, "
        "2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 31, 32, 47, 63, "
        "64, 79, 95, 111, 126, 127, -55, 42",
    )


def _render_memcpy_q15(wrapper_symbol: str) -> str:
    return _render_header_memory(
        wrapper_symbol,
        "arm_memcpy_q15",
        "int16_t",
        "-32768, -30001, -16384, -8193, -1024, -255, -16, -2, -1, 0, 1, "
        "2, 3, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 4096, "
        "8191, 16384, 24575, 30000, 32766, 32767, -4096, 777",
    )


def _render_memset_s8(wrapper_symbol: str) -> str:
    return _render_header_memory(
        wrapper_symbol,
        "arm_memset_s8",
        "int8_t",
        "",
    )


def _render_concatenation(_wrapper_symbol: str, axis: str) -> str:
    output_dimension = {
        "x": "kOutputX",
        "y": "kOutputY",
        "z": "kOutputZ",
    }.get(axis)
    extra_argument = "" if axis == "w" else f",\n        {output_dimension}"
    destination = {
        "x": "(((w * kInputZ + z) * kInputY + y) * kOutputX + (x + kOffset))",
        "y": "(((w * kInputZ + z) * kOutputY + (y + kOffset)) * kInputX + x)",
        "z": "(((w * kOutputZ + (z + kOffset)) * kInputY + y) * kInputX + x)",
        "w": "((((w + kOffset) * kInputZ + z) * kInputY + y) * kInputX + x)",
    }[axis]
    output_count = {
        "x": "kOutputX * kInputY * kInputZ * kInputW",
        "y": "kInputX * kOutputY * kInputZ * kInputW",
        "z": "kInputX * kInputY * kOutputZ * kInputW",
        "w": "kInputX * kInputY * kInputZ * kOutputW",
    }[axis]
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum {{
    kInputX = 3,
    kInputY = 2,
    kInputZ = 2,
    kInputW = 2,
    kOutputX = 5,
    kOutputY = 4,
    kOutputZ = 4,
    kOutputW = 4,
    kOffset = 1,
    kInputCount = kInputX * kInputY * kInputZ * kInputW,
    kOutputCount = {output_count},
}};

int main(void)
{{
    int8_t input[kInputCount];
    int8_t output[kOutputCount];
    int8_t expected[kOutputCount];
    for (size_t source = 0; source < kInputCount; ++source)
    {{
        input[source] = (int8_t)((source * 17 + 11) % 127 - 63);
    }}
    for (size_t index = 0; index < kOutputCount; ++index)
    {{
        output[index] = -101;
        expected[index] = -101;
    }}

    arm_concatenation_s8_{axis}(
        input, kInputX, kInputY, kInputZ, kInputW, output{extra_argument},
        kOffset);

    for (size_t w = 0; w < kInputW; ++w)
    {{
        for (size_t z = 0; z < kInputZ; ++z)
        {{
            for (size_t y = 0; y < kInputY; ++y)
            {{
                for (size_t x = 0; x < kInputX; ++x)
                {{
                    const size_t source =
                        (((w * kInputZ + z) * kInputY + y) * kInputX + x);
                    const size_t destination = {destination};
                    expected[destination] = input[source];
                }}
            }}
        }}
    }}
    for (size_t index = 0; index < kOutputCount; ++index)
    {{
        if (output[index] != expected[index])
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_concatenation_w(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "w")


def _render_concatenation_x(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "x")


def _render_concatenation_y(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "y")


def _render_concatenation_z(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "z")


_RENDERERS: dict[tuple[str, str], Callable[[str], str]] = {
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
    (
        "arm_memcpy_s8",
        "void (int8_t *restrict, const int8_t *restrict, uint32_t)",
    ): _render_memcpy_s8,
    (
        "arm_memcpy_q15",
        "void (int16_t *restrict, const int16_t *restrict, uint32_t)",
    ): _render_memcpy_q15,
    (
        "arm_memset_s8",
        "void (int8_t *, const int8_t, uint32_t)",
    ): _render_memset_s8,
    (
        "arm_concatenation_s8_w",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint32_t)",
    ): _render_concatenation_w,
    (
        "arm_concatenation_s8_x",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint16_t, const uint32_t)",
    ): _render_concatenation_x,
    (
        "arm_concatenation_s8_y",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint16_t, const uint32_t)",
    ): _render_concatenation_y,
    (
        "arm_concatenation_s8_z",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint16_t, const uint32_t)",
    ): _render_concatenation_z,
}


_HEADER_ONLY_PROTOCOLS = {
    "arm_memcpy_s8",
    "arm_memcpy_q15",
    "arm_memset_s8",
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
        rf"\b{re.escape(producer.public_symbol)}\s*\([^;{{}}]*\)\s*(?:;|{{)",
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
    wrapper_symbol: str,
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
    header_only = producer.public_symbol in _HEADER_ONLY_PROTOCOLS
    expected_sources = 0 if header_only else 1
    if len(workload.sources) != expected_sources:
        raise WorkloadProviderError(
            "generated CMSIS-NN protocol has an invalid implementation closure: "
            f"{workload.identity}"
        )
    compiled_owner = (
        None if header_only else _owned_path(workload.sources[0], external_root)
    )
    return GeneratedCmsisNnProtocol(
        source=renderer(wrapper_symbol),
        protocol_symbol=wrapper_symbol if header_only else producer.public_symbol,
        compiled_owner=compiled_owner,
        authoritative_owner=_declaration_owner(producer, external_root),
    )


def _renderer_for(
    workload: corpus_inventory.ProgramWorkload,
) -> Callable[[str], str] | None:
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
