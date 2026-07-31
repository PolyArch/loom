"""Typed CMSIS-DSP operator protocols and official pattern decoding."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class BasicIntegerType:
    suffix: str
    bits: int
    value_type: str
    value_signature: str
    unsigned_type: str
    dot_type: str
    dot_suffix: str
    scalar: int
    clip_lower: int
    clip_upper: int
    value_tolerance: int
    dot_tolerance: int


@dataclass(frozen=True)
class WindowType:
    test_class: str
    suffix: str
    value_type: str
    absolute_error: str
    relative_error: str


@dataclass(frozen=True)
class ElementaryMathType:
    suffix: str
    value_type: str
    absolute_error: str
    relative_error: str


@dataclass(frozen=True)
class ElementaryMathProtocol:
    type_spec: ElementaryMathType
    kind: str
    symbol: str
    signature: str
    input_a: str
    expected: str
    input_b: str | None = None
    dimensions: str | None = None


_BASIC_INTEGER_TYPES = {
    "BasicTestsQ31": BasicIntegerType(
        suffix="q31",
        bits=32,
        value_type="q31_t",
        value_signature="i32",
        unsigned_type="uint32_t",
        dot_type="q63_t",
        dot_suffix="q63",
        scalar=0x40000000,
        clip_lower=-0x40000000,
        clip_upper=-0x0CCCCCCD,
        value_tolerance=4,
        dot_tolerance=1 << 17,
    ),
    "BasicTestsQ15": BasicIntegerType(
        suffix="q15",
        bits=16,
        value_type="q15_t",
        value_signature="i16",
        unsigned_type="uint16_t",
        dot_type="q63_t",
        dot_suffix="q63",
        scalar=0x4000,
        clip_lower=-0x4000,
        clip_upper=-0x0CCD,
        value_tolerance=2,
        dot_tolerance=1 << 17,
    ),
    "BasicTestsQ7": BasicIntegerType(
        suffix="q7",
        bits=8,
        value_type="q7_t",
        value_signature="i8",
        unsigned_type="uint8_t",
        dot_type="q31_t",
        dot_suffix="q31",
        scalar=0x40,
        clip_lower=-0x40,
        clip_upper=-0x0D,
        value_tolerance=2,
        dot_tolerance=1 << 15,
    ),
}
_BASIC_INTEGER_OPERATIONS = {
    "add": ("binary", "Reference1_{suffix}.txt"),
    "sub": ("binary", "Reference2_{suffix}.txt"),
    "mult": ("binary", "Reference3_{suffix}.txt"),
    "negate": ("unary", "Reference4_{suffix}.txt"),
    "offset": ("scalar", "Reference5_{suffix}.txt"),
    "scale": ("scale", "Reference6_{suffix}.txt"),
    "dot_prod": ("dot", "Reference7_{dot_suffix}.txt"),
    "abs": ("unary", "Reference10_{suffix}.txt"),
    "shift": ("shift", "Shift21_{suffix}.txt"),
    "and": ("bit_binary", "And24_s{bits}.txt"),
    "or": ("bit_binary", "Or25_s{bits}.txt"),
    "not": ("bit_unary", "Not26_s{bits}.txt"),
    "xor": ("bit_binary", "Xor27_s{bits}.txt"),
    "clip": ("clip", "Reference28_{suffix}.txt"),
}
_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_WINDOW_TYPES = {
    "WindowTestsF32": WindowType(
        test_class="WindowTestsF32",
        suffix="f32",
        value_type="float32_t",
        absolute_error="2.0e-6f",
        relative_error="1.0e-6f",
    ),
    "WindowTestsF64": WindowType(
        test_class="WindowTestsF64",
        suffix="f64",
        value_type="float64_t",
        absolute_error="3.0e-15",
        relative_error="3.0e-15",
    ),
}
_ELEMENTARY_MATH_F32 = ElementaryMathType(
    suffix="f32",
    value_type="float32_t",
    absolute_error="1.0e-5f",
    relative_error="1.0e-6f",
)
_ELEMENTARY_MATH_F64 = ElementaryMathType(
    suffix="f64",
    value_type="float64_t",
    absolute_error="2.0e-16",
    relative_error="2.0e-16",
)
_ELEMENTARY_MATH_PROTOCOLS = {
    ("FastMathF32", "test_sqrt_f32"): ElementaryMathProtocol(
        type_spec=_ELEMENTARY_MATH_F32,
        kind="sqrt-status",
        symbol="arm_sqrt_f32",
        signature="i32(float,ptr)",
        input_a="SqrtInput1_f32.txt",
        expected="Sqrt1_f32.txt",
    ),
    ("FastMathF32", "test_vexp_f32"): ElementaryMathProtocol(
        type_spec=_ELEMENTARY_MATH_F32,
        kind="vector",
        symbol="arm_vexp_f32",
        signature="void(ptr,ptr,i32)",
        input_a="ExpInput1_f32.txt",
        expected="Exp1_f32.txt",
    ),
    ("FastMathF32", "test_vlog_f32"): ElementaryMathProtocol(
        type_spec=_ELEMENTARY_MATH_F32,
        kind="vector",
        symbol="arm_vlog_f32",
        signature="void(ptr,ptr,i32)",
        input_a="LogInput1_f32.txt",
        expected="Log1_f32.txt",
    ),
    ("FastMathF64", "test_vexp_f64"): ElementaryMathProtocol(
        type_spec=_ELEMENTARY_MATH_F64,
        kind="vector",
        symbol="arm_vexp_f64",
        signature="void(ptr,ptr,i32)",
        input_a="ExpInput1_f64.txt",
        expected="Exp1_f64.txt",
    ),
    ("FastMathF64", "test_vlog_f64"): ElementaryMathProtocol(
        type_spec=_ELEMENTARY_MATH_F64,
        kind="vector",
        symbol="arm_vlog_f64",
        signature="void(ptr,ptr,i32)",
        input_a="LogInput1_f64.txt",
        expected="Log1_f64.txt",
    ),
    ("StatsTestsF32", "test_entropy_f32"): ElementaryMathProtocol(
        type_spec=ElementaryMathType(
            suffix="f32",
            value_type="float32_t",
            absolute_error="0.0f",
            relative_error="1.0e-5f",
        ),
        kind="reduction",
        symbol="arm_entropy_f32",
        signature="float(ptr,i32)",
        input_a="Input22_f32.txt",
        dimensions="Dims22_s16.txt",
        expected="RefEntropy22_f32.txt",
    ),
    ("StatsTestsF64", "test_entropy_f64"): ElementaryMathProtocol(
        type_spec=ElementaryMathType(
            suffix="f64",
            value_type="float64_t",
            absolute_error="0.0",
            relative_error="4.0e-15",
        ),
        kind="reduction",
        symbol="arm_entropy_f64",
        signature="double(ptr,i32)",
        input_a="Input22_f64.txt",
        dimensions="Dims22_s16.txt",
        expected="RefEntropy22_f64.txt",
    ),
    ("StatsTestsF32", "test_kullback_leibler_f32"): ElementaryMathProtocol(
        type_spec=ElementaryMathType(
            suffix="f32",
            value_type="float32_t",
            absolute_error="0.0f",
            relative_error="1.0e-5f",
        ),
        kind="binary-reduction",
        symbol="arm_kullback_leibler_f32",
        signature="float(ptr,ptr,i32)",
        input_a="InputA24_f32.txt",
        input_b="InputB24_f32.txt",
        dimensions="Dims24_s16.txt",
        expected="RefKL24_f32.txt",
    ),
    ("StatsTestsF64", "test_kullback_leibler_f64"): ElementaryMathProtocol(
        type_spec=ElementaryMathType(
            suffix="f64",
            value_type="float64_t",
            absolute_error="0.0",
            relative_error="4.0e-15",
        ),
        kind="binary-reduction",
        symbol="arm_kullback_leibler_f64",
        signature="double(ptr,ptr,i32)",
        input_a="InputA24_f64.txt",
        input_b="InputB24_f64.txt",
        dimensions="Dims24_s16.txt",
        expected="RefKL24_f64.txt",
    ),
}


def basic_integer_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> tuple[BasicIntegerType, str, str, str] | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        return None
    type_spec = _BASIC_INTEGER_TYPES.get(producer.test_class)
    if type_spec is None or producer.selector_kind != "official":
        return None
    for operation, (kind, reference) in _BASIC_INTEGER_OPERATIONS.items():
        suffix = f"u{type_spec.bits}" if kind.startswith("bit_") else type_spec.suffix
        if producer.test_method != f"test_{operation}_{suffix}":
            continue
        symbol = f"arm_{operation}_{suffix}"
        if kind in {"binary", "bit_binary"}:
            signature = "void(ptr,ptr,ptr,i32)"
        elif kind in {"unary", "bit_unary"}:
            signature = "void(ptr,ptr,i32)"
        elif kind == "scalar":
            signature = f"void(ptr,{type_spec.value_signature},ptr,i32)"
        elif kind == "scale":
            signature = f"void(ptr,{type_spec.value_signature},i8,ptr,i32)"
        elif kind == "dot":
            signature = "void(ptr,ptr,i32,ptr)"
        elif kind == "shift":
            signature = "void(ptr,i8,ptr,i32)"
        else:
            signature = (
                f"void(ptr,ptr,{type_spec.value_signature},"
                f"{type_spec.value_signature},i32)"
            )
        if tuple((call.symbol, call.signature) for call in workload.protocol) != (
            (symbol, signature),
        ):
            return None
        return type_spec, operation, kind, reference
    return None


def window_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> WindowType | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        return None
    type_spec = _WINDOW_TYPES.get(producer.test_class)
    if (
        type_spec is None
        or producer.selector_kind != "official"
        or producer.vector_ordinal != 0
        or not producer.test_method.startswith("test_")
    ):
        return None
    symbol = "arm_" + producer.test_method.removeprefix("test_")
    if _C_IDENTIFIER.fullmatch(symbol) is None:
        return None
    if tuple((call.symbol, call.signature) for call in workload.protocol) != (
        (symbol, "void(ptr,i32)"),
    ):
        return None
    return type_spec


def elementary_math_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> ElementaryMathProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        return None
    if producer.selector_kind != "official" or producer.vector_ordinal != 0:
        return None
    protocol = _ELEMENTARY_MATH_PROTOCOLS.get(
        (producer.test_class, producer.test_method)
    )
    if protocol is None or workload.compiler_flags != ("-fno-math-errno",):
        return None
    if tuple((call.symbol, call.signature) for call in workload.protocol) != (
        (protocol.symbol, protocol.signature),
    ):
        return None
    return protocol


def window_reference_name(
    suite: object,
    test_kind: int,
    workload: corpus_inventory.ProgramWorkload,
) -> str:
    producer = workload.producer
    type_spec = window_protocol(workload)
    if type_spec is None or not isinstance(
        producer, corpus_inventory.CmsisDspWorkloadProducer
    ):
        raise WorkloadProviderError("CMSIS-DSP window workload owner is invalid")
    tests = [child for child in suite.children if child.kind == test_kind]
    matching = [
        index
        for index, child in enumerate(tests)
        if child.data.get("class") == producer.test_method
    ]
    patterns = tuple(suite.patterns)
    if len(matching) != 1 or len(patterns) != len(tests):
        raise WorkloadProviderError(
            "CMSIS-DSP window descriptor does not define a total test/reference map"
        )
    ordinal = matching[0]
    pattern_id, reference_name = patterns[ordinal]
    if (
        not isinstance(pattern_id, str)
        or not pattern_id.startswith(f"REF{ordinal + 1}_")
        or not isinstance(reference_name, str)
        or not reference_name.endswith(f"_{ordinal + 1}_{type_spec.suffix}.txt")
    ):
        raise WorkloadProviderError(
            "CMSIS-DSP window descriptor test/reference order is noncanonical"
        )
    return reference_name


def source_payload_size(path: Path) -> int:
    sample_sizes = {"B": 1, "H": 2, "W": 4, "D": 8}
    try:
        with path.open(encoding="utf-8", errors="strict") as source:
            first = source.readline().strip()
            if first in sample_sizes:
                sample_size = sample_sizes[first]
                count_text = source.readline().strip()
            else:
                sample_size = 4
                count_text = first
    except (OSError, UnicodeError) as exc:
        raise WorkloadProviderError(
            f"cannot read CMSIS-DSP pattern source {path}: {exc}"
        ) from exc
    try:
        count = int(count_text, 10)
    except ValueError as exc:
        raise WorkloadProviderError(
            f"CMSIS-DSP pattern source has an invalid count: {path}"
        ) from exc
    if count < 0:
        raise WorkloadProviderError(
            f"CMSIS-DSP pattern source has a negative count: {path}"
        )
    return sample_size * count


def pattern_segments(path: Path) -> dict[str, bytes]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise WorkloadProviderError(
            f"cannot read generated CMSIS-DSP patterns {path}: {exc}"
        ) from exc
    match = re.search(r"const\s+char\s+patterns\[\]\s*=\s*\{(.*?)\};", text, re.S)
    if match is None:
        raise WorkloadProviderError(
            f"generated CMSIS-DSP patterns have no byte array: {path}"
        )

    segments: dict[str, bytearray] = {}
    current: bytearray | None = None
    current_source: Path | None = None

    def finish_segment() -> None:
        if current is None or current_source is None:
            return
        if not current:
            return
        name = current_source.name
        if name in segments:
            raise WorkloadProviderError(
                f"generated CMSIS-DSP patterns repeat {name}: {path}"
            )
        payload_size = source_payload_size(current_source)
        padded_size = (payload_size + 7) & ~7
        if len(current) != padded_size or any(current[payload_size:]):
            raise WorkloadProviderError(
                "generated CMSIS-DSP pattern does not match its source extent: "
                f"{current_source}"
            )
        del current[payload_size:]
        segments[name] = current

    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("// "):
            finish_segment()
            candidate_source = Path(stripped[3:])
            if candidate_source.is_dir():
                current = None
                current_source = None
                continue
            current_source = candidate_source
            current = bytearray()
            continue
        for token in stripped.split(","):
            token = token.strip()
            if not token:
                continue
            if current is None:
                raise WorkloadProviderError(
                    f"generated CMSIS-DSP pattern bytes have no owner: {path}"
                )
            try:
                value = int(token, 10)
            except ValueError as exc:
                raise WorkloadProviderError(
                    f"generated CMSIS-DSP pattern byte is not decimal: {path}"
                ) from exc
            if value < 0 or value > 255:
                raise WorkloadProviderError(
                    f"generated CMSIS-DSP pattern byte is invalid: {path}"
                )
            current.append(value)
    finish_segment()
    if not segments:
        raise WorkloadProviderError(f"generated CMSIS-DSP patterns are empty: {path}")
    return {name: bytes(value) for name, value in segments.items()}


def pattern_bytes(path: Path) -> bytes:
    return b"".join(pattern_segments(path).values())


def f32_literal(raw: bytes) -> str:
    value = struct.unpack("<f", raw)[0]
    if not math.isfinite(value):
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol requires finite float input"
        )
    return f"{value.hex()}f"


def scalar_literals(
    pattern_bytes_value: bytes, scalar: str, sample_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    byte_count = sample_count * 2 * 4
    if len(pattern_bytes_value) < byte_count:
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol pattern is smaller than its input extent"
        )
    chunks = tuple(
        pattern_bytes_value[offset : offset + 4] for offset in range(0, byte_count, 4)
    )
    if scalar == "float32_t":
        values = tuple(f32_literal(chunk) for chunk in chunks)
    else:
        values = tuple(
            str(int.from_bytes(chunk, byteorder="little", signed=True))
            for chunk in chunks
        )
    return values[:sample_count], values[sample_count:]


def format_cpp_array(values: Sequence[str]) -> str:
    lines = [
        "  " + ", ".join(values[index : index + 4])
        for index in range(0, len(values), 4)
    ]
    return ",\n".join(lines)


def decode_f32_pattern(raw: bytes, name: str) -> tuple[str, ...]:
    if len(raw) % 4 != 0:
        raise WorkloadProviderError(f"CMSIS-DSP {name} is not f32-aligned")
    return tuple(
        f32_literal(raw[offset : offset + 4]) for offset in range(0, len(raw), 4)
    )


def decode_f16_pattern(raw: bytes, name: str) -> tuple[str, ...]:
    if len(raw) % 2 != 0:
        raise WorkloadProviderError(f"CMSIS-DSP {name} is not f16-aligned")
    values = tuple(
        struct.unpack("<e", raw[offset : offset + 2])[0]
        for offset in range(0, len(raw), 2)
    )
    if any(not math.isfinite(value) for value in values):
        raise WorkloadProviderError(f"CMSIS-DSP {name} requires finite float input")
    return tuple(f"static_cast<float16_t>({value.hex()}f)" for value in values)


def decode_i16_pattern(raw: bytes, name: str) -> tuple[int, ...]:
    if len(raw) % 2 != 0:
        raise WorkloadProviderError(f"CMSIS-DSP {name} is not i16-aligned")
    return tuple(
        int.from_bytes(raw[offset : offset + 2], byteorder="little", signed=True)
        for offset in range(0, len(raw), 2)
    )


def decode_f64_pattern(raw: bytes, name: str) -> tuple[str, ...]:
    if len(raw) % 8 != 0:
        raise WorkloadProviderError(f"CMSIS-DSP {name} is not f64-aligned")
    values = tuple(
        struct.unpack("<d", raw[offset : offset + 8])[0]
        for offset in range(0, len(raw), 8)
    )
    if any(not math.isfinite(value) for value in values):
        raise WorkloadProviderError(f"CMSIS-DSP {name} requires finite float input")
    return tuple(value.hex() for value in values)


def decode_integer_pattern(
    raw: bytes, bits: int, signed: bool, name: str
) -> tuple[str, ...]:
    byte_width = bits // 8
    if bits not in {8, 16, 32, 64} or len(raw) % byte_width != 0:
        raise WorkloadProviderError(
            f"CMSIS-DSP {name} is not aligned to its integer type"
        )
    values = (
        int.from_bytes(
            raw[offset : offset + byte_width], byteorder="little", signed=signed
        )
        for offset in range(0, len(raw), byte_width)
    )
    suffix = "" if signed else "u"
    return tuple(f"{value}{suffix}" for value in values)


def require_pattern_segment(segments: dict[str, bytes], name: str) -> bytes:
    try:
        value = segments[name]
    except KeyError as exc:
        raise WorkloadProviderError(
            f"generated CMSIS-DSP patterns omit {name}"
        ) from exc
    if not value:
        raise WorkloadProviderError(f"generated CMSIS-DSP pattern {name} is empty")
    return value


def render_stateless_abs_f32_protocol(
    patterns: Path, sample_count: int, protocol_symbol: str
) -> str:
    segments = pattern_segments(patterns)
    inputs = decode_f32_pattern(
        require_pattern_segment(segments, "Input1_f32.txt"),
        "absolute input",
    )
    expected = decode_f32_pattern(
        require_pattern_segment(segments, "Reference10_f32.txt"),
        "absolute reference",
    )
    if sample_count <= 0 or len(inputs) < sample_count or len(expected) < sample_count:
        raise WorkloadProviderError(
            "CMSIS-DSP absolute pattern does not cover its workload extent"
        )
    inputs = inputs[:sample_count]
    expected = expected[:sample_count]

    return f"""#include <cstddef>
#include <cstdint>
#include <cstring>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kSampleCount = {sample_count};
constexpr float32_t kInput[] = {{
{format_cpp_array(inputs)}
}};
constexpr float32_t kExpected[] = {{
{format_cpp_array(expected)}
}};

bool oracle_matches(const float32_t *output) {{
  return std::memcmp(output, kExpected, sizeof(kExpected)) == 0;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const float32_t *input, float32_t *output, std::uint32_t count) {{
  arm_abs_f32(input, output, count);
}}

int main() {{
  float32_t input[kSampleCount];
  float32_t output[kSampleCount]{{}};
  for (std::uint32_t index = 0; index < kSampleCount; ++index)
    input[index] = kInput[index];
  {protocol_symbol}(input, output, kSampleCount);
  return oracle_matches(output) ? 0 : 1;
}}
"""


def render_basic_integer_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    sample_count: int | None,
    protocol_symbol: str,
) -> str:
    protocol = basic_integer_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError("unsupported CMSIS-DSP basic integer protocol")
    type_spec, operation, kind, reference_template = protocol
    call = workload.protocol[0]
    format_values = {
        "suffix": type_spec.suffix,
        "bits": type_spec.bits,
        "dot_suffix": type_spec.dot_suffix,
    }
    reference_name = reference_template.format(**format_values)
    if kind.startswith("bit_"):
        input_name = f"BitwiseInput24_s{type_spec.bits}.txt"
        input_signed = False
        input_type = type_spec.unsigned_type
    elif kind == "shift":
        input_name = f"Input12_{type_spec.suffix}.txt"
        input_signed = True
        input_type = type_spec.value_type
    elif kind == "clip":
        input_name = f"Input28_{type_spec.suffix}.txt"
        input_signed = True
        input_type = type_spec.value_type
    else:
        input_name = f"Input1_{type_spec.suffix}.txt"
        input_signed = True
        input_type = type_spec.value_type

    segments = pattern_segments(patterns)
    input_a = decode_integer_pattern(
        require_pattern_segment(segments, input_name),
        type_spec.bits,
        input_signed,
        "basic integer input",
    )
    output_bits = (
        64
        if kind == "dot" and type_spec.dot_type == "q63_t"
        else 32
        if kind == "dot"
        else type_spec.bits
    )
    output_signed = not kind.startswith("bit_")
    output_type = (
        type_spec.dot_type
        if kind == "dot"
        else type_spec.unsigned_type
        if kind.startswith("bit_")
        else type_spec.value_type
    )
    expected = decode_integer_pattern(
        require_pattern_segment(segments, reference_name),
        output_bits,
        output_signed,
        "basic integer reference",
    )
    if sample_count is None:
        sample_count = len(expected)
    if sample_count <= 0 or len(input_a) < sample_count:
        raise WorkloadProviderError(
            "CMSIS-DSP basic integer input does not cover its workload extent"
        )
    input_a = input_a[:sample_count]

    input_b: tuple[str, ...] = ()
    if kind in {"binary", "dot", "bit_binary"}:
        second_name = (
            f"BitwiseInput25_s{type_spec.bits}.txt"
            if kind == "bit_binary"
            else f"Input2_{type_spec.suffix}.txt"
        )
        input_b = decode_integer_pattern(
            require_pattern_segment(segments, second_name),
            type_spec.bits,
            input_signed,
            "basic integer second input",
        )
        if len(input_b) < sample_count:
            raise WorkloadProviderError(
                "CMSIS-DSP basic integer second input is incomplete"
            )
        input_b = input_b[:sample_count]

    output_count = 1 if kind == "dot" else sample_count
    if len(expected) < output_count:
        raise WorkloadProviderError("CMSIS-DSP basic integer reference is incomplete")
    expected = expected[:output_count]

    if kind in {"binary", "bit_binary"}:
        parameters = f"""const {input_type} *input_a, const {input_type} *input_b,
    {output_type} *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input_a, input_b, output, count);"
        main_arguments = "input_a, input_b, output, kSampleCount"
    elif kind in {"unary", "bit_unary"}:
        parameters = (
            f"const {input_type} *input, {output_type} *output, std::uint32_t count"
        )
        invocation = f"{call.symbol}(input, output, count);"
        main_arguments = "input_a, output, kSampleCount"
    elif kind == "scalar":
        parameters = f"""const {input_type} *input, {input_type} scalar,
    {output_type} *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input, scalar, output, count);"
        main_arguments = f"input_a, {type_spec.scalar}, output, kSampleCount"
    elif kind == "scale":
        parameters = f"""const {input_type} *input, {input_type} scalar,
    std::int8_t shift, {output_type} *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input, scalar, shift, output, count);"
        main_arguments = f"input_a, {type_spec.scalar}, 0, output, kSampleCount"
    elif kind == "dot":
        parameters = f"""const {input_type} *input_a, const {input_type} *input_b,
    {output_type} *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input_a, input_b, count, &output[0]);"
        main_arguments = "input_a, input_b, output, kSampleCount"
    elif kind == "shift":
        parameters = f"""const {input_type} *input, std::int8_t shift,
    {output_type} *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input, shift, output, count);"
        main_arguments = "input_a, 1, output, kSampleCount"
    else:
        parameters = f"""const {input_type} *input, {output_type} *output,
    {input_type} lower, {input_type} upper, std::uint32_t count"""
        invocation = f"{call.symbol}(input, output, lower, upper, count);"
        main_arguments = (
            f"input_a, output, {type_spec.clip_lower}, "
            f"{type_spec.clip_upper}, kSampleCount"
        )

    input_b_declaration = ""
    input_b_initialization = ""
    if input_b:
        input_b_declaration = f"""constexpr {input_type} kInputB[] = {{
{format_cpp_array(input_b)}
}};
"""
        input_b_initialization = f"""
  {input_type} input_b[kSampleCount];
  for (std::uint32_t index = 0; index < kSampleCount; ++index)
    input_b[index] = kInputB[index];"""

    if kind.startswith("bit_"):
        oracle_support = ""
        oracle_body = "return std::memcmp(output, kExpected, sizeof(kExpected)) == 0;"
    else:
        tolerance = (
            type_spec.dot_tolerance if kind == "dot" else type_spec.value_tolerance
        )
        oracle_support = f"""
template <typename T>
bool within_absolute_error(
    T actual, T expected, std::make_unsigned_t<T> tolerance) {{
  using U = std::make_unsigned_t<T>;
  const U actual_bits = static_cast<U>(actual);
  const U expected_bits = static_cast<U>(expected);
  const bool actual_negative = actual < 0;
  const bool expected_negative = expected < 0;
  U distance;
  if (actual_negative == expected_negative) {{
    distance = actual >= expected ? actual_bits - expected_bits
                                  : expected_bits - actual_bits;
  }} else {{
    const U actual_magnitude =
        actual_negative ? U{{0}} - actual_bits : actual_bits;
    const U expected_magnitude =
        expected_negative ? U{{0}} - expected_bits : expected_bits;
    if (actual_magnitude >
        std::numeric_limits<U>::max() - expected_magnitude)
      return false;
    distance = actual_magnitude + expected_magnitude;
  }}
  return distance <= tolerance;
}}

constexpr std::make_unsigned_t<{output_type}> kAbsoluteError = {tolerance};
"""
        oracle_body = """for (std::size_t index = 0; index < kOutputCount; ++index) {
    if (!within_absolute_error(output[index], kExpected[index], kAbsoluteError))
      return false;
  }
  return true;"""

    return f"""#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kSampleCount = {sample_count};
constexpr std::size_t kOutputCount = {output_count};
constexpr {input_type} kInputA[] = {{
{format_cpp_array(input_a)}
}};
{input_b_declaration}constexpr {output_type} kExpected[] = {{
{format_cpp_array(expected)}
}};
{oracle_support}

bool oracle_matches(const {output_type} *output) {{
  {oracle_body}
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    {parameters}) {{
  {invocation}
}}

int main() {{
  {input_type} input_a[kSampleCount];
  {output_type} output[kOutputCount]{{}};
  for (std::uint32_t index = 0; index < kSampleCount; ++index)
    input_a[index] = kInputA[index];{input_b_initialization}
  {protocol_symbol}({main_arguments});
  return oracle_matches(output) ? 0 : 1;
}}
"""


def render_stateless_controller_protocol(
    workload: corpus_inventory.ProgramWorkload,
    pattern_bytes_value: bytes,
    sample_count: int,
    protocol_symbol: str,
) -> str:
    call = workload.protocol[0]
    if _C_IDENTIFIER.fullmatch(call.symbol) is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP protocol symbol is not a C identifier: {call.symbol}"
        )
    is_float = call.signature.startswith("void(float")
    scalar = "float32_t" if is_float else "q31_t"
    input_a, input_b = scalar_literals(pattern_bytes_value, scalar, sample_count)

    if call.signature in {"void(float,float,ptr,ptr)", "void(i32,i32,ptr,ptr)"}:
        wrapper_parameters = f"""const {scalar} *input_a,
    const {scalar} *input_b, {scalar} *output_a, {scalar} *output_b,
    std::uint32_t count"""
        call_arguments = (
            "input_a[index], input_b[index], &output_a[index], &output_b[index]"
        )
        main_arguments = "kInputA, kInputB, output_a, output_b, kSampleCount"
    elif call.signature in {
        "void(float,float,ptr,ptr,float,float)",
        "void(i32,i32,ptr,ptr,i32,i32)",
    }:
        wrapper_parameters = f"""const {scalar} *input_a,
    const {scalar} *input_b, {scalar} *output_a, {scalar} *output_b,
    std::uint32_t count, {scalar} coefficient_a,
    {scalar} coefficient_b"""
        call_arguments = (
            "input_a[index], input_b[index], &output_a[index], &output_b[index], "
            "coefficient_a, coefficient_b"
        )
        main_arguments = (
            "kInputA, kInputB, output_a, output_b, kSampleCount, kInputA[0], kInputB[0]"
        )
    elif call.signature in {"void(float,ptr,ptr)", "void(i32,ptr,ptr)"}:
        wrapper_parameters = f"""const {scalar} *input,
    {scalar} *output_a, {scalar} *output_b, std::uint32_t count"""
        call_arguments = "input[index], &output_a[index], &output_b[index]"
        main_arguments = "kInputA, output_a, output_b, kSampleCount"
    else:
        raise WorkloadProviderError(
            f"unsupported stateless controller signature: {call.signature}"
        )

    return f"""#include <cstddef>
#include <cstdint>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kSampleCount = {sample_count};
constexpr {scalar} kInputA[] = {{
{format_cpp_array(input_a)}
}};
constexpr {scalar} kInputB[] = {{
{format_cpp_array(input_b)}
}};

std::uint32_t digest(const void *data, std::size_t size) {{
  const auto *bytes = static_cast<const unsigned char *>(data);
  std::uint32_t value = 2166136261u;
  for (std::size_t index = 0; index < size; ++index) {{
    value ^= bytes[index];
    value *= 16777619u;
  }}
  return value;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    {wrapper_parameters}) {{
  for (std::uint32_t index = 0; index < count; ++index) {{
    {call.symbol}({call_arguments});
  }}
}}

int main() {{
  {scalar} output_a[kSampleCount]{{}};
  {scalar} output_b[kSampleCount]{{}};
  {protocol_symbol}({main_arguments});
  const std::uint32_t first = digest(output_a, sizeof(output_a));
  const std::uint32_t second = digest(output_b, sizeof(output_b));
  return static_cast<int>(first ^ (second * 16777619u));
}}
"""


def render_window_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    reference_name: str,
    protocol_symbol: str,
) -> str:
    type_spec = window_protocol(workload)
    if type_spec is None:
        raise WorkloadProviderError("unsupported CMSIS-DSP window protocol")
    segments = pattern_segments(patterns)
    raw_reference = require_pattern_segment(segments, reference_name)
    expected = (
        decode_f32_pattern(raw_reference, "window reference")
        if type_spec.suffix == "f32"
        else decode_f64_pattern(raw_reference, "window reference")
    )
    if not expected:
        raise WorkloadProviderError("CMSIS-DSP window reference is empty")
    call = workload.protocol[0]

    return f"""#include <cmath>
#include <cstddef>
#include <cstdint>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kSampleCount = {len(expected)};
constexpr {type_spec.value_type} kExpected[] = {{
{format_cpp_array(expected)}
}};
constexpr {type_spec.value_type} kAbsoluteError = {type_spec.absolute_error};
constexpr {type_spec.value_type} kRelativeError = {type_spec.relative_error};

bool oracle_matches(const {type_spec.value_type} *output) {{
  for (std::size_t index = 0; index < kSampleCount; ++index) {{
    const {type_spec.value_type} expected = kExpected[index];
    const {type_spec.value_type} difference = std::fabs(output[index] - expected);
    const {type_spec.value_type} magnitude = std::fabs(expected);
    if (!std::isfinite(output[index]) ||
        difference > kAbsoluteError + kRelativeError * magnitude)
      return false;
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    {type_spec.value_type} *output, std::uint32_t count) {{
  {call.symbol}(output, count);
}}

int main() {{
  {type_spec.value_type} output[kSampleCount]{{}};
  {protocol_symbol}(output, kSampleCount);
  return oracle_matches(output) ? 0 : 1;
}}
"""


def _decode_elementary_float_pattern(
    raw: bytes, type_spec: ElementaryMathType, name: str
) -> tuple[str, ...]:
    if type_spec.suffix == "f32":
        return decode_f32_pattern(raw, name)
    if type_spec.suffix == "f64":
        return decode_f64_pattern(raw, name)
    raise WorkloadProviderError(
        f"CMSIS-DSP elementary math type is unsupported: {type_spec.suffix}"
    )


def _elementary_dimensions(
    segments: dict[str, bytes], protocol: ElementaryMathProtocol, input_count: int
) -> tuple[int, ...]:
    if protocol.dimensions is None:
        raise WorkloadProviderError(
            "CMSIS-DSP elementary reduction omits its dimensions"
        )
    dimensions = decode_i16_pattern(
        require_pattern_segment(segments, protocol.dimensions),
        "elementary math dimensions",
    )
    if not dimensions or dimensions[0] <= 0:
        raise WorkloadProviderError(
            "CMSIS-DSP elementary reduction has no workload vectors"
        )
    pattern_count = dimensions[0]
    extents = dimensions[1:]
    if len(extents) != pattern_count or any(extent <= 0 for extent in extents):
        raise WorkloadProviderError(
            "CMSIS-DSP elementary reduction dimensions are noncanonical"
        )
    if sum(extents) != input_count:
        raise WorkloadProviderError(
            "CMSIS-DSP elementary reduction dimensions do not cover the input"
        )
    return extents


def render_elementary_math_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    protocol_symbol: str,
) -> str:
    protocol = elementary_math_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError("unsupported CMSIS-DSP elementary math protocol")
    segments = pattern_segments(patterns)
    type_spec = protocol.type_spec
    input_a = _decode_elementary_float_pattern(
        require_pattern_segment(segments, protocol.input_a),
        type_spec,
        "elementary math input",
    )
    expected = _decode_elementary_float_pattern(
        require_pattern_segment(segments, protocol.expected),
        type_spec,
        "elementary math reference",
    )

    input_b: tuple[str, ...] = ()
    if protocol.input_b is not None:
        input_b = _decode_elementary_float_pattern(
            require_pattern_segment(segments, protocol.input_b),
            type_spec,
            "elementary math second input",
        )
        if len(input_b) != len(input_a):
            raise WorkloadProviderError(
                "CMSIS-DSP elementary math inputs have different extents"
            )

    dimensions: tuple[int, ...] = ()
    if protocol.kind in {"reduction", "binary-reduction"}:
        dimensions = _elementary_dimensions(segments, protocol, len(input_a))
        if len(expected) != len(dimensions):
            raise WorkloadProviderError(
                "CMSIS-DSP elementary reduction reference has the wrong extent"
            )
    elif len(input_a) != len(expected):
        raise WorkloadProviderError(
            "CMSIS-DSP elementary math input and reference extents differ"
        )

    input_b_declaration = ""
    input_b_initialization = ""
    if input_b:
        input_b_declaration = f"""constexpr {type_spec.value_type} kInputB[] = {{
{format_cpp_array(input_b)}
}};
"""
        input_b_initialization = f"""
  {type_spec.value_type} input_b[kInputCount];
  for (std::uint32_t index = 0; index < kInputCount; ++index)
    input_b[index] = kInputB[index];"""

    dimensions_declaration = ""
    if dimensions:
        dimensions_declaration = f"""constexpr std::uint32_t kDimensions[] = {{
{format_cpp_array(tuple(str(value) for value in dimensions))}
}};
"""

    status_parameter = ""
    status_declaration = ""
    status_argument = ""
    status_oracle = ""
    if protocol.kind == "sqrt-status":
        status_parameter = ", std::int32_t *status"
        status_declaration = "  std::int32_t status[kOutputCount]{};\n"
        status_argument = ", status"
        status_oracle = """
    const std::int32_t expected_status =
        kInputA[index] < 0.0f ? ARM_MATH_ARGUMENT_ERROR : ARM_MATH_SUCCESS;
    if (status[index] != expected_status)
      return false;"""

    if protocol.kind == "vector":
        wrapper_parameters = f"""const {type_spec.value_type} *input,
    {type_spec.value_type} *output, std::uint32_t count"""
        wrapper_body = f"{protocol.symbol}(input, output, count);"
        main_arguments = "input_a, output, kInputCount"
    elif protocol.kind == "sqrt-status":
        wrapper_parameters = f"""const {type_spec.value_type} *input,
    {type_spec.value_type} *output, std::int32_t *status,
    std::uint32_t count"""
        wrapper_body = f"""for (std::uint32_t index = 0; index < count; ++index) {{
    status[index] = static_cast<std::int32_t>(
        {protocol.symbol}(input[index], &output[index]));
  }}"""
        main_arguments = "input_a, output, status, kInputCount"
    elif protocol.kind == "reduction":
        wrapper_parameters = f"""const {type_spec.value_type} *input,
    const std::uint32_t *dimensions, {type_spec.value_type} *output,
    std::uint32_t pattern_count"""
        wrapper_body = f"""std::uint32_t offset = 0;
  for (std::uint32_t index = 0; index < pattern_count; ++index) {{
    output[index] = {protocol.symbol}(input + offset, dimensions[index]);
    offset += dimensions[index];
  }}"""
        main_arguments = "input_a, kDimensions, output, kOutputCount"
    elif protocol.kind == "binary-reduction":
        wrapper_parameters = f"""const {type_spec.value_type} *input_a,
    const {type_spec.value_type} *input_b,
    const std::uint32_t *dimensions, {type_spec.value_type} *output,
    std::uint32_t pattern_count"""
        wrapper_body = f"""std::uint32_t offset = 0;
  for (std::uint32_t index = 0; index < pattern_count; ++index) {{
    output[index] = {protocol.symbol}(input_a + offset, input_b + offset,
                                      dimensions[index]);
    offset += dimensions[index];
  }}"""
        main_arguments = "input_a, input_b, kDimensions, output, kOutputCount"
    else:
        raise WorkloadProviderError(
            f"unknown CMSIS-DSP elementary math protocol kind: {protocol.kind}"
        )

    return f"""#include <cmath>
#include <cstddef>
#include <cstdint>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kInputCount = {len(input_a)};
constexpr std::uint32_t kOutputCount = {len(expected)};
constexpr {type_spec.value_type} kInputA[] = {{
{format_cpp_array(input_a)}
}};
{input_b_declaration}{dimensions_declaration}constexpr {type_spec.value_type} kExpected[] = {{
{format_cpp_array(expected)}
}};
constexpr {type_spec.value_type} kAbsoluteError = {type_spec.absolute_error};
constexpr {type_spec.value_type} kRelativeError = {type_spec.relative_error};

bool oracle_matches(const {type_spec.value_type} *output{status_parameter}) {{
  for (std::uint32_t index = 0; index < kOutputCount; ++index) {{{status_oracle}
    const {type_spec.value_type} expected = kExpected[index];
    const {type_spec.value_type} difference = std::fabs(output[index] - expected);
    if (!std::isfinite(output[index]) ||
        difference > kAbsoluteError + kRelativeError * std::fabs(expected))
      return false;
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    {wrapper_parameters}) {{
  {wrapper_body}
}}

int main() {{
  {type_spec.value_type} input_a[kInputCount];
  {type_spec.value_type} output[kOutputCount]{{}};
  for (std::uint32_t index = 0; index < kInputCount; ++index)
    input_a[index] = kInputA[index];{input_b_initialization}
{status_declaration}  {protocol_symbol}({main_arguments});
  return oracle_matches(output{status_argument}) ? 0 : 1;
}}
"""


def render_basic_f32_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    sample_count: int | None,
    protocol_symbol: str,
) -> str:
    call = workload.protocol[0]
    specs = {
        "arm_add_f32": ("Reference1_f32.txt", "binary"),
        "arm_sub_f32": ("Reference2_f32.txt", "binary"),
        "arm_mult_f32": ("Reference3_f32.txt", "binary"),
        "arm_negate_f32": ("Reference4_f32.txt", "unary"),
        "arm_offset_f32": ("Reference5_f32.txt", "scalar"),
        "arm_scale_f32": ("Reference6_f32.txt", "scalar"),
        "arm_dot_prod_f32": ("Reference7_f32.txt", "dot"),
        "arm_clip_f32": ("Reference12_f32.txt", "clip"),
    }
    try:
        reference_name, kind = specs[call.symbol]
    except KeyError as exc:
        raise WorkloadProviderError(
            f"unsupported CMSIS-DSP BasicTestsF32 protocol: {call.symbol}"
        ) from exc
    segments = pattern_segments(patterns)
    input_name = "Input12_f32.txt" if kind == "clip" else "Input1_f32.txt"
    input_a = decode_f32_pattern(
        require_pattern_segment(segments, input_name),
        "basic f32 input",
    )
    expected = decode_f32_pattern(
        require_pattern_segment(segments, reference_name),
        "basic f32 reference",
    )
    if sample_count is None:
        sample_count = len(expected)
    if sample_count <= 0 or len(input_a) < sample_count:
        raise WorkloadProviderError(
            "CMSIS-DSP basic f32 input does not cover its workload extent"
        )
    input_a = input_a[:sample_count]
    input_b: tuple[str, ...] = ()
    if kind in {"binary", "dot"}:
        input_b = decode_f32_pattern(
            require_pattern_segment(segments, "Input2_f32.txt"),
            "basic f32 second input",
        )
        if len(input_b) < sample_count:
            raise WorkloadProviderError(
                "CMSIS-DSP basic f32 second input is incomplete"
            )
        input_b = input_b[:sample_count]
    output_count = len(expected) if kind == "dot" else sample_count
    if len(expected) < output_count:
        raise WorkloadProviderError("CMSIS-DSP basic f32 reference is incomplete")
    expected = expected[:output_count]

    if kind == "binary":
        parameters = """const float32_t *input_a, const float32_t *input_b,
    float32_t *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input_a, input_b, output, count);"
        main_arguments = "input_a, input_b, output, kSampleCount"
    elif kind == "unary":
        parameters = "const float32_t *input, float32_t *output, std::uint32_t count"
        invocation = f"{call.symbol}(input, output, count);"
        main_arguments = "input_a, output, kSampleCount"
    elif kind == "scalar":
        parameters = """const float32_t *input, float32_t scalar,
    float32_t *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input, scalar, output, count);"
        main_arguments = "input_a, 0.5f, output, kSampleCount"
    elif kind == "dot":
        parameters = """const float32_t *input_a, const float32_t *input_b,
    float32_t *output, std::uint32_t count"""
        invocation = f"{call.symbol}(input_a, input_b, count, &output[0]);"
        main_arguments = "input_a, input_b, output, kSampleCount"
    else:
        parameters = """const float32_t *input, float32_t *output,
    float32_t lower, float32_t upper, std::uint32_t count"""
        invocation = f"{call.symbol}(input, output, lower, upper, count);"
        main_arguments = "input_a, output, -0.5f, -0.1f, kSampleCount"

    input_b_declaration = ""
    input_b_initialization = ""
    if input_b:
        input_b_declaration = f"""constexpr float32_t kInputB[] = {{
{format_cpp_array(input_b)}
}};
"""
        input_b_initialization = """
  float32_t input_b[kSampleCount];
  for (std::uint32_t index = 0; index < kSampleCount; ++index)
    input_b[index] = kInputB[index];"""

    return f"""#include <cmath>
#include <cstddef>
#include <cstdint>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kSampleCount = {sample_count};
constexpr std::size_t kOutputCount = {output_count};
constexpr float32_t kInputA[] = {{
{format_cpp_array(input_a)}
}};
{input_b_declaration}constexpr float32_t kExpected[] = {{
{format_cpp_array(expected)}
}};

bool oracle_matches(const float32_t *output) {{
  for (std::size_t index = 0; index < kOutputCount; ++index) {{
    const float32_t expected = kExpected[index];
    const float32_t difference = std::fabs(output[index] - expected);
    const float32_t magnitude = std::fabs(expected);
    if (!std::isfinite(output[index]) ||
        difference > 1.0e-6f + 5.0e-5f * magnitude)
      return false;
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    {parameters}) {{
  {invocation}
}}

int main() {{
  float32_t input_a[kSampleCount];
  float32_t output[kOutputCount]{{}};
  for (std::uint32_t index = 0; index < kSampleCount; ++index)
    input_a[index] = kInputA[index];{input_b_initialization}
  {protocol_symbol}({main_arguments});
  return oracle_matches(output) ? 0 : 1;
}}
"""
