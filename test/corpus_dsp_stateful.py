"""Typed stateful CMSIS-DSP operator protocols."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import corpus_dsp_protocol
import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class FirProtocol:
    test_class: str
    test_method: str
    suffix: str
    value_type: str
    bits: int | None
    init_signature: str
    owner_header: str
    absolute_error: str | None = None
    relative_error: str | None = None
    integer_error: int | None = None

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        return (
            (f"arm_fir_init_{self.suffix}", self.init_signature),
            (f"arm_fir_{self.suffix}", "void(ptr,ptr,ptr,i32)"),
        )


@dataclass(frozen=True)
class SvmValueType:
    test_class: str
    suffix: str
    value_type: str
    scalar_signature: str
    owner_header: str


@dataclass(frozen=True)
class SvmKernel:
    name: str
    pattern_ordinal: int
    scalar_parameter_count: int
    has_degree: bool = False


@dataclass(frozen=True)
class SvmProtocol:
    value: SvmValueType
    kernel: SvmKernel

    @property
    def test_method(self) -> str:
        return f"test_svm_{self.kernel.name}_predict_{self.value.suffix}"

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        tail = ""
        if self.kernel.has_degree:
            tail += ",i32"
        tail += f",{self.value.scalar_signature}" * self.kernel.scalar_parameter_count
        return (
            (
                f"arm_svm_{self.kernel.name}_init_{self.value.suffix}",
                f"void(ptr,i32,i32,{self.value.scalar_signature},ptr,ptr,ptr{tail})",
            ),
            (
                f"arm_svm_{self.kernel.name}_predict_{self.value.suffix}",
                "void(ptr,ptr,ptr)",
            ),
        )


_FIR_PROTOCOLS = {
    ("FIRF16", "test_fir_f16"): FirProtocol(
        test_class="FIRF16",
        test_method="test_fir_f16",
        suffix="f16",
        value_type="float16_t",
        bits=None,
        init_signature="void(ptr,i16,ptr,ptr,i32)",
        owner_header="filtering_functions_f16.h",
        absolute_error="1.0e-3f",
        relative_error="2.0e-2f",
    ),
    ("FIRF32", "test_fir_f32"): FirProtocol(
        test_class="FIRF32",
        test_method="test_fir_f32",
        suffix="f32",
        value_type="float32_t",
        bits=None,
        init_signature="void(ptr,i16,ptr,ptr,i32)",
        owner_header="filtering_functions.h",
        absolute_error="1.0e-6f",
        relative_error="3.0e-5f",
    ),
    ("FIRF64", "test_fir_f64"): FirProtocol(
        test_class="FIRF64",
        test_method="test_fir_f64",
        suffix="f64",
        value_type="float64_t",
        bits=None,
        init_signature="void(ptr,i16,ptr,ptr,i32)",
        owner_header="filtering_functions.h",
        absolute_error="1.0e-15",
        relative_error="5.0e-14",
    ),
    ("FIRQ15", "test_fir_q15"): FirProtocol(
        test_class="FIRQ15",
        test_method="test_fir_q15",
        suffix="q15",
        value_type="q15_t",
        bits=16,
        init_signature="i32(ptr,i16,ptr,ptr,i32)",
        owner_header="filtering_functions.h",
        integer_error=2,
    ),
    ("FIRQ31", "test_fir_q31"): FirProtocol(
        test_class="FIRQ31",
        test_method="test_fir_q31",
        suffix="q31",
        value_type="q31_t",
        bits=32,
        init_signature="void(ptr,i16,ptr,ptr,i32)",
        owner_header="filtering_functions.h",
        integer_error=2,
    ),
    ("FIRQ7", "test_fir_q7"): FirProtocol(
        test_class="FIRQ7",
        test_method="test_fir_q7",
        suffix="q7",
        value_type="q7_t",
        bits=8,
        init_signature="void(ptr,i16,ptr,ptr,i32)",
        owner_header="filtering_functions.h",
        integer_error=2,
    ),
}
_SVM_VALUE_TYPES = {
    "SVMF16": SvmValueType(
        test_class="SVMF16",
        suffix="f16",
        value_type="float16_t",
        scalar_signature="half",
        owner_header="svm_functions_f16.h",
    ),
    "SVMF32": SvmValueType(
        test_class="SVMF32",
        suffix="f32",
        value_type="float32_t",
        scalar_signature="float",
        owner_header="svm_functions.h",
    ),
}
_SVM_KERNELS = (
    SvmKernel(name="linear", pattern_ordinal=1, scalar_parameter_count=0),
    SvmKernel(
        name="polynomial",
        pattern_ordinal=2,
        scalar_parameter_count=2,
        has_degree=True,
    ),
    SvmKernel(name="rbf", pattern_ordinal=3, scalar_parameter_count=1),
    SvmKernel(name="sigmoid", pattern_ordinal=4, scalar_parameter_count=2),
)


def fir_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> FirProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        return None
    if producer.selector_kind != "official":
        return None
    protocol = _FIR_PROTOCOLS.get((producer.test_class, producer.test_method))
    if protocol is None:
        return None
    if tuple((call.symbol, call.signature) for call in workload.protocol) != (
        protocol.calls
    ):
        return None
    return protocol


def svm_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> SvmProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        return None
    if producer.selector_kind != "official":
        return None
    value = _SVM_VALUE_TYPES.get(producer.test_class)
    if value is None:
        return None
    for kernel in _SVM_KERNELS:
        protocol = SvmProtocol(value=value, kernel=kernel)
        if producer.test_method != protocol.test_method:
            continue
        if tuple((call.symbol, call.signature) for call in workload.protocol) != (
            protocol.calls
        ):
            return None
        return protocol
    return None


def _decode_f16_pattern(raw: bytes, name: str) -> tuple[str, ...]:
    if len(raw) % 2 != 0:
        raise WorkloadProviderError(f"CMSIS-DSP {name} is not f16-aligned")
    values = tuple(
        struct.unpack("<e", raw[offset : offset + 2])[0]
        for offset in range(0, len(raw), 2)
    )
    if any(not math.isfinite(value) for value in values):
        raise WorkloadProviderError(f"CMSIS-DSP {name} requires finite float input")
    return tuple(f"static_cast<float16_t>({value.hex()}f)" for value in values)


def _decode_values(
    suffix: str, bits: int | None, raw: bytes, name: str
) -> tuple[str, ...]:
    if suffix == "f16":
        return _decode_f16_pattern(raw, name)
    if suffix == "f32":
        return corpus_dsp_protocol.decode_f32_pattern(raw, name)
    if suffix == "f64":
        return corpus_dsp_protocol.decode_f64_pattern(raw, name)
    if bits is None:
        raise WorkloadProviderError("CMSIS-DSP numeric value type is incomplete")
    return corpus_dsp_protocol.decode_integer_pattern(raw, bits, True, name)


def _oracle_body(protocol: FirProtocol) -> str:
    if protocol.integer_error is not None:
        return f"""    const std::int64_t actual = output[index];
    const std::int64_t expected = kExpected[index];
    const std::int64_t difference =
        actual > expected ? actual - expected : expected - actual;
    if (difference > {protocol.integer_error})
      return false;"""
    if protocol.absolute_error is None or protocol.relative_error is None:
        raise WorkloadProviderError("CMSIS-DSP FIR float tolerance is incomplete")
    comparison_type = "float64_t" if protocol.suffix == "f64" else "float32_t"
    return f"""    const {comparison_type} actual =
        static_cast<{comparison_type}>(output[index]);
    const {comparison_type} expected =
        static_cast<{comparison_type}>(kExpected[index]);
    const {comparison_type} difference =
        actual > expected ? actual - expected : expected - actual;
    const {comparison_type} magnitude = expected < 0 ? -expected : expected;
    if (difference > {protocol.absolute_error} +
                         {protocol.relative_error} * magnitude)
      return false;"""


def render_fir_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    protocol_symbol: str,
) -> str:
    protocol = fir_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError("CMSIS-DSP FIR protocol is inconsistent")
    segments = corpus_dsp_protocol.pattern_segments(patterns)
    suffix = protocol.suffix
    inputs = _decode_values(
        protocol.suffix,
        protocol.bits,
        corpus_dsp_protocol.require_pattern_segment(
            segments, f"FirInput1_{suffix}.txt"
        ),
        "FIR input",
    )
    coefficients = _decode_values(
        protocol.suffix,
        protocol.bits,
        corpus_dsp_protocol.require_pattern_segment(
            segments, f"FirCoefs1_{suffix}.txt"
        ),
        "FIR coefficients",
    )
    expected = _decode_values(
        protocol.suffix,
        protocol.bits,
        corpus_dsp_protocol.require_pattern_segment(segments, f"FirRefs1_{suffix}.txt"),
        "FIR reference",
    )
    configs = corpus_dsp_protocol.decode_i16_pattern(
        corpus_dsp_protocol.require_pattern_segment(segments, "FirConfigs1_s16.txt"),
        "FIR configuration",
    )
    if len(configs) % 2 != 0 or not configs:
        raise WorkloadProviderError("CMSIS-DSP FIR configuration is not paired")
    pairs = tuple(zip(configs[0::2], configs[1::2], strict=True))
    if any(block <= 0 or taps <= 0 for block, taps in pairs):
        raise WorkloadProviderError("CMSIS-DSP FIR configuration is nonpositive")
    if len(inputs) < 2 * max(block for block, _ in pairs):
        raise WorkloadProviderError("CMSIS-DSP FIR input does not cover its blocks")
    coefficient_count = sum(taps for _, taps in pairs)
    if len(coefficients) < coefficient_count:
        raise WorkloadProviderError("CMSIS-DSP FIR coefficient projection is not total")
    coefficients = coefficients[:coefficient_count]
    if len(expected) != sum(2 * block for block, _ in pairs):
        raise WorkloadProviderError("CMSIS-DSP FIR reference projection is not total")
    state_count = max(block + taps for block, taps in pairs)
    init_call = f"arm_fir_init_{suffix}"
    if protocol.init_signature.startswith("i32("):
        init_call = f"(void){init_call}"
    headers = '#include "arm_math.h"'
    if protocol.suffix == "f16":
        headers = (
            '#include "arm_math_types_f16.h"\n#include "dsp/filtering_functions_f16.h"'
        )

    return f"""#include <cstddef>
#include <cstdint>

{headers}

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kConfigCount = {len(pairs)};
constexpr std::size_t kOutputCount = {len(expected)};
constexpr std::size_t kStateCount = {state_count};
constexpr {protocol.value_type} kInput[] = {{
{corpus_dsp_protocol.format_cpp_array(inputs)}
}};
constexpr {protocol.value_type} kCoefficients[] = {{
{corpus_dsp_protocol.format_cpp_array(coefficients)}
}};
constexpr std::int16_t kConfigs[] = {{
{corpus_dsp_protocol.format_cpp_array(tuple(str(value) for value in configs))}
}};
constexpr {protocol.value_type} kExpected[] = {{
{corpus_dsp_protocol.format_cpp_array(expected)}
}};

bool oracle_matches(const {protocol.value_type} *output) {{
  for (std::size_t index = 0; index < kOutputCount; ++index) {{
{_oracle_body(protocol)}
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input,
    const {protocol.value_type} *coefficients,
    const std::int16_t *configs, {protocol.value_type} *state,
    {protocol.value_type} *output) {{
  std::size_t coefficient_offset = 0;
  std::size_t output_offset = 0;
  for (std::size_t config = 0; config < kConfigCount; ++config) {{
    const std::uint32_t block_size =
        static_cast<std::uint32_t>(configs[2 * config]);
    const std::uint16_t num_taps =
        static_cast<std::uint16_t>(configs[2 * config + 1]);
    arm_fir_instance_{suffix} instance{{}};
    {init_call}(&instance, num_taps, coefficients + coefficient_offset,
                state, block_size);
    arm_fir_{suffix}(&instance, input, output + output_offset, block_size);
    arm_fir_{suffix}(&instance, input + block_size,
                     output + output_offset + block_size, block_size);
    coefficient_offset += num_taps;
    output_offset += 2 * block_size;
  }}
}}

int main() {{
  {protocol.value_type} state[kStateCount]{{}};
  {protocol.value_type} output[kOutputCount]{{}};
  {protocol_symbol}(kInput, kCoefficients, kConfigs, state, output);
  return oracle_matches(output) ? 0 : 1;
}}
"""


def render_svm_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    protocol_symbol: str,
) -> str:
    protocol = svm_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError("CMSIS-DSP SVM protocol is inconsistent")
    value = protocol.value
    kernel = protocol.kernel
    ordinal = kernel.pattern_ordinal
    segments = corpus_dsp_protocol.pattern_segments(patterns)
    dims = corpus_dsp_protocol.decode_i16_pattern(
        corpus_dsp_protocol.require_pattern_segment(segments, f"Dims{ordinal}_s16.txt"),
        "SVM dimensions",
    )
    expected_dim_count = 7 if kernel.has_degree else 6
    if len(dims) != expected_dim_count or dims[0] != ordinal:
        raise WorkloadProviderError("CMSIS-DSP SVM dimensions are noncanonical")
    classes = dims[1:3]
    sample_count, vector_dimension, support_vector_count = dims[3:6]
    if sample_count <= 0 or vector_dimension <= 0 or support_vector_count <= 0:
        raise WorkloadProviderError("CMSIS-DSP SVM dimensions are nonpositive")
    if classes[0] == classes[1]:
        raise WorkloadProviderError("CMSIS-DSP SVM classes are not distinct")

    parameters = _decode_values(
        value.suffix,
        None,
        corpus_dsp_protocol.require_pattern_segment(
            segments, f"Params{ordinal}_{value.suffix}.txt"
        ),
        "SVM parameters",
    )
    support_value_count = vector_dimension * support_vector_count
    expected_parameter_count = (
        support_value_count + support_vector_count + 1 + kernel.scalar_parameter_count
    )
    if len(parameters) != expected_parameter_count:
        raise WorkloadProviderError("CMSIS-DSP SVM parameter projection is not total")
    support_vectors = parameters[:support_value_count]
    dual_coefficients = parameters[
        support_value_count : support_value_count + support_vector_count
    ]
    scalar_parameters = parameters[support_value_count + support_vector_count :]
    intercept = scalar_parameters[0]
    kernel_scalars = scalar_parameters[1:]

    samples = _decode_values(
        value.suffix,
        None,
        corpus_dsp_protocol.require_pattern_segment(
            segments, f"Samples{ordinal}_{value.suffix}.txt"
        ),
        "SVM samples",
    )
    if len(samples) != sample_count * vector_dimension:
        raise WorkloadProviderError("CMSIS-DSP SVM sample projection is not total")
    expected = corpus_dsp_protocol.decode_integer_pattern(
        corpus_dsp_protocol.require_pattern_segment(
            segments, f"Reference{ordinal}_s32.txt"
        ),
        32,
        True,
        "SVM reference",
    )
    if len(expected) != sample_count:
        raise WorkloadProviderError("CMSIS-DSP SVM reference projection is not total")

    headers = '#include "arm_math.h"'
    if value.suffix == "f16":
        headers = '#include "arm_math_types_f16.h"\n#include "dsp/svm_functions_f16.h"'
    constants = ""
    init_tail = ""
    main_tail = ""
    if kernel.has_degree:
        degree = dims[6]
        if degree <= 0:
            raise WorkloadProviderError("CMSIS-DSP SVM degree is nonpositive")
        constants += f"constexpr std::int32_t kDegree = {degree};\n"
        init_tail += ", kDegree"
    scalar_names = (
        ("kCoef0", "kGamma")[-kernel.scalar_parameter_count :]
        if kernel.scalar_parameter_count
        else ()
    )
    for name, literal in zip(scalar_names, kernel_scalars, strict=True):
        constants += f"constexpr {value.value_type} {name} = {literal};\n"
        init_tail += f", {name.lower().removeprefix('k')}"
        main_tail += f", {name}"
    wrapper_scalar_parameters = "".join(
        f", {value.value_type} {name.lower().removeprefix('k')}"
        for name in scalar_names
    )

    return f"""#include <cstddef>
#include <cstdint>

{headers}

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kSampleCount = {sample_count};
constexpr std::size_t kVectorDimension = {vector_dimension};
constexpr std::size_t kSupportVectorCount = {support_vector_count};
constexpr {value.value_type} kSamples[] = {{
{corpus_dsp_protocol.format_cpp_array(samples)}
}};
constexpr {value.value_type} kSupportVectors[] = {{
{corpus_dsp_protocol.format_cpp_array(support_vectors)}
}};
constexpr {value.value_type} kDualCoefficients[] = {{
{corpus_dsp_protocol.format_cpp_array(dual_coefficients)}
}};
constexpr {value.value_type} kIntercept = {intercept};
constexpr std::int32_t kClasses[] = {{{classes[0]}, {classes[1]}}};
{constants}constexpr std::int32_t kExpected[] = {{
{corpus_dsp_protocol.format_cpp_array(expected)}
}};

bool oracle_matches(const std::int32_t *output) {{
  for (std::size_t sample = 0; sample < kSampleCount; ++sample) {{
    if (output[sample] != kExpected[sample])
      return false;
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {value.value_type} *samples,
    const {value.value_type} *support_vectors,
    const {value.value_type} *dual_coefficients,
    {value.value_type} intercept, const std::int32_t *classes,
    std::int32_t *output{wrapper_scalar_parameters}) {{
  arm_svm_{kernel.name}_instance_{value.suffix} instance;
  arm_svm_{kernel.name}_init_{value.suffix}(
      &instance, kSupportVectorCount, kVectorDimension, intercept,
      dual_coefficients, support_vectors, classes{init_tail});
  for (std::size_t sample = 0; sample < kSampleCount; ++sample) {{
    arm_svm_{kernel.name}_predict_{value.suffix}(
        &instance, samples + sample * kVectorDimension, output + sample);
  }}
}}

int main() {{
  std::int32_t output[kSampleCount];
  {protocol_symbol}(kSamples, kSupportVectors, kDualCoefficients, kIntercept,
                    kClasses, output{main_tail});
  return oracle_matches(output) ? 0 : 1;
}}
"""
