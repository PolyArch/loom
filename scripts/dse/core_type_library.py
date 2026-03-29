"""Combinatorial KHG core type library for LOOM DSE.

Defines the 24 combinatorial KHG types (3 compute-mix x 2 PE-type x 2 SPM x
2 array-size) and provides naming convention encoding/decoding, parameter
mapping to CoreDesignParams equivalents, and enumeration for the outer DSE.

Naming convention: C{I|F|M}{S|T}{Y|N}{8|12}
  - C:     prefix (Core)
  - I/F/M: INT-heavy / FP-heavy / Mixed compute mix
  - S/T:   Spatial / Temporal PE type
  - Y/N:   with / without scratchpad memory
  - 8/12:  8x8 / 12x12 array size
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dimension enumerations
# ---------------------------------------------------------------------------

class ComputeMix(Enum):
    """FU compute-mix dimension."""
    INT_HEAVY = "I"
    FP_HEAVY = "F"
    MIXED = "M"


class PEKind(Enum):
    """PE type dimension."""
    SPATIAL = "S"
    TEMPORAL = "T"


class SPMPresence(Enum):
    """Scratchpad memory presence dimension."""
    WITH_SPM = "Y"
    WITHOUT_SPM = "N"


class ArraySize(Enum):
    """PE array size dimension."""
    SIZE_8 = "8"
    SIZE_12 = "12"


# ---------------------------------------------------------------------------
# FU allocation per compute mix
# ---------------------------------------------------------------------------

COMPUTE_MIX_FU_COUNTS: Dict[ComputeMix, Dict[str, int]] = {
    ComputeMix.INT_HEAVY: {"alu": 6, "mul": 4, "fp": 1},
    ComputeMix.FP_HEAVY:  {"alu": 2, "mul": 2, "fp": 6},
    ComputeMix.MIXED:     {"alu": 4, "mul": 3, "fp": 3},
}

# ---------------------------------------------------------------------------
# Array dimensions per size
# ---------------------------------------------------------------------------

ARRAY_DIMENSIONS: Dict[ArraySize, Tuple[int, int]] = {
    ArraySize.SIZE_8:  (8, 8),
    ArraySize.SIZE_12: (12, 12),
}

# ---------------------------------------------------------------------------
# SPM parameters
# ---------------------------------------------------------------------------

SPM_PARAMS_WITH: Dict[str, int] = {
    "spm_size_kb": 16,
    "spm_ld_ports": 2,
    "spm_st_ports": 2,
}

SPM_PARAMS_WITHOUT: Dict[str, int] = {
    "spm_size_kb": 0,
    "spm_ld_ports": 0,
    "spm_st_ports": 0,
}

# ---------------------------------------------------------------------------
# Temporal PE defaults
# ---------------------------------------------------------------------------

TEMPORAL_PE_DEFAULTS: Dict[str, int] = {
    "instruction_slots": 8,
    "num_registers": 8,
}


# ---------------------------------------------------------------------------
# KHG Type Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KHGTypeParams:
    """Concrete parameters for one KHG combinatorial core type."""

    type_id: str
    compute_mix: ComputeMix
    pe_kind: PEKind
    spm_presence: SPMPresence
    array_size: ArraySize

    # Derived parameters
    array_rows: int
    array_cols: int
    fu_alu_count: int
    fu_mul_count: int
    fu_fp_count: int
    spm_size_kb: int
    spm_ld_ports: int
    spm_st_ports: int
    instruction_slots: int
    num_registers: int
    data_width: int = 32

    @property
    def total_pes(self) -> int:
        return self.array_rows * self.array_cols

    @property
    def is_temporal(self) -> bool:
        return self.pe_kind == PEKind.TEMPORAL

    @property
    def has_spm(self) -> bool:
        return self.spm_presence == SPMPresence.WITH_SPM


# ---------------------------------------------------------------------------
# Naming convention: encode / decode
# ---------------------------------------------------------------------------

_TYPE_ID_REGEX = re.compile(r"^C([IFM])([ST])([YN])(8|12)$")

_COMPUTE_MIX_FROM_CHAR: Dict[str, ComputeMix] = {
    "I": ComputeMix.INT_HEAVY,
    "F": ComputeMix.FP_HEAVY,
    "M": ComputeMix.MIXED,
}

_PE_KIND_FROM_CHAR: Dict[str, PEKind] = {
    "S": PEKind.SPATIAL,
    "T": PEKind.TEMPORAL,
}

_SPM_FROM_CHAR: Dict[str, SPMPresence] = {
    "Y": SPMPresence.WITH_SPM,
    "N": SPMPresence.WITHOUT_SPM,
}

_SIZE_FROM_STR: Dict[str, ArraySize] = {
    "8": ArraySize.SIZE_8,
    "12": ArraySize.SIZE_12,
}


def encode_type_id(
    compute: ComputeMix,
    pe: PEKind,
    spm: SPMPresence,
    size: ArraySize,
) -> str:
    """Encode a 4D parameter tuple into a KHG type ID string.

    Returns a string like 'CISY8', 'CFTY12', etc.
    """
    return f"C{compute.value}{pe.value}{spm.value}{size.value}"


def decode_type_id(
    type_id: str,
) -> Tuple[ComputeMix, PEKind, SPMPresence, ArraySize]:
    """Decode a KHG type ID string back into the 4D parameter tuple.

    Raises ValueError if the type_id does not match the expected format.
    """
    m = _TYPE_ID_REGEX.match(type_id)
    if m is None:
        raise ValueError(
            f"Invalid KHG type ID '{type_id}': "
            f"expected format C[IFM][ST][YN](8|12)"
        )
    return (
        _COMPUTE_MIX_FROM_CHAR[m.group(1)],
        _PE_KIND_FROM_CHAR[m.group(2)],
        _SPM_FROM_CHAR[m.group(3)],
        _SIZE_FROM_STR[m.group(4)],
    )


def is_valid_type_id(type_id: str) -> bool:
    """Return True if the string is a valid KHG type ID."""
    return _TYPE_ID_REGEX.match(type_id) is not None


# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------

def make_khg_params(
    compute: ComputeMix,
    pe: PEKind,
    spm: SPMPresence,
    size: ArraySize,
) -> KHGTypeParams:
    """Build the full KHGTypeParams for a given 4D parameter combination."""
    type_id = encode_type_id(compute, pe, spm, size)
    fu = COMPUTE_MIX_FU_COUNTS[compute]
    rows, cols = ARRAY_DIMENSIONS[size]
    spm_p = SPM_PARAMS_WITH if spm == SPMPresence.WITH_SPM else SPM_PARAMS_WITHOUT

    if pe == PEKind.TEMPORAL:
        islots = TEMPORAL_PE_DEFAULTS["instruction_slots"]
        nregs = TEMPORAL_PE_DEFAULTS["num_registers"]
    else:
        islots = 0
        nregs = 0

    return KHGTypeParams(
        type_id=type_id,
        compute_mix=compute,
        pe_kind=pe,
        spm_presence=spm,
        array_size=size,
        array_rows=rows,
        array_cols=cols,
        fu_alu_count=fu["alu"],
        fu_mul_count=fu["mul"],
        fu_fp_count=fu["fp"],
        spm_size_kb=spm_p["spm_size_kb"],
        spm_ld_ports=spm_p["spm_ld_ports"],
        spm_st_ports=spm_p["spm_st_ports"],
        instruction_slots=islots,
        num_registers=nregs,
    )


def params_from_type_id(type_id: str) -> KHGTypeParams:
    """Build KHGTypeParams from a type ID string like 'CISY8'."""
    compute, pe, spm, size = decode_type_id(type_id)
    return make_khg_params(compute, pe, spm, size)


# ---------------------------------------------------------------------------
# Combinatorial Generator
# ---------------------------------------------------------------------------

# Canonical enumeration order: compute x pe x spm x size
ALL_TYPE_IDS: List[str] = [
    encode_type_id(c, p, s, z)
    for c in ComputeMix
    for p in PEKind
    for s in SPMPresence
    for z in ArraySize
]

assert len(ALL_TYPE_IDS) == 24, f"Expected 24 types, got {len(ALL_TYPE_IDS)}"
assert len(set(ALL_TYPE_IDS)) == 24, "Duplicate type IDs detected"


class CombinatorialGenerator:
    """Enumerates all 24 KHG combinatorial types and produces their params.

    Usage:
        gen = CombinatorialGenerator()
        for params in gen:
            print(params.type_id, params.array_rows, params.fu_alu_count)

        # Or get a specific type:
        p = gen.get("CMSY8")
    """

    def __init__(self) -> None:
        self._cache: Dict[str, KHGTypeParams] = {}
        for tid in ALL_TYPE_IDS:
            self._cache[tid] = params_from_type_id(tid)

    @property
    def type_ids(self) -> List[str]:
        """All 24 type ID strings in canonical order."""
        return list(ALL_TYPE_IDS)

    @property
    def count(self) -> int:
        return 24

    def get(self, type_id: str) -> KHGTypeParams:
        """Get parameters for a specific type ID.

        Raises KeyError if the type_id is not one of the 24 types.
        """
        if type_id not in self._cache:
            raise KeyError(
                f"Unknown KHG type ID '{type_id}'. "
                f"Valid IDs: {', '.join(ALL_TYPE_IDS)}"
            )
        return self._cache[type_id]

    def __iter__(self):
        for tid in ALL_TYPE_IDS:
            yield self._cache[tid]

    def __len__(self) -> int:
        return 24

    def filter_by(
        self,
        compute: Optional[ComputeMix] = None,
        pe: Optional[PEKind] = None,
        spm: Optional[SPMPresence] = None,
        size: Optional[ArraySize] = None,
    ) -> List[KHGTypeParams]:
        """Return a subset of types matching the given dimension filters."""
        result = []
        for params in self:
            if compute is not None and params.compute_mix != compute:
                continue
            if pe is not None and params.pe_kind != pe:
                continue
            if spm is not None and params.spm_presence != spm:
                continue
            if size is not None and params.array_size != size:
                continue
            result.append(params)
        return result

    def to_design_space_configs(self) -> List[dict]:
        """Export all 24 types as dicts compatible with CoreTypeConfig fields."""
        configs = []
        for params in self:
            configs.append({
                "type_id": params.type_id,
                "pe_grid_rows": params.array_rows,
                "pe_grid_cols": params.array_cols,
                "fu_alu_count": params.fu_alu_count,
                "fu_mul_count": params.fu_mul_count,
                "fu_fp_count": params.fu_fp_count,
                "spm_size_kb": params.spm_size_kb,
                "pe_kind": params.pe_kind.value,
                "is_temporal": params.is_temporal,
                "instruction_slots": params.instruction_slots,
                "num_registers": params.num_registers,
            })
        return configs
