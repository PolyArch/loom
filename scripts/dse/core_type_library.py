"""Combinatorial KHG core type library for LOOM DSE.

Defines the 30-type core library: 6 domain-specific types (D1-D6) and
24 combinatorial KHG types (3 compute-mix x 2 PE-type x 2 SPM x
2 array-size). Provides naming convention encoding/decoding, parameter
mapping to CoreDesignParams, and enumeration for the outer DSE.

Naming convention for combinatorial types: C{I|F|M}{S|T}{Y|N}{8|12}
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


# ---------------------------------------------------------------------------
# CoreDesignParams: unified parameter type for the 30-type library
# ---------------------------------------------------------------------------

# FU category sets for op support checking.
_FP_OPS = frozenset({"fadd", "fmul", "fsub", "fdiv", "fcmp", "fma", "fexp",
                      "fsqrt", "arith.addf", "arith.mulf", "arith.subf",
                      "arith.divf", "arith.cmpf", "math.exp", "math.sqrt",
                      "math.fma"})
_INT_OPS = frozenset({"add", "sub", "mul", "div", "cmp", "and", "or", "xor",
                       "shl", "shr", "select",
                       "arith.addi", "arith.subi", "arith.muli",
                       "arith.andi", "arith.ori", "arith.xori",
                       "arith.shli", "arith.shrsi", "arith.shrui",
                       "arith.cmpi", "arith.select"})
_MEM_OPS = frozenset({"load", "store", "handshake.load", "handshake.store"})


@dataclass(frozen=True)
class CoreDesignParams:
    """Unified design parameters for any core type (domain-specific or KHG)."""

    type_id: str
    name: str
    array_rows: int
    array_cols: int
    fu_alu_count: int
    fu_mul_count: int
    fu_fp_count: int
    fu_mem_count: int = 2
    spm_size_kb: int = 0
    is_temporal: bool = False
    instruction_slots: int = 0
    num_registers: int = 0
    data_width: int = 32
    has_fp: bool = False
    has_int: bool = True

    @property
    def fu_mix(self) -> dict:
        """Return FU type counts as a dict for affinity scoring."""
        return {
            "alu": self.fu_alu_count,
            "mul": self.fu_mul_count,
            "fp": self.fu_fp_count,
            "mem": self.fu_mem_count,
        }

    @property
    def total_pes(self) -> int:
        return self.array_rows * self.array_cols

    def supports_ops(self, ops: Dict[str, int]) -> bool:
        """Return True if this core can handle all requested op types."""
        for op_name in ops:
            op_lower = op_name.lower()
            if op_lower in _FP_OPS and not self.has_fp:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "type_id": self.type_id,
            "name": self.name,
            "array_rows": self.array_rows,
            "array_cols": self.array_cols,
            "fu_alu_count": self.fu_alu_count,
            "fu_mul_count": self.fu_mul_count,
            "fu_fp_count": self.fu_fp_count,
            "fu_mem_count": self.fu_mem_count,
            "spm_size_kb": self.spm_size_kb,
            "is_temporal": self.is_temporal,
            "total_pes": self.total_pes,
        }

    def fu_capability_vector(self) -> List[int]:
        """Return a vector [alu, mul, fp, mem] of FU counts."""
        return [self.fu_alu_count, self.fu_mul_count,
                self.fu_fp_count, self.fu_mem_count]


# ---------------------------------------------------------------------------
# Domain-Specific Types (D1-D6)
# ---------------------------------------------------------------------------

_DOMAIN_SPECIFIC_DEFS: List[CoreDesignParams] = [
    CoreDesignParams(
        type_id="D1", name="LLM (FP-heavy, large)",
        array_rows=6, array_cols=6,
        fu_alu_count=4, fu_mul_count=4, fu_fp_count=4, fu_mem_count=2,
        spm_size_kb=64, is_temporal=False,
        has_fp=True, has_int=True,
    ),
    CoreDesignParams(
        type_id="D2", name="CV (mixed, medium)",
        array_rows=4, array_cols=4,
        fu_alu_count=4, fu_mul_count=3, fu_fp_count=3, fu_mem_count=2,
        spm_size_kb=32, is_temporal=False,
        has_fp=True, has_int=True,
    ),
    CoreDesignParams(
        type_id="D3", name="Signal (temporal, multiply)",
        array_rows=4, array_cols=4,
        fu_alu_count=3, fu_mul_count=4, fu_fp_count=2, fu_mem_count=2,
        spm_size_kb=16, is_temporal=True,
        instruction_slots=8, num_registers=8,
        has_fp=True, has_int=True,
    ),
    CoreDesignParams(
        type_id="D4", name="Crypto (INT-heavy, bitwise)",
        array_rows=4, array_cols=4,
        fu_alu_count=6, fu_mul_count=4, fu_fp_count=0, fu_mem_count=2,
        spm_size_kb=8, is_temporal=False, data_width=64,
        has_fp=False, has_int=True,
    ),
    CoreDesignParams(
        type_id="D5", name="Sensor (temporal, control)",
        array_rows=4, array_cols=4,
        fu_alu_count=4, fu_mul_count=2, fu_fp_count=1, fu_mem_count=2,
        spm_size_kb=8, is_temporal=True,
        instruction_slots=16, num_registers=8,
        has_fp=True, has_int=True,
    ),
    CoreDesignParams(
        type_id="D6", name="Control (spatial, balanced)",
        array_rows=4, array_cols=4,
        fu_alu_count=4, fu_mul_count=2, fu_fp_count=0, fu_mem_count=2,
        spm_size_kb=4, is_temporal=False,
        has_fp=False, has_int=True,
    ),
]


def _khg_to_core_design_params(p: KHGTypeParams) -> CoreDesignParams:
    """Convert a KHGTypeParams to a CoreDesignParams."""
    return CoreDesignParams(
        type_id=p.type_id,
        name=p.type_id,
        array_rows=p.array_rows,
        array_cols=p.array_cols,
        fu_alu_count=p.fu_alu_count,
        fu_mul_count=p.fu_mul_count,
        fu_fp_count=p.fu_fp_count,
        fu_mem_count=2,
        spm_size_kb=p.spm_size_kb,
        is_temporal=p.is_temporal,
        instruction_slots=p.instruction_slots,
        num_registers=p.num_registers,
        data_width=p.data_width,
        has_fp=(p.fu_fp_count > 0),
        has_int=True,
    )


# ---------------------------------------------------------------------------
# Unified 30-type library
# ---------------------------------------------------------------------------

DOMAIN_SPECIFIC_TYPES: Dict[str, CoreDesignParams] = {
    d.type_id: d for d in _DOMAIN_SPECIFIC_DEFS
}

COMBINATORIAL_TYPES: Dict[str, CoreDesignParams] = {
    tid: _khg_to_core_design_params(params_from_type_id(tid))
    for tid in ALL_TYPE_IDS
}

ALL_TYPES: Dict[str, CoreDesignParams] = {}
ALL_TYPES.update(DOMAIN_SPECIFIC_TYPES)
ALL_TYPES.update(COMBINATORIAL_TYPES)

NUM_TYPES: int = len(ALL_TYPES)
assert NUM_TYPES == 30, f"Expected 30 types, got {NUM_TYPES}"

# Canonical ordering: D1-D6, then 24 combinatorial types.
_ORDERED_TYPE_IDS: List[str] = (
    [f"D{i}" for i in range(1, 7)] + list(ALL_TYPE_IDS)
)
assert len(_ORDERED_TYPE_IDS) == 30

_TYPE_ID_TO_INDEX: Dict[str, int] = {
    tid: idx for idx, tid in enumerate(_ORDERED_TYPE_IDS)
}


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------

def get_all_type_ids() -> List[str]:
    """Return all 30 type IDs in canonical order."""
    return list(_ORDERED_TYPE_IDS)


def index_to_type_id(idx: int) -> str:
    """Convert a 0-based index to its type ID string."""
    if idx < 0 or idx >= NUM_TYPES:
        raise IndexError(f"Type index {idx} out of range [0, {NUM_TYPES})")
    return _ORDERED_TYPE_IDS[idx]


def type_id_to_index(type_id: str) -> int:
    """Convert a type ID string to its 0-based index."""
    if type_id not in _TYPE_ID_TO_INDEX:
        raise KeyError(f"Unknown type ID '{type_id}'")
    return _TYPE_ID_TO_INDEX[type_id]


def get_core_design_params(type_id: str) -> CoreDesignParams:
    """Get the CoreDesignParams for a type ID."""
    if type_id not in ALL_TYPES:
        raise KeyError(f"Unknown type ID '{type_id}'")
    return ALL_TYPES[type_id]


def get_type_name(type_id: str) -> str:
    """Get a human-readable name for a type ID."""
    if type_id not in ALL_TYPES:
        raise KeyError(f"Unknown type ID '{type_id}'")
    return ALL_TYPES[type_id].name


def type_supports_ops(type_id: str, ops: Dict[str, int]) -> bool:
    """Check if a core type can handle all requested operations."""
    params = get_core_design_params(type_id)
    return params.supports_ops(ops)


def get_fu_capability_vector(type_id: str) -> List[int]:
    """Return the FU capability vector [alu, mul, fp, mem] for a type."""
    params = get_core_design_params(type_id)
    return params.fu_capability_vector()
