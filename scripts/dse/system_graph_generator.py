"""System graph generator: converts DSE DesignPoint to SystemTopologySpec.

Takes a DesignPoint (type subset + counts + NoC + L2) and produces a
SystemTopologySpec with mesh dimensions, core placement, and L2 bank layout.
Optionally generates system-level MLIR ADG text for the fabric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .core_type_library import (
    ALL_TYPES,
    CoreDesignParams,
    NUM_TYPES,
    get_core_design_params,
    index_to_type_id,
)
from .design_space import DesignPoint
from .dse_config import MAX_CORE_TYPES


# ---------------------------------------------------------------------------
# System topology spec (pure Python version for DSE)
# ---------------------------------------------------------------------------

@dataclass
class CorePlacement:
    """Placement of one core instance on the mesh."""

    type_index: int = 0
    type_id: str = ""
    instance_id: int = 0
    row: int = 0
    col: int = 0


@dataclass
class L2BankPlacement:
    """Placement of one L2 bank on the mesh."""

    bank_id: int = 0
    row: int = 0
    col: int = 0
    size_kb: int = 32


@dataclass
class SystemTopologySpec:
    """System-level topology specification produced by the outer DSE."""

    # NoC parameters
    noc_topology: str = "mesh"
    noc_bandwidth: int = 1
    mesh_rows: int = 2
    mesh_cols: int = 2

    # Shared L2 memory
    l2_total_size_kb: int = 256
    l2_bank_count: int = 8

    # Selected core type info: list of (type_index, type_id, instance_count)
    core_library: List[Dict[str, Any]] = field(default_factory=list)

    # Core placement
    core_placements: List[CorePlacement] = field(default_factory=list)

    # L2 bank placement
    l2_bank_placements: List[L2BankPlacement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "noc": {
                "topology": self.noc_topology,
                "bandwidth": self.noc_bandwidth,
                "mesh_rows": self.mesh_rows,
                "mesh_cols": self.mesh_cols,
            },
            "shared_memory": {
                "l2_total_size_kb": self.l2_total_size_kb,
                "l2_bank_count": self.l2_bank_count,
            },
            "core_library": self.core_library,
            "core_placement": [
                {
                    "type": cp.type_index,
                    "type_id": cp.type_id,
                    "instance": cp.instance_id,
                    "row": cp.row,
                    "col": cp.col,
                }
                for cp in self.core_placements
            ],
            "l2_bank_placement": [
                {
                    "bank_id": bp.bank_id,
                    "row": bp.row,
                    "col": bp.col,
                    "size_kb": bp.size_kb,
                }
                for bp in self.l2_bank_placements
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# System graph generator
# ---------------------------------------------------------------------------

class SystemGraphGenerator:
    """Converts a DesignPoint into a SystemTopologySpec."""

    def generate(
        self,
        design_point: DesignPoint,
        core_library: Optional[Dict[str, CoreDesignParams]] = None,
    ) -> SystemTopologySpec:
        """Generate a full system topology from a design point.

        Args:
            design_point: The candidate design from the BO loop.
            core_library: Optional override for the type library.
                Uses ALL_TYPES by default.

        Returns:
            A complete SystemTopologySpec with placements.
        """
        if core_library is None:
            core_library = ALL_TYPES

        spec = SystemTopologySpec()
        spec.noc_topology = design_point.noc_topology
        spec.noc_bandwidth = design_point.noc_bandwidth
        spec.l2_total_size_kb = design_point.l2_size_kb
        spec.l2_bank_count = design_point.l2_bank_count

        # Collect selected types and their instance counts
        selected_types: List[Tuple[int, str, int]] = []
        total_core_count = 0
        all_type_ids = list(core_library.keys())

        for i in range(min(MAX_CORE_TYPES, len(all_type_ids))):
            if design_point.type_mask[i] and design_point.instance_counts[i] > 0:
                type_id = all_type_ids[i]
                count = design_point.instance_counts[i]
                selected_types.append((i, type_id, count))
                total_core_count += count

        if total_core_count == 0:
            return spec

        # Compute mesh dimensions (square-ish)
        mesh_rows, mesh_cols = self._compute_mesh_dims(total_core_count)
        spec.mesh_rows = mesh_rows
        spec.mesh_cols = mesh_cols

        # Build core library metadata
        spec.core_library = []
        for type_idx, type_id, inst_count in selected_types:
            params = core_library[type_id]
            spec.core_library.append({
                "type_index": type_idx,
                "type_id": type_id,
                "instance_count": inst_count,
                "params": params.to_dict() if hasattr(params, "to_dict") else {},
            })

        # Generate locality-aware core placement
        spec.core_placements = self._generate_core_placement(
            selected_types, mesh_rows, mesh_cols
        )

        # Generate L2 bank placement (along mesh edges)
        spec.l2_bank_placements = self._generate_l2_placement(
            design_point.l2_bank_count,
            design_point.l2_size_kb,
            mesh_rows,
            mesh_cols,
        )

        return spec

    @staticmethod
    def _compute_mesh_dims(total_cores: int) -> Tuple[int, int]:
        """Compute mesh dimensions for a given core count.

        Produces a roughly square layout: rows <= cols, rows*cols >= total.
        """
        if total_cores <= 0:
            return (2, 2)

        side = math.isqrt(total_cores)
        if side * side >= total_cores:
            return (side, side)

        # Try side x (side+1)
        if side * (side + 1) >= total_cores:
            return (side, side + 1)

        return (side + 1, side + 1)

    @staticmethod
    def _generate_core_placement(
        selected_types: List[Tuple[int, str, int]],
        mesh_rows: int,
        mesh_cols: int,
    ) -> List[CorePlacement]:
        """Place cores on the mesh with locality grouping.

        Groups instances of the same type in adjacent mesh positions
        to minimize intra-type communication distance.
        """
        placements: List[CorePlacement] = []
        pos = 0
        total_slots = mesh_rows * mesh_cols

        for type_idx, type_id, inst_count in selected_types:
            for inst in range(inst_count):
                if pos >= total_slots:
                    # Wrap around or clamp to last position
                    row = mesh_rows - 1
                    col = mesh_cols - 1
                else:
                    row = pos // mesh_cols
                    col = pos % mesh_cols
                placements.append(CorePlacement(
                    type_index=type_idx,
                    type_id=type_id,
                    instance_id=inst,
                    row=row,
                    col=col,
                ))
                pos += 1

        return placements

    @staticmethod
    def _generate_l2_placement(
        bank_count: int,
        total_size_kb: int,
        mesh_rows: int,
        mesh_cols: int,
    ) -> List[L2BankPlacement]:
        """Place L2 banks evenly along the mesh edges."""
        if bank_count <= 0:
            return []

        per_bank_kb = max(1, total_size_kb // bank_count)
        placements: List[L2BankPlacement] = []

        # Distribute banks along the edges of the mesh
        edge_positions: List[Tuple[int, int]] = []

        # Top edge
        for c in range(mesh_cols):
            edge_positions.append((0, c))
        # Bottom edge (if rows > 1)
        if mesh_rows > 1:
            for c in range(mesh_cols):
                edge_positions.append((mesh_rows - 1, c))
        # Left edge (excluding corners)
        for r in range(1, mesh_rows - 1):
            edge_positions.append((r, 0))
        # Right edge (excluding corners)
        if mesh_cols > 1:
            for r in range(1, mesh_rows - 1):
                edge_positions.append((r, mesh_cols - 1))

        if not edge_positions:
            edge_positions = [(0, 0)]

        for bank_id in range(bank_count):
            pos_idx = bank_id % len(edge_positions)
            row, col = edge_positions[pos_idx]
            placements.append(L2BankPlacement(
                bank_id=bank_id,
                row=row,
                col=col,
                size_kb=per_bank_kb,
            ))

        return placements


# ---------------------------------------------------------------------------
# MLIR generation helper
# ---------------------------------------------------------------------------

def to_system_mlir(topology_spec: SystemTopologySpec) -> str:
    """Generate a system-level Fabric MLIR module from the topology spec.

    This is a structural placeholder that produces valid MLIR syntax
    with fabric.module and fabric.core ops. The actual ADG content per
    core type is filled in by the inner DSE (C12).
    """
    lines = []
    lines.append('fabric.module @system {')

    # NoC configuration attribute
    lines.append(f'  // NoC: {topology_spec.noc_topology}, '
                 f'BW={topology_spec.noc_bandwidth}, '
                 f'{topology_spec.mesh_rows}x{topology_spec.mesh_cols} mesh')

    # Core instances
    for cp in topology_spec.core_placements:
        inst_name = f"core_t{cp.type_index}_i{cp.instance_id}"
        lines.append(
            f'  fabric.core @{inst_name} '
            f'{{type_index = {cp.type_index}, '
            f'type_id = "{cp.type_id}", '
            f'row = {cp.row}, col = {cp.col}}} {{'
        )
        lines.append('  }')

    # L2 banks
    for bp in topology_spec.l2_bank_placements:
        lines.append(
            f'  fabric.l2bank @l2_bank{bp.bank_id} '
            f'{{size_kb = {bp.size_kb}, '
            f'row = {bp.row}, col = {bp.col}}}'
        )

    # Router instances
    for r in range(topology_spec.mesh_rows):
        for c in range(topology_spec.mesh_cols):
            lines.append(
                f'  fabric.router @router_r{r}c{c} '
                f'{{row = {r}, col = {c}, '
                f'bandwidth = {topology_spec.noc_bandwidth}}}'
            )

    lines.append('}')
    return '\n'.join(lines)
