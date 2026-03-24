#!/usr/bin/env python3
"""E22: RTL Synthesis Results.

Collects and reformats synthesis data from actual Synopsys DC runs stored in
out/experiments/e8_rtl_synthesis/. Produces a unified per-component area,
frequency, and power breakdown CSV.

If synthesis data is not available, reports that honestly.

Data source: e8_rtl_synthesis/synthesis_summary.json contains real DC results
for noc_router and tapestry_spm at SAED14nm FinFET, 500MHz target.

Usage:
    python3 scripts/experiments/run_e22_synthesis.py
"""

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SYNTHESIS_JSON = REPO_ROOT / "out" / "experiments" / "e8_rtl_synthesis" / "synthesis_summary.json"

# Design configurations from co-optimization (3 Pareto-optimal points)
# Each design specifies core composition and target frequency
DESIGNS = {
    "design_A_hetero_2core": {
        "label": "Heterogeneous 2-core (1x 10x10 + 1x 8x8)",
        "cores": [
            {"type": "10x10", "pes": 100, "count": 1},
            {"type": "8x8", "pes": 64, "count": 1},
        ],
        "total_pes": 164,
        "noc_routers": 4,
        "spm_banks": 4,  # 2 per core
        "l2_banks": 2,
    },
    "design_B_hetero_4core": {
        "label": "Heterogeneous 4-core (2x 8x8 + 2x 6x6)",
        "cores": [
            {"type": "8x8", "pes": 64, "count": 2},
            {"type": "6x6", "pes": 36, "count": 2},
        ],
        "total_pes": 200,
        "noc_routers": 8,
        "spm_banks": 8,
        "l2_banks": 4,
    },
    "design_C_homo_4core": {
        "label": "Homogeneous 4-core (4x 8x8)",
        "cores": [
            {"type": "8x8", "pes": 64, "count": 4},
        ],
        "total_pes": 256,
        "noc_routers": 8,
        "spm_banks": 8,
        "l2_banks": 4,
    },
}

# PE area estimate: each PE is roughly 2 NAND2 equivalent gates
# From synthesis, we know noc_router = 4427.79 um2 cell area, ~9401 cells
# SPM (4KB) = 67090.4 um2 cell area
# PE (estimated from published CGRA data at 14nm) ~ 1200 um2 per PE
PE_AREA_UM2 = 1200.0
PE_POWER_MW = 0.15  # per PE at 500MHz (estimated)


def git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_synthesis_data():
    """Load real synthesis data from DC runs."""
    if not SYNTHESIS_JSON.exists():
        return None
    with open(SYNTHESIS_JSON) as f:
        return json.load(f)


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E22"
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_data = load_synthesis_data()

    print("E22: RTL Synthesis Results")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")

    if synth_data is None:
        print("  WARNING: No synthesis data found at", SYNTHESIS_JSON)
        print("  Cannot produce E22 results without DC synthesis runs.")
        with open(out_dir / "synthesis.csv", "w") as f:
            f.write("# No synthesis data available\n")
        return 1

    # Extract real component data
    noc_data = synth_data["designs"].get("noc_router", {})
    spm_data = synth_data["designs"].get("tapestry_spm", {})
    pdk = synth_data.get("pdk", "SAED14nm")
    tool = synth_data.get("synthesis_tool", "Synopsys DC")
    clock_ns = synth_data.get("clock_period_ns", 2.0)
    target_mhz = 1000.0 / clock_ns

    noc_area = noc_data.get("area", {}).get("total_cell_area", 0)
    noc_power = noc_data.get("power", {}).get("total_power_mw", 0)
    noc_leakage = noc_data.get("power", {}).get("cell_leakage_power_uw", 0) / 1000.0
    noc_cells = noc_data.get("cell_count", {}).get("total_cells", 0)
    noc_slack = noc_data.get("timing", {}).get("worst_slack_ns", 0)
    noc_met = noc_data.get("timing", {}).get("met_timing", False)

    spm_area = spm_data.get("area", {}).get("total_cell_area", 0)
    spm_power = spm_data.get("power", {}).get("total_power_mw", 0)
    spm_leakage = spm_data.get("power", {}).get("cell_leakage_power_uw", 0) / 1000.0
    spm_cells = spm_data.get("cell_count", {}).get("total_cells", 0)

    print(f"\n  Synthesis tool: {tool}")
    print(f"  PDK: {pdk}")
    print(f"  Target frequency: {target_mhz:.0f} MHz ({clock_ns} ns)")
    print(f"\n  Real synthesis components:")
    print(f"    NoC Router: area={noc_area:.0f} um2, power={noc_power:.2f} mW, "
          f"cells={noc_cells}, timing_met={noc_met}")
    print(f"    SPM (4KB):  area={spm_area:.0f} um2, power={spm_power:.2f} mW, "
          f"cells={spm_cells}")
    print(f"    PE (est):   area={PE_AREA_UM2:.0f} um2/PE, "
          f"power={PE_POWER_MW:.2f} mW/PE")

    rows = []

    for design_name, design in DESIGNS.items():
        print(f"\n  Design: {design['label']}")
        total_pes = design["total_pes"]
        n_routers = design["noc_routers"]
        n_spm = design["spm_banks"]
        n_l2 = design["l2_banks"]

        # Component area/power calculations
        pe_total_area = total_pes * PE_AREA_UM2
        pe_total_power = total_pes * PE_POWER_MW
        router_total_area = n_routers * noc_area
        router_total_power = n_routers * noc_power
        spm_total_area = n_spm * spm_area
        spm_total_power = n_spm * spm_power
        # L2 bank: ~4x SPM area (16KB vs 4KB) -- estimated
        l2_area_per = spm_area * 4.0
        l2_power_per = spm_power * 3.5
        l2_total_area = n_l2 * l2_area_per
        l2_total_power = n_l2 * l2_power_per
        # DMA controller: similar to NoC router complexity
        dma_area = noc_area * 1.5
        dma_power = noc_power * 1.2

        system_area = pe_total_area + router_total_area + spm_total_area + l2_total_area + dma_area
        system_power = pe_total_power + router_total_power + spm_total_power + l2_total_power + dma_power
        system_area_mm2 = system_area / 1e6

        components = [
            ("pe_array", pe_total_area, pe_total_power, total_pes * 50,
             PE_POWER_MW * 0.02 * total_pes),
            ("noc_router", router_total_area, router_total_power,
             noc_cells * n_routers, noc_leakage * n_routers),
            ("spm_bank", spm_total_area, spm_total_power,
             spm_cells * n_spm, spm_leakage * n_spm),
            ("l2_bank", l2_total_area, l2_total_power,
             int(spm_cells * 4 * n_l2), spm_leakage * 4 * n_l2),
            ("dma_ctrl", dma_area, dma_power,
             int(noc_cells * 1.5), noc_leakage * 1.5),
            ("system_total", system_area, system_power, 0, 0),
        ]

        for comp_name, area, power, cells, leakage in components:
            row = {
                "design": design_name,
                "component": comp_name,
                "area_um2": round(area, 2),
                "area_mm2": round(area / 1e6, 4),
                "frequency_mhz": target_mhz,
                "timing_met": noc_met,
                "total_power_mw": round(power, 2),
                "leakage_power_mw": round(leakage, 4),
                "cell_count": cells,
                "data_source": "real_dc" if comp_name in ("noc_router", "spm_bank") else "estimated",
                "git_hash": ghash,
                "timestamp": timestamp,
            }
            rows.append(row)

            if comp_name == "system_total":
                print(f"    TOTAL: area={area/1e6:.3f} mm2, power={power:.1f} mW")
            else:
                pct = area / system_area * 100 if system_area > 0 else 0
                print(f"    {comp_name:12s}: area={area:12.0f} um2 ({pct:5.1f}%), "
                      f"power={power:8.2f} mW")

    # Write CSV
    csv_path = out_dir / "synthesis.csv"
    fieldnames = [
        "design", "component", "area_um2", "area_mm2", "frequency_mhz",
        "timing_met", "total_power_mw", "leakage_power_mw", "cell_count",
        "data_source", "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Wrote {len(rows)} rows to {csv_path}")

    # Summary JSON
    summary = {
        "experiment": "E22_rtl_synthesis",
        "timestamp": timestamp,
        "git_hash": ghash,
        "synthesis_tool": tool,
        "pdk": pdk,
        "target_frequency_mhz": target_mhz,
        "timing_met": noc_met,
        "real_components_synthesized": ["noc_router", "tapestry_spm"],
        "estimated_components": ["pe_array", "l2_bank", "dma_ctrl"],
        "note": "PE area estimated from published CGRA data at comparable technology. "
                "L2 and DMA scaled from synthesized SPM and router. "
                "Only noc_router and tapestry_spm have actual DC synthesis results.",
        "per_design_summary": {},
    }
    for design_name, design in DESIGNS.items():
        design_rows = [r for r in rows if r["design"] == design_name and r["component"] == "system_total"]
        if design_rows:
            r = design_rows[0]
            summary["per_design_summary"][design_name] = {
                "label": design["label"],
                "total_pes": design["total_pes"],
                "area_mm2": r["area_mm2"],
                "power_mw": r["total_power_mw"],
            }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
