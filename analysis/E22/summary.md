# E22: RTL Synthesis Results -- Summary

## Methodology
Collected RTL synthesis data from Synopsys DC Ultra (W-2024.09-SP5) at
SAED14nm FinFET, 500MHz target (2.0ns clock period). Two components have
real synthesis results (noc_router, tapestry_spm 4KB). PE array, L2 banks,
and DMA controller areas are estimated from published CGRA data at comparable
technology nodes.

## Synthesis Environment
- Tool: Synopsys DC Ultra W-2024.09-SP5
- PDK: SAED14nm FinFET (TT corner, 0.8V, 25C)
- Target library: saed14rvt_base_tt0p8v25c.db
- Clock: 500 MHz (2.0 ns period)
- Timing: **met** (worst slack = +0.000489 ns on noc_router)

## Real Synthesized Components

| Component   | Cell Area (um2) | Power (mW) | Cells | Timing Met |
|-------------|----------------:|-----------:|------:|-----------:|
| NoC Router  |        4,427.79 |       3.39 | 9,401 | Yes        |
| SPM (4KB)   |       67,090.40 |      38.52 |124,864| Yes        |

Note: SPM synthesized as flip-flop array (no SRAM macros in SAED14nm lib).
With SRAM macros, SPM area would be ~10x smaller.

## System-Level Area Estimates (3 Pareto-optimal designs)

| Design                       | PEs | Area (mm2) | Power (mW) |
|------------------------------|----:|-----------:|-----------:|
| Hetero 2-core (10x10 + 8x8) | 164 |      1.026 |      465.9 |
| Hetero 4-core (2x8x8+2x6x6) | 200 |      1.892 |      908.6 |
| Homo 4-core (4x 8x8)        | 256 |      1.959 |      917.0 |

### Area Breakdown (Design A, representative)
- PE array: 19.2%
- SPM banks: 26.2%
- L2 banks: 52.3% (estimated, dominant component)
- NoC routers: 1.7%
- DMA controller: 0.6%

## Key Findings

1. **500 MHz timing met** with positive slack on both synthesized components.
   The NoC router critical path is 35 levels of logic at 1.88ns, leaving
   margin for the 2.0ns period.

2. **Memory dominates area**: SPM + L2 account for ~78% of total system area.
   This is because SPM is synthesized as flip-flops without SRAM macros.
   With real SRAM, the PE array would become the dominant component.

3. **NoC routers are area-efficient**: only 1.7% of total area. The 5-port
   router with 256-bit links synthesizes to under 5000 um2.

4. **Power is dominated by memory** as well (SPM + L2 = ~90% of total).
   PE array consumes only 24.6-38.4 mW across designs.

5. **Heterogeneous 2-core design** achieves the smallest area (1.026 mm2)
   while mapping all 14 kernels, validating the co-optimization approach.

## Limitations
- PE area is estimated (1200 um2/PE), not synthesized. Real PE synthesis
  requires the complete PE RTL including FU, register file, and switch.
- L2 banks are scaled from SPM (4x area for 4x capacity), not independently
  synthesized.
- No SRAM macros available in SAED14nm educational library.

## Data provenance
- CSV: out/experiments/E22/synthesis.csv
- Source: out/experiments/e8_rtl_synthesis/synthesis_summary.json (real DC data)
- Tool version verified: W-2024.09-SP5 (not X-2025.06-SP3)
