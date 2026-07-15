# Loom, Compiler for Fabric

## High-Level Source Corpus

Loom's complete high-level source corpus has exactly three suites:
`loombench`, `cmsis-dsp`, and `cmsis-nn`. Membership is owned by
`test/app/manifest.json` for LoomBench and by every tracked `Source/**/*.c`
file in the pinned CMSIS-DSP and CMSIS-NN submodules.

Emit the complete, mechanically counted inventory as deterministic JSON:

```sh
python3 test/corpus_inventory.py list
```

Select a complete suite or explicit canonical cases without creating another
manifest:

```sh
python3 test/corpus_inventory.py list --suite cmsis-dsp
python3 test/corpus_inventory.py list \
  --case loombench:vecadd \
  --case cmsis-nn:ActivationFunctions/arm_relu6_s8.c
```

Inventory rows identify inputs only. Compilation, lowering, simulation,
mapping, and other pipeline attempts must report their own per-case outcomes.
The CMSIS DFG target files under `test/cmsis-dsp` and `test/cmsis-nn` are
explicit smoke subsets and never define complete-suite membership.
