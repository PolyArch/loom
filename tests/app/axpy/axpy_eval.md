# AXPY Performance
Parameters: `N = 8`, `alpha = 3`

## Cycle + Instruction Count

**Loop classification.** `i` (trip = `N` = 8): **parallel** — no carry through register or memory; `alpha` is read-only, `input_x` / `input_y` are read-only, and each iter writes a distinct `output_y[i]`. Fully unrolled → all 8 lanes overlap.

**Critical path (`total_cycles = 4`).** Per-iter chain through `input_x` (the longer of the two load chains):
```
1 (load input_x[i]) + 1 (mul) + 1 (add) + 1 (store output_y[i]) = 4
```
The bare `[i]` subscripts add no address-gen cycle. The `input_y` chain is shorter (no mul): `1 + 1 + 1 = 3`. `alpha` is loop-invariant, hoisted once. The 8 lanes are independent, so `total_cycles` stays at the per-instance depth.

**Op counts (N = 8).**

| op           | algorithmic | overhead | total |
|--------------|-------------|----------|------:|
| loads        | 16 (`input_x[i]`, `input_y[i]`) | 8 (`i` read) + 2 (`alpha`, `N` param hoists) | **26** |
| stores       | 8 (`output_y[i]`) | 8 (`i` write) | **16** |
| adds         | 8 (`α·x + y`)     | 8 (`i++`) | **16** |
| address_adds | 0                 | 0 (bare `[i]` subscripts charge no address add) | **0** |
| mul          | 8                 | 0 | **8** |
| compares     | 0                 | 8 (bound `i < N`) | **8** |

The array accesses (`input_x[i]`, `input_y[i]`, `output_y[i]`) all use bare-variable subscripts, so none charges an address add. The induction var `i` charges (load + add + store + cmp) per iter.

## Data Dependency Graph
Shown is one of 8 parallel instances; lanes share alpha, which is loaded once. 
```mermaid
graph TD
%% Define the input nodes
i_in(("i"))
alpha(("alpha"))
xi(("input_x[i]"))
yi(("input_y[i]"))

%% Define the computation nodes
mult((" * "))
add((" + "))
addi((" + "))

%% Define the final output
i_out(("i"))
outputy(("output_y[i]"))

%% Inner loop dependencies
xi -->|load| mult
yi -->|load| add
alpha -->|loaded once| mult
mult --> add

%% Accumulator dependency
add -->|store| outputy

%% Iterator dependency
i_in -..-> addi
addi -..-> i_out

%% Critical Path N/A
```