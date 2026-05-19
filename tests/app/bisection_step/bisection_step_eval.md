# Bisection Step Performance
Parameters: `N = 64`

## Cycle + Instruction Count
- Expected cycle count:  
loop cycles = 64 * 1  
**total ≈ 64**
- loads = 256   (4 × 64)
- stores = 128   (2 × 64)
- adds = 64
- multiplies = 128   (2 × 64)
- compares = 64

## Data Dependency Graph
```mermaid
graph TD
    %% Define the input nodes
    inputa(("input_a[i]"))
    inputb(("input_b[i]"))
    inputfa(("input_fa[i]"))
    inputfc(("input_fc[i]"))

    %% Define the computation nodes
    mult((" * "))
    add1((" + "))
    mult1((" * 0.5 = c"))
    mult2((" * "))
    cmp_lt((" < "))
    
    %% Define the final output
    output1(("output_a[i] = input_a[i]<br>output_b[i] = c"))
    output2(("output_b[i] = input_b[i]<br>output_a[i] = c"))

    %% Calculating c
    inputa -->|load| add1
    inputb -->|load| add1
    add1 --> mult1

    %% Comparison
    inputfa -->|load| mult2
    inputfc -->|load| mult2
    mult2 --> cmp_lt
    cmp_lt-.T -> 2 stores.-> output1
    cmp_lt-.F -> 2 stores.-> output2

    %% Accumulator dependency

    %% Critical Path N/A
```