# AXPY Performance
Parameters: `N = 8`, `alpha = 3`

## Cycle + Instruction Count
- Expected cycle count:  
loop cycles = 8 * 1 
**total ≈ 8**
- loads = 16   (2 × 8)
- stores = 8   (1 × N)
- adds = 8
- multiplies = 8

## Data Dependency Graph
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
    alpha --> mult
    mult --> add

    %% Accumulator dependency
    add -->|store| outputy

    %% Iterator dependency
    i_in --> addi
    addi --> i_out

    %% Critical Path N/A
```