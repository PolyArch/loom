# Bitonic Stage Performance
Parameters: `N = 8`, `stage = 1`, `pass = 0` ⇒ `distance = 1`, `block_size = 4`  
`float initial_input[N] = {3.0f, 1.0f, 4.0f, 2.0f, 8.0f, 6.0f, 7.0f, 5.0f};`

## Cycle + Instruction Count
- Expected cycle count:
  - `II = 1` — active iterations are independent. The predicate `(idx_in_block & distance) == 0` selects only the lower element of each compare-pair (even values of i), so iter `i` and its partner iter `i+distance` are never both active, and writes to `inplace[i]` / `inplace[i+distance]` never collide across iters.
  - `inner_cycles = N * II = 8`
  - `outer_cycles ≈ 2` for the loop-invariant prologue: `distance = 1 << pass`, `block_size = 1 << (stage+1)`.
  - **total ≈ 10**
- loads = `2 * N = 16`   (`inplace[i]`, `inplace[partner]` per iter)
- stores = `2 * N = 16`   (predicated swap; counted at worst-case path)
- divs = `1 * N = 8`   (`block_idx = i / block_size`)
- mods = `1 * N = 8`   (`idx_in_block = i % block_size`)
- bitops = `2 * N = 16`   (`block_idx & 1` for `% 2`, `idx_in_block & distance`)
- compares = `4 * N = 32`   (`ascending == 0`, `(idx_in_block & distance) == 0`, `partner < N`, value `>` or `<`)
- adds = `1 * N = 8`   (`partner = i + distance`)

## Data Dependency Graph
Note that `partner = i + distance` and its edges were omitted for readability. The 2 possible values of `should_swap` are assumed to be calculated in parallel and is passed into the multiplexer. 
```mermaid
graph TD
    %% Inputs
    i(("i"))
    block_size(("block_size"))
    inplace_i(("inplace[i]"))
    distance(("distance"))
    inplace_p(("inplace[partner]"))

    %% Control-predicate chain
    div((" / "))
    mod((" % "))
    band_asc((" & 1 "))
    cmp_asc((" == 1 "))
    band_pred((" & "))
    cmp_pred((" == 0 "))
    cmp_in_bounds((" partner < N "))

    %% Data-compare chain
    cmp_gt((" > "))
    cmp_lt((" < "))

    sel_dir((" multiplexer "))

    %% block_idx = i / block_size; ascending = (block_idx & 1) == 0
    i --> div & mod
    block_size --> div & mod
    div -->|block_idx| band_asc
    band_asc -->|control bit| cmp_asc
    cmp_asc --> sel_dir

    %% predicate = (idx_in_block & distance) == 0
    mod -->|idx_in_block| band_pred
    distance --> band_pred
    band_pred --> cmp_pred

    %% in-bounds = partner < N (partner = i + distance, index arith — not a counted op) NOT SHOWN

    %% Loaded values feed the direction-selected compare
    inplace_i --> cmp_gt & cmp_lt
    inplace_p --> cmp_gt & cmp_lt
    cmp_gt -->|if True| sel_dir
    cmp_lt -->|if False| sel_dir

    %% Final swap-enable
    cmp_pred --> and_active
    cmp_in_bounds --> and_active
    sel_dir --> and_active

    %% Predicated swap stores
    inplace_p -.store.-> inplace_i_out
    inplace_i -.store.-> inplace_p_out
    and_active -.enable.-> inplace_i_out
    and_active -.enable.-> inplace_p_out

    %% Critical Path N/A — feed-forward only (II = 1)
```