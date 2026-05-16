# Batchnorm Performance
Parameters: `C = 4`, `H = 8`, `W = 8`

## Cycle + Instruction Count
- Expected cycle count:
  - `N = C * H * W = 4 * 8 * 8 = 256`
    -  `II = 1` because there is no loop-carried data recurrence in the pixel loop
  - `inner_cycles = N * II = 256`
  - `outer_cycles = C * 3 = 12` for `variance[c] + epsilon`, `sqrtf`, and reciprocal division
  - **total ≈ 268**
- Operation counts:
  - loads = `N` input loads + `4 * C` channel-parameter loads = `256 + 16 = 272`
  - stores = `N` output stores = `256`
  - adds/subtracts = `4 * N` inner adds/subs + `C` variance/epsilon adds = `1024 + 4 = 1028`
  - multiplies = `5 * N = 1280`
  - divides = `C = 4`
  - transcendentals = `sqrt: C = 4`

Index arithmetic is intentionally counted here: per pixel, `idx = c * (H * W) + h * W + w`
contributes 3 multiplies and 2 adds.

## Data Dependency Graph
```mermaid
graph TD
    subgraph channel["Once per c"]
        direction TB
        one(("1.0"))
        epsilon(("epsilon"))
        variance(("variance[c]"))
        add_var_eps((" + "))
        sqrt_var(("sqrt"))
        div_inv_std((" / "))
        inv_std(("inv_std"))

        variance -->|load| add_var_eps
        epsilon --> add_var_eps
        add_var_eps --> sqrt_var
        one --> div_inv_std
        sqrt_var --> div_inv_std
        div_inv_std --> inv_std
    end

    subgraph pixel["Per-pixel inner loop"]
        direction TB
        c(("c"))
        h(("h"))
        H(("H"))
        W(("W"))
        w(("w"))
        input(("input[idx]"))
        mean(("mean[c]"))
        gamma(("gamma[c]"))
        beta(("beta[c]"))
        mult_cHW((" * "))
        mult_HW((" * "))
        mult_hW((" * "))
        mult_norm((" * "))
        mult_output((" * "))
        add1((" + "))
        add2((" + "))
        add_output((" + "))
        sub(("input[idx] - mean[c]"))
        normalized(("normalized"))
        idx(("idx"))
        output(("output[idx]"))

        H --> mult_HW
        W --> mult_HW & mult_hW
        mult_HW --> mult_cHW
        c --> mult_cHW
        h --> mult_hW
        mult_cHW --> add1
        mult_hW --> add1
        add1 --> add2
        w --> add2
        add2 --> idx

        idx --> input
        input -->|load| sub
        mean -->|load| sub
        sub --> mult_norm
        mult_norm --> normalized

        normalized --> mult_output
        gamma -->|load| mult_output
        mult_output --> add_output
        beta -->|load| add_output
        add_output -->|store| output
    end

    inv_std --> mult_norm

    %% Critical path is feed-forward only; no loop-carried data recurrence, so II = 1.
```
