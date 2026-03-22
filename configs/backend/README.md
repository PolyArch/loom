# Backend EDA Tool Configuration

Local configuration for EDA tools used by RTL tests (synthesis, simulation).

## Setup

Copy `template.json` to `config.json` and fill in the values for your machine:

```bash
cp template.json config.json
```

`config.json` is gitignored and will not be committed.

## Fields

### tools.\<name\>

Each tool entry has two optional fields:

| Field    | Description |
|----------|-------------|
| `path`   | Explicit path to the tool binary. Highest priority. Leave empty to use PATH or module loading. |
| `module` | Environment-module spec (e.g. `synopsys/syn/W-2024.09-SP5`). Used as fallback when the tool is not on PATH. |

**Tool resolution order:** explicit path > PATH lookup > module loading.

### synth

| Field             | Description |
|-------------------|-------------|
| `lib_search_path` | Directory containing standard cell `.db` files for DC synthesis (e.g. `saed32rvt_ff0p85v25c.db`). |
| `license_server`  | Synopsys license server address (e.g. `27020@host`). Optional; a built-in default is used if empty. |

**Library path resolution order:** `--lib-search-path` CLI arg > `LOOM_SYNTH_LIB_PATH` env var > config.json value.

## Example (SAED32 on NAS)

```json
{
  "tools": {
    "dc_shell": {
      "path": "",
      "module": "synopsys/syn/W-2024.09-SP5"
    },
    "verilator": {
      "path": "",
      "module": "verilator/5.044"
    }
  },
  "synth": {
    "lib_search_path": "/mnt/nas0/eda.libs/saed32/EDK_08_2025/lib/stdcell_rvt/db_nldm",
    "license_server": ""
  }
}
```

## Missing Tools

When a required tool or library is not available, the corresponding tests are **skipped** (not failed). This allows `make check` to pass on machines without commercial EDA tools.
