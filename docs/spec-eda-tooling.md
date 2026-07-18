# EDA Tooling

This document specifies the portable EDA tooling model for Loom. It
defines how Loom discovers, selects, and records backend tool profiles
without embedding machine-specific installation details in public
project specifications.

## Purpose

Loom needs multiple hardware backend flows:

* fast RTL sanity checks;
* RTL simulation;
* logic synthesis and area estimation;
* timing estimation;
* power estimation;
* optional physical or FPGA-oriented evaluation.

The public project contract must describe these flows without assuming
one workstation, one install root, one license setup, or one vendor
environment.

## Public Specification Boundary

Public documents under `docs/` may describe:

* backend tool classes;
* tool profile schemas;
* required tool capabilities;
* generic command roles;
* expected inputs and outputs;
* report schemas;
* failure and diagnostic behavior;
* privacy and reproducibility requirements.

Public documents under `docs/` must not record:

* local absolute installation paths;
* user names or host names;
* license server names or license files;
* credentials, tokens, or private environment variables;
* private PDK paths or proprietary library locations;
* workstation-specific module names as normative requirements.

Local activation commands, local install paths, and private library
roots belong in ignored local configuration, private CI configuration,
or temporary execution guides. They are not part of the public Loom
specification.

## Tool Profiles

A tool profile is a declarative description of a backend capability.
Profiles are selected by capability and fidelity, not by hard-coded
tool name.

A profile identifies:

* profile id;
* tool class;
* supported capabilities;
* expected input artifact kinds;
* produced output artifact kinds;
* required library profile kinds;
* optional activation recipe reference;
* optional environment variable allowlist;
* report parser or adapter kind;
* portability and confidentiality level.

The activation recipe may refer to a local script or environment setup
entry, but public example profiles must use placeholders or portable
open-source commands. Private profiles may contain site-specific
activation commands outside the public repository.

## Target Universe And Layering

The EDA target universe includes every backend profile class that Loom
can discover, describe, activate, execute, parse, normalize, calibrate,
and report without embedding private machine details in public specs.
This includes open-source RTL tools, FPGA flows, ASIC logic synthesis
flows, timing and power analysis tools, physical-estimate flows, formal
or structural checkers, and custom adapters.

Implementation may be layered by capability, but the profile contract is
not limited to the profiles that are currently executable on one
machine. Each profile class has the same objective completion ladder:

* discovery: Loom can identify that the profile exists and which
  capabilities it claims;
* activation: Loom can apply the selected local activation recipe
  without exposing private details in public reports;
* execution: Loom can invoke the backend role on declared inputs;
* parsing: Loom can parse backend-specific logs or reports;
* normalization: Loom can emit stable Loom report records;
* calibration: Loom can use backend reports as calibration inputs when
  the selected FPA model supports calibration;
* reporting: Loom can reference the normalized evidence from report
  bundles and DSE records.

An example or descriptor may declare a target profile class before the
corresponding execution adapter is complete, but it must not count as
passing backend evidence until the required activation, execution,
parsing, normalization, and reporting records exist.

## Tool Classes

Baseline tool classes are:

* `rtl_lint`: static and structural RTL checks;
* `rtl_sim`: RTL simulation;
* `rtl_synth`: logic synthesis and structural area estimation;
* `timing`: timing analysis or critical-path estimation;
* `power`: power estimation from structural data and activity;
* `physical`: placement, routing, extraction, or physical estimates;
* `fpga`: FPGA synthesis, implementation, or prototyping;
* `formal`: equivalence, property, or connectivity checking;
* `format`: RTL formatting or syntax normalization;
* `custom`: user-provided backend adapter.

Tool classes define required capabilities. A concrete profile may
implement multiple classes when a backend supports multiple roles.

Normalized EDA reports record the fidelity class implied by the
executed backend role. Baseline mappings are:

* `rtl_lint` produces `rtl_structural` evidence;
* `rtl_sim` produces `rtl_functional` evidence.

These mappings classify the backend evidence itself. They do not
relabel analytic FPA metrics that merely cite a backend report as a
calibration or structural input.

## Library Profiles

A library profile describes technology or platform data used by an EDA
flow. It may represent standard-cell libraries, SRAM compilers, IO
libraries, FPGA parts, timing corners, power corners, process corners,
or abstract analytical cost tables.

A library profile identifies:

* library profile id;
* technology or platform family;
* library kind;
* supported corners;
* supported voltage and temperature metadata when available;
* logical file roles required by tools;
* optional source release identifiers;
* confidentiality level.

Public library profiles must avoid private paths. Local profiles may
resolve logical file roles to local paths outside the public docs.

## Artifact Boundary

Tool profiles consume and produce explicit artifacts. Baseline artifact
kinds are:

* RTL manifest;
* SystemVerilog source set;
* testbench source set;
* constraint manifest;
* activity data;
* synthesis report;
* timing report;
* power report;
* area report;
* physical estimate report;
* FPGA report;
* normalized FPA report specified in `docs/spec-fpa-estimation.md`.

Structured local process execution is owned by
`docs/spec-evaluation-tool-runner.md`; EDA adapters retain ownership of
tool-specific report parsing and normalization.

Adapters normalize backend-specific output into Loom reports. Backend
log text is evidence, but normalized reports are the stable interface
used by DSE and user-facing summaries.

The global evidence policy in `docs/spec-loom-stack.md` applies to EDA
outputs. In particular, schema fixtures or scaffold descriptors must not
be consumed as backend pass records by FPA, reporting, or DSE.

## Reproducibility

Every backend run must record:

* selected tool profile id;
* selected library profile id;
* input artifact identities when available;
* backend command role;
* backend version when available;
* report parser version;
* success or failure status;
* diagnostics.

Local paths may appear in private run logs, but portable report
summaries should prefer profile ids and artifact identities.

## Error Handling

Tool discovery must distinguish:

* no profile matches the requested capability;
* profile exists but activation failed;
* required library profile is missing;
* backend tool executable is unavailable;
* backend license or permission check failed;
* backend executed but produced unsupported output;
* backend report parser failed;
* backend result failed Loom acceptance checks.

Diagnostics must identify the missing capability or failed profile
without exposing secrets in public reports.

## Relationship To RTL And FPA

RTL lowering is specified in `docs/spec-rtl-lowering.md`. FPA
estimation is specified in `docs/spec-fpa-estimation.md`.

RTL lowering emits portable RTL artifacts. EDA tooling profiles define
how those artifacts are checked or evaluated in a selected environment.
FPA estimation consumes normalized backend reports and activity data.

## Acceptance Criteria

The EDA tooling contract is complete when:

* Loom can select backend flows by capability rather than hard-coded
  workstation details;
* public docs and tracked example profiles do not contain private
  installation paths or license details;
* local users can provide private profiles to activate installed tools;
* backend-specific logs can be normalized into stable Loom reports;
* missing tools, missing libraries, activation failures, and parser
  failures produce structured diagnostics.
