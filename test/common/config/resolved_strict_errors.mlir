// RUN: not loom-config-test --resolved-json %p/resolved_unknown_key.yaml 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not loom-config-test --resolved-json %p/resolved_duplicate_key.yaml 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not loom-config-test --resolved-json %p/resolved_conflicting_include.yaml 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_objective.yaml 2>&1 | FileCheck %s --check-prefix=OBJECTIVE
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_policy.json 2>&1 | FileCheck %s --check-prefix=POLICY
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_algorithm.yaml 2>&1 | FileCheck %s --check-prefix=OBSOLETE
// RUN: not loom-config-test --resolved-json %p/resolved_extra_document.yaml 2>&1 | FileCheck %s --check-prefix=MULTIDOC
// RUN: not loom-config-test --resolved-json %p/resolved_string_numeric.json 2>&1 | FileCheck %s --check-prefix=TYPE
// RUN: not loom-config-test --resolved-json %p/resolved_zero_tech_mapping.json 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_tech_mapping.json 2>&1 | FileCheck %s --check-prefix=TECHMAP-UNKNOWN

// UNKNOWN: config_unknown_key
// DUPLICATE: config_duplicate_key
// CONFLICT: config_conflicting_sources
// OBJECTIVE: config_unknown_objective
// POLICY: config_unknown_policy
// OBSOLETE: config_unknown_key: fabric_techmap
// MULTIDOC: config_parse_failed
// TYPE: config_type_mismatch
// ZERO: config_type_mismatch: dse.tech_mapping.match_row_attempt_limit
// TECHMAP-UNKNOWN: config_unknown_key: dse.tech_mapping.wall_time_limit
