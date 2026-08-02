// RUN: not loom-config-test --resolved-json %p/resolved_unknown_key.yaml 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not loom-config-test --resolved-json %p/resolved_duplicate_key.yaml 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not loom-config-test --resolved-json %p/resolved_conflicting_include.yaml 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not loom-config-test --resolved-json %p/resolved_removed_config_id.json 2>&1 | FileCheck %s --check-prefix=REMOVED-ID
// RUN: not loom-config-test --resolved-json %p/resolved_string_numeric.json 2>&1 | FileCheck %s --check-prefix=REMOVED-GLOBAL
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_objective.yaml 2>&1 | FileCheck %s --check-prefix=REMOVED-OBJECTIVE
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_policy.json 2>&1 | FileCheck %s --check-prefix=REMOVED-RANKING
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_algorithm.yaml 2>&1 | FileCheck %s --check-prefix=OBSOLETE
// RUN: not loom-config-test --resolved-json %p/resolved_extra_document.yaml 2>&1 | FileCheck %s --check-prefix=MULTIDOC
// RUN: not loom-config-test --resolved-json %p/resolved_zero_tech_mapping.json 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not loom-config-test --resolved-json %p/resolved_unknown_tech_mapping.json 2>&1 | FileCheck %s --check-prefix=TECHMAP-UNKNOWN

// UNKNOWN: config_unknown_key
// DUPLICATE: config_duplicate_key
// CONFLICT: config_conflicting_sources
// REMOVED-ID: config_unknown_key: config_id
// REMOVED-GLOBAL: config_unknown_key: global
// REMOVED-OBJECTIVE: config_unknown_key: dse.objectives
// REMOVED-RANKING: config_unknown_key: dse.ranking_policy
// OBSOLETE: config_unknown_key: fabric_techmap
// MULTIDOC: config_parse_failed
// ZERO: config_type_mismatch: dse.tech_mapping.match_row_attempt_limit
// TECHMAP-UNKNOWN: config_unknown_key: dse.tech_mapping.wall_time_limit
