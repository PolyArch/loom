// RUN: not loom-synth-config-test %p/multiple_documents.yaml 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: yaml line 4 column 1: multiple YAML documents are not allowed
