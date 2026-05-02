// RUN: loom-synth-config-test %p/workers_auto.yaml | FileCheck %s

// `workers: auto` parses to 0 (meaning std::thread::hardware_concurrency()).

// CHECK: parallelism.workers=0
