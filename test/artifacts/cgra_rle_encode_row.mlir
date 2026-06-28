// RUN: rm -rf %t.dir
// RUN: timeout 900s bash %S/../e2e/run_cgra_sim_evidence_sweep.sh --output-dir %t.dir --case rle_encode --jobs 1
// RUN: %python %S/assert_rle_encode_cgra_evidence.py %t.dir
