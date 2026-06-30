// RUN: rm -rf %t.dir
// RUN: timeout 900s bash %S/../e2e/run_cgra_sim_evidence_sweep.sh --output-dir %t.dir --case batchnorm --jobs 1
// RUN: %python %S/assert_batchnorm_cgra_evidence.py %t.dir

// The test body is intentionally empty; the RUN lines above validate the row
// evidence produced by the e2e chain.
