import json
import os
import subprocess
import sys

import lit.formats
from lit.llvm import llvm_config

config.name = "LOOM"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".test"]

# Hardware tools and scale Mapping tests launch nested workers and carry large
# resident sets. Keep enough independent work in flight for this host without
# letting lit's outer worker pool oversubscribe those inner workloads.
lit_config.parallelism_groups["resource-intensive"] = 6

if getattr(config, "loom_have_circt", False):
    config.available_features.add("circt")

gem5_readiness_path = os.path.join(
    config.loom_obj_root, "gem5", "loom-gem5-readiness.json")
try:
    with open(gem5_readiness_path, encoding="utf-8") as readiness_file:
        gem5_readiness = json.load(readiness_file)
    gem5_binary = gem5_readiness.get("binary")
    if isinstance(gem5_binary, str) and os.path.isabs(gem5_binary):
        gem5_directory = os.path.dirname(gem5_binary)
        config.environment["PATH"] = os.pathsep.join(
            [gem5_directory, config.environment["PATH"]])
except (OSError, json.JSONDecodeError):
    pass

catalog_executable = os.path.join(
    config.loom_obj_root, "bin", "loom-backend-tool-catalog")
try:
    catalog_result = subprocess.run(
        [catalog_executable, "--probe-dir", config.loom_obj_root],
        check=True,
        capture_output=True,
        env=config.environment,
        text=True,
        timeout=60,
    )
    catalog_projection = json.loads(catalog_result.stdout)
except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
    lit_config.fatal("backend tool catalog probe failed: {}".format(error))

if (catalog_projection.get("schema") !=
        "loom.external_tool.backend_catalog" or
        catalog_projection.get("version") != "1.0"):
    lit_config.fatal("backend tool catalog projection has an unknown schema")
available_backend_features = catalog_projection.get("available_features")
if not isinstance(available_backend_features, list) or not all(
        isinstance(feature, str) for feature in available_backend_features):
    lit_config.fatal("backend tool catalog projection has invalid features")
config.available_features.update(available_backend_features)

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.loom_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
# %python expands to the interpreter running lit.
config.substitutions.append(("%python", sys.executable))
product_acceleration_profile = lit_config.params.get(
    "loom_product_acceleration_profile", "")
if product_acceleration_profile:
    product_acceleration_profile = os.path.abspath(
        product_acceleration_profile)
    if not os.path.isfile(product_acceleration_profile):
        lit_config.fatal(
            "product acceleration profile does not exist: {}".format(
                product_acceleration_profile))
    product_acceleration_profile = (
        "--loom-accel-profile=" + product_acceleration_profile)
config.substitutions.append(
    ("%loom-product-acceleration-profile", product_acceleration_profile))
# %loom_include is the tracked include root, so a generator anchor can name
# the one canonical registry source without a relative path walk.
config.substitutions.append(
    ("%loom_include", os.path.join(config.loom_src_root, "include")))
config.substitutions.append(
    ("%loom_external", config.loom_external_source_root))

llvm_config.with_system_environment(
    ["HOME", "INCLUDE", "LIB", "TMP", "TEMP",
     "JOBS", "LOOM_TEST_JOBS",
     "LOOM_EXTERNAL_TOOL_CACHE_ROOT", "LOOM_VERBOSE_LEVEL",
     "LOOM_NATIVE_RUNNER_JOBS", "LOCALDOMAIN", "LM_LICENSE_FILE",
     "ALTERAD_LICENSE_FILE", "ALTERA_INSTALL_ROOT_HOME", "QUARTUS_ROOTDIR",
     "QUARTUS_ROOTDIR_OVERRIDE", "QCORE_ROOTDIR", "QSYS_ROOTDIR"])
llvm_config.use_default_substitutions()

config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]

tool_dirs = [
    os.path.join(config.loom_obj_root, "tools", "loom"),
    os.path.join(config.loom_obj_root, "tools", "loom-adg"),
    os.path.join(config.loom_obj_root, "tools", "loom-cc"),
    os.path.join(config.loom_obj_root, "tools", "loom-payload"),
    os.path.join(config.loom_obj_root, "tools", "loom-raise-opt"),
    os.path.join(config.loom_obj_root, "tools", "loom-tblgen"),
    os.path.join(config.loom_obj_root, "tools", "loom-config-test"),
    os.path.join(config.loom_obj_root, "tools", "loom-dfg-sim"),
    os.path.join(config.loom_obj_root, "tools", "loom-dse"),
    os.path.join(config.loom_obj_root, "test", "adg"),
    os.path.join(config.loom_obj_root, "test", "dataflow"),
    os.path.join(config.loom_obj_root, "test", "deployment"),
    os.path.join(config.loom_obj_root, "test", "dse"),
    os.path.join(config.loom_obj_root, "test", "eda"),
    os.path.join(config.loom_obj_root, "test", "evaluation"),
    os.path.join(config.loom_obj_root, "test", "fabric"),
    os.path.join(config.loom_obj_root, "test", "frontend"),
    os.path.join(config.loom_obj_root, "test", "hardware"),
    os.path.join(config.loom_obj_root, "test", "mapping"),
    os.path.join(config.loom_obj_root, "test", "simulator"),
    os.path.join(config.loom_obj_root, "test", "system"),
    os.path.join(config.loom_obj_root, "bin"),
    config.llvm_tools_dir,
]
tools = [
    "loom",
    "loom-adg",
    "loom-backend-tool-catalog",
    "loom-tblgen",
    "loom-config-test",
    "loom-compiler-target-binding-test",
    "loom-adg-builder-api-test",
    "loom-adg-mesh-switch-network-test",
    "loom-dataflow-activity-definedness-test",
    "loom-dataflow-ordered-cardinality-handshake-test",
    "loom-dataflow-rewrite-decision-test",
    "loom-dataflow-sync-rewrite-test",
    "loom-dataflow-cardinality-rewrite-test",
    "loom-dataflow-fanout-rewrite-test",
    "loom-dataflow-graph-definition-rewrite-test",
    "loom-dataflow-vector-decomposition-rewrite-test",
    "loom-deployment-closure-test",
    "loom-deployment-executable-leaves-test",
    "loom-deployment-package-test",
    "loom-dataflow-canonical-artifact-test",
    "loom-dataflow-channel-create-test",
    "loom-dataflow-canonical-reference-test",
    "loom-dataflow-memory-effect-test",
    "loom-dataflow-operation-schema-codec-test",
    "loom-dataflow-operation-schema-test",
    "loom-dataflow-service-schema-test",
    "loom-dse-invocation-manifest-test",
    "loom-dse",
    "loom-dse-model-parameter-training-test",
    "loom-dse-ground-truth-campaign-integration-test",
    "loom-dse-portable-spatial-core-rtl-generator-test",
    "loom-mapped-rtl-simulation-test",
    "loom-openroad-routed-test",
    "loom-openroad-static-fpa-test",
    "loom-evaluation-production-registry-test",
    "loom-evaluation-fabric-fast-evaluation-test",
    "loom-dfg-sim",
    "loom-dfg-run",
    "loom-system-run",
    "loom-pre-mapping",
    "loom-tech-map",
    "loom-tech-mapping-activity-definedness-test",
    "loom-fabric-artifact-gate-test",
    "loom-fabric-artifact-codec-test",
    "loom-fabric-artifact-finalizer-test",
    "loom-fabric-behavior-relation-finalization-test",
    "loom-fabric-pe-configuration-test",
    "loom-fabric-module-domain-import-test",
    "loom-fabric-system-attachment-finalizer-test",
    "loom-fabric-module-boundary-transport-passthrough-test",
    "loom-fabric-handshake-model-test",
    "loom-fabric-boundary-data-path-test",
    "loom-fabric-boundary-transfer-test",
    "loom-fabric-canonical-labeling-test",
    "loom-fabric-elaboration-alias-chain-test",
    "loom-fabric-fixed-vector-float-behavior-test",
    "loom-fabric-fixed-vector-integer-behavior-test",
    "loom-fabric-float-behavior-profile-test",
    "loom-fabric-implementation-family-behavior-inventory-test",
    "loom-fabric-implementation-family-fixed-behavior-test",
    "loom-fabric-implementation-family-test",
    "loom-fabric-scalar-float-behavior-test",
    "loom-fabric-scalar-float-compare-behavior-test",
    "loom-fabric-scalar-integer-behavior-test",
    "loom-fabric-special-math-implementation-family-test",
    "loom-fabric-elaboration-api-test",
    "loom-fabric-elaboration-header-test",
    "loom-fabric-fifo-resource-contract-test",
    "loom-fabric-memory-capability-domain-test",
    "loom-fabric-memory-actor-contract-domain-test",
    "loom-fabric-memory-capability-finalization-test",
    "loom-fabric-memory-port-transaction-test",
    "loom-fabric-memory-operation-port-test",
    "loom-fabric-memory-role-bindings-test",
    "loom-fabric-memory-configuration-test",
    "loom-fabric-memory-consistency-contract-test",
    "loom-fabric-physical-timing-profile-test",
    "loom-fabric-physical-identity-test",
    "loom-fabric-system-physical-identity-test",
    "loom-fabric-persistent-ref-test",
    "loom-fabric-physical-tag-test",
    "loom-fabric-resource-contract-record-test",
    "loom-fabric-resource-contract-test",
    "loom-fabric-operation-resource-contract-test",
    "loom-fabric-system-contract-test",
    "loom-fabric-service-leg-carrier-attachment-test",
    "loom-fabric-system-service-contract-test",
    "loom-fabric-temporal-operand-buffer-test",
    "loom-fabric-temporal-switch-route-test",
    "loom-fabric-switch-resource-contract-test",
    "loom-hardware-configuration-diagnostics-test",
    "loom-constant-callback-specialization-test",
    "loom-candidate-hint-test",
    "loom-frontend-dfg-integration-test",
    "loom-static-global-memory-test",
    "loom-lower",
    "loom-raise",
    "loom-raise-opt",
    "loom-simulator-atomic-order-test",
    "loom-simulator-cgra-event-queue-test",
    "loom-simulator-cgra-compute-runtime-test",
    "loom-simulator-cgra-graph-activation-test",
    "loom-simulator-cgra-physical-action-test",
    "loom-simulator-cgra-resource-runtime-test",
    "loom-simulator-cgra-transport-storage-runtime-test",
    "loom-simulator-cgra-transport-runtime-test",
    "loom-simulator-spatial-trace-test",
    "loom-simulator-spatial-observation-comparison-test",
    "loom-simulator-system-simulation-artifact-test",
    "loom-simulator-dfg-actor-transition-probe-test",
    "loom-simulator-dynamic-work-test",
    "loom-simulator-dfg-evaluation-test",
    "loom-simulator-dfg-execution-session-test",
    "loom-simulator-operation-schema-projection-test",
    "loom-simulator-simulation-execution-test",
    "loom-simulator-simulation-wire-test",
    "loom-simulator-spatial-store-test",
    "loom-simulator-structured-program-native-execution-test",
    "loom-simulator-structured-program-wire-test",
    "loom-simulator-synchronization-test",
    "loom-simulator-vector-boundary-test",
    "loom-simulator-vector-structure-test",
    "loom-structured-address-index-narrowing-test",
    "loom-structured-call-ownership-test",
    "loom-structured-program-artifact-test",
    "loom-structured-thread-domain-test",
    "loom-structured-ownership-lineage-index-test",
    "loom-structured-schedule-generator-test",
    "loom-structured-memory-communication-generator-test",
    "loom-structured-memory-communication-lineage-test",
    "loom-structured-memory-channel-test",
    "loom-source-backed-attention-channel-test",
    "loom-system-execution-matrix-test",
    "loom-heterogeneous-system-anchor-test",
    "loom-product-mapping-inspection-test",
    "loom-structured-memory-layout-test",
    "loom-structured-memory-pipeline-test",
    "loom-structured-execution-shape-generator-test",
    "loom-structured-special-math-accuracy-generator-test",
    "loom-dataflow-rewrite-generator-test",
    "loom-pre-mapping-compilation-test",
    "loom-pointer-service-boundary-test",
    "mlir-opt",
    "mlir-translate",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)

config.substitutions.append(
    (
        "%loom-gem5-readiness",
        gem5_readiness_path,
    )
)

# Loom driver tools share name prefixes with other substitutions. Lit's
# substitution is substring-based, so match their complete placeholders.
_loom_cc_dir = os.path.join(config.loom_obj_root, "bin")
config.substitutions.insert(
    0,
    (
        "%loom-runtime-deployment-loader-test\\b",
        os.path.join(
            config.loom_obj_root,
            "test",
            "runtime",
            "loom-runtime-deployment-loader-test",
        ),
    ),
)
config.substitutions.insert(
    0,
    (
        "%loom-runtime-platform-binding-test\\b",
        os.path.join(
            config.loom_obj_root,
            "test",
            "runtime",
            "loom-runtime-platform-binding-test",
        ),
    ),
)
config.substitutions.insert(
    0,
    (
        "%loom-runtime-gem5-bridge-test\\b",
        os.path.join(
            config.loom_obj_root,
            "test",
            "runtime",
            "loom-runtime-gem5-bridge-test",
        ),
    ),
)
config.substitutions.insert(
    0,
    (
        "%loom-runtime-gem5-spatial-channel-test\\b",
        os.path.join(
            config.loom_obj_root,
            "test",
            "runtime",
            "loom-runtime-gem5-spatial-channel-test",
        ),
    ),
)
config.substitutions.insert(
    0,
    (
        "%loom-runtime-ordered-channel-abi-test\\b",
        os.path.join(
            config.loom_obj_root,
            "test",
            "runtime",
            "loom-runtime-ordered-channel-abi-test",
        ),
    ),
)
config.substitutions.insert(
    0,
    (
        "%loom-runtime-gem5-simulation-binding-test\\b",
        os.path.join(
            config.loom_obj_root,
            "test",
            "runtime",
            "loom-runtime-gem5-simulation-binding-test",
        ),
    ),
)
config.substitutions.insert(
    0, ("%loom-c\\+\\+", os.path.join(_loom_cc_dir, "loom-c++")))
config.substitutions.insert(
    1, ("%loom-cc\\b", os.path.join(_loom_cc_dir, "loom-cc")))
config.substitutions.insert(
    2,
    (
        "%loom-application-build\\b",
        os.path.join(config.loom_obj_root, "bin", "loom-application-build"),
    ),
)
config.substitutions.insert(
    3, ("%loom-raise\\b", os.path.join(_loom_cc_dir, "loom-raise")))
config.substitutions.insert(
    4, ("%loom-lower\\b", os.path.join(_loom_cc_dir, "loom-lower")))
config.substitutions.insert(
    5, ("%loom-payload\\b", os.path.join(_loom_cc_dir, "loom-payload")))
config.substitutions.insert(
    6, ("%objdump-h",
        os.path.join(config.llvm_tools_dir, "llvm-objdump") + " -h"))

# Source-backed execution anchors must enter the runner with the exact shared
# InstructionCore target. The runner's CompilerTargetBinding remains
# authoritative and rejects this invocation projection if it drifts.
_riscv64_c_headers = os.path.join(
    config.loom_src_root, "test", "frontend", "Inputs", "minimal-c-runtime")
_riscv64_cc = " ".join([
    os.path.join(_loom_cc_dir, "loom-cc"),
    "--target=riscv64-unknown-elf",
    "-march=rv64imafdc_zicsr_zifencei",
    "-mabi=lp64d",
    "-mcmodel=medany",
    "-mcpu=generic-rv64",
    "-isystem",
    _riscv64_c_headers,
])
config.substitutions.insert(0, ("%loom-riscv64-cc\\b", _riscv64_cc))
