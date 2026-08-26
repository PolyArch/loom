# Bounded reverse-FU synthesis

`integer-add-sub.mlir` is a reusable canonical Dataflow graph set for the
public bounded reverse-FU workflow. It contains rooted add and subtract
functions with the exact token-completion boundary admitted by the current
scalar integer provider.

Run the graph set through the production DSE entry point with an existing
ArtifactStore, BlobStore, and resolved configuration:

```sh
loom-dse \
  --config resolved.json \
  --artifact-store artifacts \
  --blob-store blobs \
  --run-root reverse-fu-run \
  --producer-build loom.release.example \
  --fu-reverse-synthesis-dataflow integer-add-sub.mlir \
  --fu-reverse-synthesis-evidence reverse-fu-evidence.json
```

Repeating the command with the same immutable inputs and run root performs
journal replay. The evidence projection records exact Dataflow, Fabric,
TechMapping, SpatialMapping, SystemMapping, ConfigurationABI, and portable RTL
roots together with the current dispatch count.
