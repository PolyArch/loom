// Companion DFG for constant_tag.fabric.mlir.
// Minimal pass-through kernel for tag-attachment verification.
//
// NOTE: add_tag is an infrastructure-only ADG with no compute FU.
// The mapper may not be able to map a pass-through DFG to this ADG
// since there is no FunctionUnit to host the identity operation.
// Behaviour verification for this case may require manual golden traces.
module {
  handshake.func @tag_test(%in: i32) -> (i32)
      attributes {argNames = ["in"], resNames = ["out"]} {
    handshake.return %in : i32
  }
}
