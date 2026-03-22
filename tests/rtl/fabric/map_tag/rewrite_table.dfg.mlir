// Companion DFG for rewrite_table.fabric.mlir.
//
// NOTE: map_tag is an infrastructure-only ADG (add_tag + map_tag + del_tag)
// with no compute FU. The mapper may not be able to map a pass-through DFG
// to this ADG since there is no FunctionUnit to host the identity operation.
// Behaviour verification for this case may require manual golden traces.
module {
  handshake.func @map_tag_test(%in: i32) -> (i32)
      attributes {argNames = ["in"], resNames = ["out"]} {
    handshake.return %in : i32
  }
}
