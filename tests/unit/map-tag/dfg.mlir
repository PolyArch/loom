module {
  handshake.func @map_tag_test(%arg0: i32, ...) -> (i32)
      attributes {argNames = ["a"], resNames = ["out"]} {
    return %arg0 : i32
  }
}
