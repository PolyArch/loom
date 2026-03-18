module {
  handshake.func @builder_tag_ops(%a: i32) -> (i32) attributes {argNames = ["a"], resNames = ["result"]} {
    return %a : i32
  }
}
