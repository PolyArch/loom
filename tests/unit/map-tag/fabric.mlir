module {
  fabric.module @map_tag_test(%a: i32) -> (i32) {
    %tagged0 = fabric.add_tag %a {tag = 0 : i64}
        : i32 -> !fabric.tagged<!fabric.bits<32>, i1>
    %mapped = fabric.map_tag %tagged0
        [table_size = 2 : i64]
        attributes {table = [[1 : i64, 0 : i64, 0 : i64],
                             [1 : i64, 1 : i64, 1 : i64]]}
        : !fabric.tagged<!fabric.bits<32>, i1>
          -> !fabric.tagged<!fabric.bits<32>, i1>
    %out = fabric.del_tag %mapped
        : !fabric.tagged<!fabric.bits<32>, i1> -> i32
    fabric.yield %out : i32
  }
}
