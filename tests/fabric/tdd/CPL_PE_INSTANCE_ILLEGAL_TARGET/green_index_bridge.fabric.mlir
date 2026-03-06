// RUN: loom --adg %s

// PE-body instance bridging native index to bits<57> interface.
fabric.pe @addr_inc(%a: !dataflow.bits<57>)
    [latency = [1 : i16, 1 : i16, 1 : i16]]
    -> (!dataflow.bits<57>) {
  ^bb0(%x: index):
  %c1 = arith.constant 1 : index
  %r = arith.addi %x, %c1 : index
  fabric.yield %r : index
}

fabric.module @top(%a: !dataflow.bits<57>, %b: !dataflow.bits<57>) -> (!dataflow.bits<57>) {
  %r = fabric.pe %a, %b : (!dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<57>) {
  ^bb0(%x: index, %y: index):
    %out = fabric.instance @addr_inc(%x) : (index) -> (index)
    %sum = arith.addi %out, %y : index
    fabric.yield %sum : index
  }
  fabric.yield %r : !dataflow.bits<57>
}
