"""Generate a synthetic dataflow.graph with N arith.addi ops for the ILP
size-fallback lit test. N is read from sys.argv[1]."""

import sys


def main() -> None:
    n = int(sys.argv[1])
    print('fabric.module @fu_addi {')
    print('  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>')
    print('  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>')
    print('  fabric.spatial_pe(%pa = %a : !fabric.bits<32>, '
          '%pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {')
    print('    fabric.fu(%x = %pa : !fabric.bits<32>, '
          '%y = %pb : !fabric.bits<32>)')
    print('                  -> !fabric.bits<32> {')
    print('      %k = fabric.op [@arith.addi] (%x, %y) : '
          '(!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>')
    print('      fabric.yield %k : !fabric.bits<32>')
    print('    }')
    print('  }')
    print('  fabric.yield')
    print('}')
    print('func.func @graph_big(%a: i32, %b: i32) -> i32 {')
    print('  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {')
    print('    %v0 = arith.addi %x, %y : i32')
    for i in range(1, n):
        print(f'    %v{i} = arith.addi %v{i - 1}, %y : i32')
    print(f'    dataflow.yield %v{n - 1} : i32')
    print('  }')
    print('  return %r : i32')
    print('}')


if __name__ == '__main__':
    main()
