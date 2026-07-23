// RUN: loom-dfg-sim %s --graph ranked_add --arg 0=6618611909121 \
// RUN:   --arg 1=66186119091210 --output %t.add.json
// RUN: FileCheck %s --check-prefix=ADD < %t.add.json
// RUN: loom-dfg-sim %s --graph ranked_compare --arg 0=6618611909121 \
// RUN:   --arg 1=9934292977920 --output %t.compare.json
// RUN: FileCheck %s --check-prefix=COMPARE < %t.compare.json
// RUN: loom-dfg-sim %s --graph ranked_select --arg 0=13 \
// RUN:   --arg 1=6618611909121 --arg 2=66186119091210 --output %t.select.json
// RUN: FileCheck %s --check-prefix=SELECT < %t.select.json
// RUN: loom-dfg-sim %s --graph scalar_condition_select --arg 0=1 \
// RUN:   --arg 1=6618611909121 --arg 2=66186119091210 \
// RUN:   --output %t.scalar-condition.json
// RUN: FileCheck %s --check-prefix=SCALAR-CONDITION < %t.scalar-condition.json

// One firing of a rank-2 elementwise primitive publishes one complete vector
// token covering every flattened lane. Operand lanes {1,2,3,4,5,6} and
// {10,20,30,40,50,60} occupy ascending bit slices, so the sums
// {11,22,33,44,55,66} do too. A lane count taken from the leading dimension
// alone would publish 0x160B and leave the four high lanes zero.
// ADD-DAG: "graph": "ranked_add"
// ADD-DAG: "status": "pass"
// ADD-DAG: "arith.addi": 1
// ADD-DAG: "vector<2x3xi8>:0x42372C21160B"

// A comparison keeps the operand shape and publishes a same-shape 'i1' vector.
// Lanes {1,2,3,4,5,6} above {0,9,1,2,9,9} give the asymmetric active pattern
// {1,0,1,1,0,0}, which reads 0xD with flattened lane zero in the lowest bit.
// COMPARE-DAG: "graph": "ranked_compare"
// COMPARE-DAG: "status": "pass"
// COMPARE-DAG: "arith.cmpi": 1
// COMPARE-DAG: "vector<2x3xi1>:0xD"

// A same-shape 'i1' condition selects lane by lane. Condition {1,0,1,1,0,0}
// keeps true lanes {1,3,4} and takes false lanes {20,50,60}, so the published
// vector is {1,20,3,4,50,60}.
// SELECT-DAG: "graph": "ranked_select"
// SELECT-DAG: "status": "pass"
// SELECT-DAG: "arith.select": 1
// SELECT-DAG: "vector<2x3xi8>:0x3C3204031401"

// A scalar condition is not a vector operand. The selection stays unsupported
// rather than being broadcast across the lanes.
// SCALAR-CONDITION-DAG: "event_count": 0
// SCALAR-CONDITION-DAG: "status": "unsupported"
// SCALAR-CONDITION-DAG: "unsupported op: arith.select: vector primitive operands must be fixed-size and positive-rank"

module {
  dataflow.graph private @ranked_add(
      %start: none, %packed_lhs: i48, %packed_rhs: i48) -> vector<2x3xi8>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.unpack %packed_lhs : i48 -> vector<2x3xi8>
    %rhs = dataflow.unpack %packed_rhs : i48 -> vector<2x3xi8>
    %sum = arith.addi %lhs, %rhs : vector<2x3xi8>
    %published:2 = dataflow.sync %start, %sum
        : (none, vector<2x3xi8>) -> (none, vector<2x3xi8>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<2x3xi8>
  }

  dataflow.graph private @ranked_compare(
      %start: none, %packed_lhs: i48, %packed_rhs: i48) -> vector<2x3xi1>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.unpack %packed_lhs : i48 -> vector<2x3xi8>
    %rhs = dataflow.unpack %packed_rhs : i48 -> vector<2x3xi8>
    %active = arith.cmpi sgt, %lhs, %rhs : vector<2x3xi8>
    %published:2 = dataflow.sync %start, %active
        : (none, vector<2x3xi1>) -> (none, vector<2x3xi1>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<2x3xi1>
  }

  dataflow.graph private @ranked_select(
      %start: none, %packed_condition: i6, %packed_true: i48,
      %packed_false: i48) -> vector<2x3xi8>
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %condition = dataflow.unpack %packed_condition : i6 -> vector<2x3xi1>
    %taken = dataflow.unpack %packed_true : i48 -> vector<2x3xi8>
    %untaken = dataflow.unpack %packed_false : i48 -> vector<2x3xi8>
    %selected = arith.select %condition, %taken, %untaken
        : vector<2x3xi1>, vector<2x3xi8>
    %published:2 = dataflow.sync %start, %selected
        : (none, vector<2x3xi8>) -> (none, vector<2x3xi8>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<2x3xi8>
  }

  dataflow.graph private @scalar_condition_select(
      %start: none, %condition: i1, %packed_true: i48, %packed_false: i48)
      -> vector<2x3xi8>
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %taken = dataflow.unpack %packed_true : i48 -> vector<2x3xi8>
    %untaken = dataflow.unpack %packed_false : i48 -> vector<2x3xi8>
    %selected = arith.select %condition, %taken, %untaken : vector<2x3xi8>
    %published:2 = dataflow.sync %start, %selected
        : (none, vector<2x3xi8>) -> (none, vector<2x3xi8>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<2x3xi8>
  }
}
