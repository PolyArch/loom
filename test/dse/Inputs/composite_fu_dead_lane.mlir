// One whole-layer graph whose rendezvous pairs leave their second lane unread.
//
// A pair of chained `dataflow.sync` actors is the densest two-actor shape here
// and costs the fewest boundary ports, because a result no actor consumes is
// not an FU output port. It is also the one shape an FU cannot hold: the
// authored `fabric.op` result would reach nothing. The multiply-accumulate
// motifs below read every result they produce, so they are what mining must
// report instead.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @dead_lane(%start: none, %a: i64, %b: i64, %zp: i64,
                                    %acc: i64) -> i64
      attributes {input_segments = array<i32: 4, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %pa:2 = dataflow.sync %a, %b : (i64, i64) -> (i64, i64)
    %qa:2 = dataflow.sync %pa#0, %zp : (i64, i64) -> (i64, i64)
    %pb:2 = dataflow.sync %b, %zp : (i64, i64) -> (i64, i64)
    %qb:2 = dataflow.sync %pb#0, %acc : (i64, i64) -> (i64, i64)
    %pc:2 = dataflow.sync %zp, %acc : (i64, i64) -> (i64, i64)
    %qc:2 = dataflow.sync %pc#0, %a : (i64, i64) -> (i64, i64)

    %ca = arith.subi %qa#0, %zp : i64
    %cb = arith.subi %qb#0, %zp : i64
    %first = arith.muli %ca, %cb : i64
    %running = arith.addi %first, %acc : i64
    %cc = arith.subi %qc#0, %zp : i64
    %cd = arith.subi %a, %zp : i64
    %second = arith.muli %cc, %cd : i64
    %total = arith.addi %second, %running : i64

    %done:2 = dataflow.sync %start, %total : (none, i64) -> (none, i64)
    dataflow.graph.return values(%done#1 : i64) streams() memories()
        complete(%done#0 : none)
  }

  dataflow.thread private @dead_lane_worker domain(#dataflow.thread_domain<dense>)(
      %a: i64, %b: i64, %zp: i64, %acc: i64) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @dead_lane deps(%ctrl)
        values(%a, %b, %zp, %acc) stream_inputs() memories() stream_outputs()
        : (none, i64, i64, i64, i64) -> (i64, none)
    dataflow.thread.yield %done : none
  }

  func.func @application() {
    %a = arith.constant 3 : i64
    %b = arith.constant 5 : i64
    %zp = arith.constant 1 : i64
    %acc = arith.constant 7 : i64
    %thread = dataflow.thread.launch @dead_lane_worker(%a, %b, %zp, %acc)
        : (i64, i64, i64, i64) -> !dataflow.thread_token
    return
  }
}
