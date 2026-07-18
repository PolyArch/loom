// RUN: echo '{"schema_version":"3.0","kind":"pnr_mapping","config_id":"loom.default","resolved_config_identity":"97cdecd0746efcda044cc79d8d66263c874ca27b13bbd037d425510186e1b81d","workload":"toy","hardware":"route_binding_adg","mapping_id":"toy__route_binding_adg","status":"pass","placed_records":2,"unplaced_records":0,"routed_edges":1,"unrouted_edges":0,"config_records":0,"placements":[{"software":"arith.addi#0","operation":"arith.addi","resource_kind":"fabric.op","hardware":"route_binding_adg::fabric.op#0","schedule":"spatial"},{"software":"arith.cmpi#0","operation":"arith.cmpi","resource_kind":"fabric.op","hardware":"route_binding_adg::fabric.op#1","schedule":"spatial"}],"routes":[{"record_id":"route#0","edge_ref":"arith.cmpi#0.result0->arith.addi#0.operand0","producer_binding":"placement:arith.cmpi#0","consumer_binding":"placement:arith.addi#0","payload_kind":"data","from":"arith.cmpi#0","to":"arith.addi#0","status":"routed","segments":[{"segment_id":"seg0","segment_kind":"resource_edge","source_endpoint":"route_binding_adg::fabric.op#0.result0","sink_endpoint":"route_binding_adg::fabric.op#1.operand0","hardware_ref":"route_binding_adg::ssa_edge#0"}]}],"config_bitstream":[]}' > %t.wrong-edge.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.wrong-edge.json --hardware-mlir %s --output %t.wrong-edge.estimate.json 2>&1 | FileCheck %s --check-prefix=EDGE-BINDING
// RUN: echo '{"schema_version":"3.0","kind":"pnr_mapping","config_id":"loom.default","resolved_config_identity":"97cdecd0746efcda044cc79d8d66263c874ca27b13bbd037d425510186e1b81d","workload":"toy","hardware":"route_binding_adg","mapping_id":"toy__route_binding_adg","status":"pass","placed_records":2,"unplaced_records":0,"routed_edges":1,"unrouted_edges":0,"config_records":0,"placements":[{"software":"arith.addi#0","operation":"arith.addi","resource_kind":"fabric.op","hardware":"route_binding_adg::fabric.op#0","schedule":"spatial"},{"software":"arith.cmpi#0","operation":"arith.cmpi","resource_kind":"fabric.op","hardware":"route_binding_adg::fabric.op#1","schedule":"spatial"}],"routes":[{"record_id":"route#0","edge_ref":"arith.addi#0.result0->arith.cmpi#0.operand0","producer_binding":"placement:arith.addi#0","consumer_binding":"placement:arith.cmpi#0","payload_kind":"data","from":"arith.addi#0","to":"arith.cmpi#0","status":"routed","segments":[{"segment_id":"seg0","segment_kind":"resource_edge","source_endpoint":"route_binding_adg::fabric.op#0.result0","sink_endpoint":"route_binding_adg::fabric.op#1.operand0","hardware_ref":"route_binding_adg::ssa_edge#99"}]}],"config_bitstream":[]}' > %t.bad-ref.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.bad-ref.json --hardware-mlir %s --output %t.bad-ref.estimate.json 2>&1 | FileCheck %s --check-prefix=HARDWARE-REF
// RUN: echo '{"schema_version":"3.0","kind":"pnr_mapping","config_id":"loom.default","resolved_config_identity":"97cdecd0746efcda044cc79d8d66263c874ca27b13bbd037d425510186e1b81d","workload":"toy","hardware":"route_binding_adg","mapping_id":"toy__route_binding_adg","status":"pass","placed_records":2,"unplaced_records":0,"routed_edges":1,"unrouted_edges":0,"config_records":0,"placements":[{"software":"arith.addi#0","operation":"arith.addi","resource_kind":"fabric.op","hardware":"route_binding_adg::fabric.op#0","schedule":"spatial"},{"software":"arith.cmpi#0","operation":"arith.cmpi","resource_kind":"fabric.op","hardware":"route_binding_adg::fabric.op#1","schedule":"spatial"}],"routes":[{"record_id":"route#0","edge_ref":"arith.addi#0.result0->arith.cmpi#0.operand0","producer_binding":"placement:arith.addi#0","consumer_binding":"placement:arith.cmpi#0","payload_kind":"data","from":"arith.addi#0","to":"arith.cmpi#0","status":"routed","segments":[{"segment_id":"seg0","segment_kind":"resource_edge","source_endpoint":"route_binding_adg::fabric.op#0.result0","sink_endpoint":"route_binding_adg::fabric.op#1.operand0","hardware_ref":99}]}],"config_bitstream":[]}' > %t.non-string-ref.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.non-string-ref.json --hardware-mlir %s --output %t.non-string-ref.estimate.json 2>&1 | FileCheck %s --check-prefix=NON-STRING-REF

// EDGE-BINDING: mapping route source endpoint does not match mapped producer
// HARDWARE-REF: mapping route segment hardware_ref route_binding_adg::ssa_edge#99 does not match hardware topology ref route_binding_adg::ssa_edge#0
// NON-STRING-REF: mapping route segment hardware_ref is not a string

module {
  fabric.module @route_binding_adg(%i32a : !fabric.bits<32>,
                                   %i32b : !fabric.bits<32>,
                                   %i32c : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>,
                         %pc = %i32c : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %limit = %pc : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%lhs, %rhs)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %cmp = fabric.op [@arith.cmpi] (%sum, %limit)
               {hw_params = [{predicate = ["slt"]}]}
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
        fabric.yield
      }
    }
    fabric.yield
  }
}
