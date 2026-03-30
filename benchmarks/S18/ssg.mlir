"builtin.module"() ({
  "tdg.graph"() ({
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "force_compute", variants = [{domain_rank = 0 : i64, name = "force_compute_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "force_cutoff", unroll_factor = 1 : i64}, {domain_rank = 2 : i64, name = "force_tree", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "force_direct_unroll2", unroll_factor = 2 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "position_update", variants = [{domain_rank = 0 : i64, name = "position_update_default", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "update_verlet_unroll2", unroll_factor = 2 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "neighbor_rebuild", variants = [{domain_rank = 0 : i64, name = "neighbor_rebuild_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "rebuild_verlet_list", unroll_factor = 1 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "energy_reduce", variants = [{domain_rank = 0 : i64, name = "energy_reduce_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "energy_ke_pe", unroll_factor = 1 : i64}]} : () -> ()
    "tdg.contract"() {consumer = @position_update, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @force_compute} : () -> ()
    "tdg.contract"() {consumer = @neighbor_rebuild, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @position_update} : () -> ()
    "tdg.contract"() {consumer = @energy_reduce, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @position_update} : () -> ()
    "tdg.contract"() {consumer = @force_compute, data_type = i32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @neighbor_rebuild} : () -> ()
  }) {sym_name = "nbody_simulation"} : () -> ()
}) : () -> ()
