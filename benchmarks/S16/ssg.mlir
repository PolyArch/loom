"builtin.module"() ({
  "tdg.graph"() ({
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "halo_exchange", variants = [{domain_rank = 0 : i64, name = "halo_exchange_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "halo_row_col", unroll_factor = 1 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "stencil_compute", variants = [{domain_rank = 0 : i64, name = "stencil_compute_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "stencil_9pt", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "stencil_5pt_unroll2", unroll_factor = 2 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "boundary_apply", variants = [{domain_rank = 0 : i64, name = "boundary_apply_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "boundary_neumann", unroll_factor = 1 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "residual_check", variants = [{domain_rank = 0 : i64, name = "residual_check_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "residual_max", unroll_factor = 1 : i64}]} : () -> ()
    "tdg.contract"() {consumer = @stencil_compute, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @halo_exchange} : () -> ()
    "tdg.contract"() {consumer = @boundary_apply, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @stencil_compute} : () -> ()
    "tdg.contract"() {consumer = @residual_check, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @boundary_apply} : () -> ()
  }) {sym_name = "jacobi_2d"} : () -> ()
}) : () -> ()
