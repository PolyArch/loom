"builtin.module"() ({
  "tdg.graph"() ({
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "spmv", variants = [{domain_rank = 0 : i64, name = "spmv_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "spmv_ell", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "spmv_csr_unroll2", unroll_factor = 2 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "dot_pq", variants = [{domain_rank = 0 : i64, name = "dot_pq_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "dot_tree_reduce", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "dot_sequential_unroll2", unroll_factor = 2 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "axpy_x", variants = [{domain_rank = 0 : i64, name = "axpy_x_default", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "axpy_unroll4", unroll_factor = 4 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "axpy_r", variants = [{domain_rank = 0 : i64, name = "axpy_r_default", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "axpy_unroll4", unroll_factor = 4 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "precondition", variants = [{domain_rank = 0 : i64, name = "precondition_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "precond_block4", unroll_factor = 1 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "dot_rz", variants = [{domain_rank = 0 : i64, name = "dot_rz_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "dot_tree_reduce", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "dot_sequential_unroll2", unroll_factor = 2 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "axpy_p", variants = [{domain_rank = 0 : i64, name = "axpy_p_default", unroll_factor = 1 : i64}, {domain_rank = 0 : i64, name = "axpy_unroll4", unroll_factor = 4 : i64}]} : () -> ()
    "tdg.kernel"() ({
    ^bb0:
    }) {kernel_type = "auto", sym_name = "convergence_check", variants = [{domain_rank = 0 : i64, name = "convergence_check_default", unroll_factor = 1 : i64}, {domain_rank = 1 : i64, name = "conv_max", unroll_factor = 1 : i64}]} : () -> ()
    "tdg.contract"() {consumer = @dot_pq, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @spmv} : () -> ()
    "tdg.contract"() {consumer = @axpy_r, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @spmv} : () -> ()
    "tdg.contract"() {consumer = @axpy_x, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @dot_pq} : () -> ()
    "tdg.contract"() {consumer = @axpy_r, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @dot_pq} : () -> ()
    "tdg.contract"() {consumer = @precondition, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @axpy_r} : () -> ()
    "tdg.contract"() {consumer = @convergence_check, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @axpy_r} : () -> ()
    "tdg.contract"() {consumer = @dot_rz, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @precondition} : () -> ()
    "tdg.contract"() {consumer = @axpy_p, data_type = f32, ordering = "FIFO", placement = "LOCAL_SPM", producer = @dot_rz} : () -> ()
  }) {sym_name = "conjugate_gradient"} : () -> ()
}) : () -> ()
