// RUN: loom %s -split-input-file -verify-diagnostics
// RUN: %python -c 'n=50000; p=chr(37); lines=[f"dataflow.thread private @t_deep() ctrl ({p}ctrl: none) iv ({p}i: index) {{", "  dataflow.thread.yield", "}", "func.func @deep_extent_chain() {", f"  {p}v0 = arith.constant {-n - 1} : index", f"  {p}one = arith.constant 1 : index"]; lines += [f"  {p}v{i} = arith.addi {p}v{i - 1}, {p}one : index" for i in range(1, n + 1)]; lines += ["  // expected-error @+1 {{grid upper bound #0 must be nonnegative}}", f"  {p}result = dataflow.thread.launch @t_deep() grid({p}v{n}) : () -> !dataflow.thread_token", "  return", "}"]; print("\n".join(lines))' > %t.deep.mlir
// RUN: timeout 5s loom %t.deep.mlir -verify-diagnostics
// RUN: %python -c 'n=6400; p=chr(37); lines=[f"dataflow.thread private @t_shared() ctrl ({p}ctrl: none) iv ({p}i: index) {{", "  dataflow.thread.yield", "}", "func.func @shared_extent_dag() {", f"  {p}v0 = arith.constant 0 : index", f"  {p}zero = arith.constant 0 : index"]; lines += [f"  {p}v{i} = arith.addi {p}v{i - 1}, {p}zero : index" for i in range(1, n + 1)]; lines += [f"  {p}result{i} = dataflow.thread.launch @t_shared() grid({p}v{n}) : () -> !dataflow.thread_token" for i in range(n)]; lines += ["  return", "}"]; print("\n".join(lines))' > %t.shared.mlir
// RUN: timeout 5s loom %t.shared.mlir

// -----
// Thread definitions require explicit private visibility.
// expected-error @+1 {{requires explicit 'private' visibility}}
dataflow.thread @t_missing_visibility() ctrl (%ctrl: none) {
  dataflow.thread.yield
}

// -----
// Generic syntax cannot bypass the result-free thread ABI.
// expected-error @+1 {{must not declare function results}}
"dataflow.thread"() <{function_type = () -> i32,
                     sym_name = "t_with_result",
                     sym_visibility = "private"}> ({
^bb0(%ctrl: none):
  dataflow.thread.yield
}) : () -> ()

// -----
// Launch rank must match the callee's logical domain rank.
dataflow.thread private @t_rank_one() ctrl (%ctrl: none) iv (%i: index) {
  dataflow.thread.yield
}
func.func @launch_rank_mismatch() {
  // expected-error @+1 {{grid upper bound count (0) must match callee rank (1)}}
  %token = dataflow.thread.launch @t_rank_one() : () -> !dataflow.thread_token
  return
}

// -----
// Statically known launch extents must be nonnegative.
dataflow.thread private @t_nonnegative_extent() ctrl (%ctrl: none) iv (%i: index) {
  dataflow.thread.yield
}
func.func @launch_negative_extent() {
  %extent = arith.constant -1 : index
  // expected-error @+1 {{grid upper bound #0 must be nonnegative}}
  %token = dataflow.thread.launch @t_nonnegative_extent() grid(%extent) : () -> !dataflow.thread_token
  return
}

// -----
// Foldable launch extents must satisfy the same nonnegative contract.
dataflow.thread private @t_foldable_extent() ctrl (%ctrl: none)
    iv (%i: index) {
  dataflow.thread.yield
}
func.func @launch_foldable_negative_extent() {
  %minus_two = arith.constant -2 : index
  %one = arith.constant 1 : index
  %extent = arith.addi %minus_two, %one : index
  // expected-error @+1 {{grid upper bound #0 must be nonnegative}}
  %token = dataflow.thread.launch @t_foldable_extent() grid(%extent)
      : () -> !dataflow.thread_token
  return
}

// -----
// Malformed extent producers must be diagnosed before constant evaluation.
dataflow.thread private @t_malformed_extent() ctrl (%ctrl: none)
    iv (%i: index) {
  dataflow.thread.yield
}
func.func @launch_malformed_extent() {
  // expected-error @+1 {{expected 2 operands, but found 0}}
  %extent = "arith.addi"() : () -> index
  %token = dataflow.thread.launch @t_malformed_extent() grid(%extent)
      : () -> !dataflow.thread_token
  return
}

// -----
// An overflowing nsw/nuw addition is poison, so it yields no static extent,
// while a flagged addition that stays in range still folds.
dataflow.thread private @t_overflow_extent() ctrl (%ctrl: none)
    iv (%i: index) {
  dataflow.thread.yield
}
func.func @launch_signed_overflow_extent() {
  %max = arith.constant 9223372036854775807 : index
  %one = arith.constant 1 : index
  %extent = arith.addi %max, %one overflow<nsw> : index
  %token = dataflow.thread.launch @t_overflow_extent() grid(%extent)
      : () -> !dataflow.thread_token
  return
}
func.func @launch_unsigned_overflow_extent() {
  %minus_one = arith.constant -1 : index
  %extent = arith.addi %minus_one, %minus_one overflow<nuw> : index
  %token = dataflow.thread.launch @t_overflow_extent() grid(%extent)
      : () -> !dataflow.thread_token
  return
}
func.func @launch_flagged_in_range_extent() {
  %minus_two = arith.constant -2 : index
  %one = arith.constant 1 : index
  %extent = arith.addi %minus_two, %one overflow<nsw> : index
  // expected-error @+1 {{grid upper bound #0 must be nonnegative}}
  %token = dataflow.thread.launch @t_overflow_extent() grid(%extent)
      : () -> !dataflow.thread_token
  return
}

// -----
// Dynamic launch extents remain statically admissible.
dataflow.thread private @t_dynamic_extent() ctrl (%ctrl: none) iv (%i: index) {
  dataflow.thread.yield
}
func.func @launch_dynamic_extent(%extent: index) {
  %token = dataflow.thread.launch @t_dynamic_extent() grid(%extent) : () -> !dataflow.thread_token
  return
}

// -----
// Wait tokens must come directly from a thread launch.
func.func @wait_rejects_forged_token() {
  %token = ub.poison : !dataflow.thread_token
  // expected-error @+1 {{operand #0 must be produced directly by dataflow.thread.launch}}
  dataflow.thread.wait %token : !dataflow.thread_token
  return
}

// -----
// Launch's body operand types must match callee's function inputs.
dataflow.thread private @t_int(%x: i32) ctrl (%c: none) {
  dataflow.thread.yield
}
func.func @launch_type_mismatch(%y: f32) {
  // expected-error @+1 {{body operand #0 type 'f32' does not match callee input type 'i32'}}
  %token = dataflow.thread.launch @t_int(%y) : (f32) -> !dataflow.thread_token
  return
}

// -----
// Launch must reference a real dataflow.thread symbol.
func.func @launch_unknown_callee() {
  // expected-error @+1 {{'unknown_thread' does not reference a valid 'dataflow.thread' op}}
  %token = dataflow.thread.launch @unknown_thread() : () -> !dataflow.thread_token
  return
}

// -----
// A launch always returns exactly one completion token.
dataflow.thread private @t_launch_result() ctrl (%ctrl: none) {
  dataflow.thread.yield
}
func.func @launch_requires_completion_token() {
  // expected-error @+1 {{requires one result}}
  dataflow.thread.launch @t_launch_result() : () -> ()
  return
}

// -----
// A wait consumes at least one thread completion token.
func.func @wait_requires_completion_token() {
  // expected-error @+1 {{expected 1 or more operands, but found 0}}
  "dataflow.thread.wait"() : () -> ()
  return
}

// -----
// A wait cannot consume a dataflow control value.
func.func @wait_rejects_control(%ctrl: none) {
  // expected-error @+1 {{must be variadic of one-shot async completion handle for a dataflow.thread.launch, but got 'none'}}
  dataflow.thread.wait %ctrl : none
  return
}

// -----
// Completion frontier operands are none-typed.
dataflow.thread private @t_rejects_non_none_frontier(%value: i32) ctrl (%ctrl: none) {
  // expected-error @+1 {{must be variadic of none type, but got 'i32'}}
  dataflow.thread.yield %value : i32
}

// -----
// A body-carrying thread must have the `thread_ctrl` slot per spec
// section 5.4.1's `(args_*, thread_ctrl, iv_*)` layout. A thread
// whose entry block has only the function-input args (no ctrl, no
// ivs) is rejected.
// expected-error @+1 {{entry block must have at least 1 arguments (function inputs + 1 thread_ctrl slot)}}
"dataflow.thread"() <{function_type = () -> (), sym_name = "t_no_ctrl",
                     sym_visibility = "private"}> ({
^bb0:
  dataflow.thread.yield
}) : () -> ()

// -----
// thread_ctrl must sit immediately after the function-input args.
// Putting an `index` iv between them is rejected because slot N
// (here index 0, since function_type is empty) must be `none`.
// expected-error @+1 {{entry block argument #0 (thread_ctrl) must have type `none`, got 'index'}}
"dataflow.thread"() <{function_type = () -> (),
                     sym_name = "t_ctrl_wrong_position",
                     sym_visibility = "private"}> ({
^bb0(%i: index, %c: none, %j: index):
  dataflow.thread.yield
}) : () -> ()

// -----
// Grid iv slots must be `index`-typed. A non-index trailing slot
// after the thread_ctrl is rejected.
// expected-error @+1 {{entry block argument #2 (grid iv) must have type `index`, got 'i32'}}
"dataflow.thread"() <{function_type = () -> (),
                     sym_name = "t_iv_wrong_type",
                     sym_visibility = "private"}> ({
^bb0(%c: none, %i: index, %bad: i32):
  dataflow.thread.yield
}) : () -> ()

// -----
// A thread body cannot launch another thread.
dataflow.thread private @nested_thread_leaf() ctrl (%ctrl: none) {
  dataflow.thread.yield
}
dataflow.thread private @nested_thread_parent() ctrl (%ctrl: none) {
  scf.execute_region {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    %token = dataflow.thread.launch @nested_thread_leaf() : () -> !dataflow.thread_token
    scf.yield
  }
  dataflow.thread.yield
}

// -----
// A thread body cannot wait on a caller-side completion token.
dataflow.thread private @thread_wait_in_body(%token: !dataflow.thread_token) ctrl (%ctrl: none) {
  scf.execute_region {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    dataflow.thread.wait %token : !dataflow.thread_token
    scf.yield
  }
  dataflow.thread.yield
}
