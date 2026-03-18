module {
  fabric.function_unit @fu_def() -> () [latency = 1, interval = 1] {
    fabric.yield
  }

  fabric.module @invalid_module_target() {
    fabric.instance @fu_def() : () -> ()
    fabric.yield
  }
}
