module {
  fabric.function_unit @fu_def(%ctrl: none) -> (none) [latency = 1, interval = 1] {
    %done = handshake.join %ctrl : none
    fabric.yield %done : none
  }

  fabric.module @invalid_module_target(%ctrl: none) -> (none) {
    %done = fabric.instance @fu_def(%ctrl) : (none) -> (none)
    fabric.yield %done#0 : none
  }
}
