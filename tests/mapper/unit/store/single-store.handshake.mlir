module {
  handshake.func @single_store(%addr: index, %data: i32, %ctrl: none, ...) -> (index, i32)
      attributes {argNames = ["addr", "data", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["addr_out", "data_out"]} {
    %data_out, %addr_out = handshake.store [%addr] %data, %ctrl : index, i32
    handshake.return %addr_out, %data_out : index, i32
  }
}
