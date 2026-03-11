module {
  handshake.func @single_load(%addr: index, %data: i32, %ctrl: none, ...) -> (i32, index)
      attributes {argNames = ["addr", "data", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["data_out", "addr_out"]} {
    %data_out, %addr_out = handshake.load [%addr] %data, %ctrl : index, i32
    handshake.return %data_out, %addr_out : i32, index
  }
}
