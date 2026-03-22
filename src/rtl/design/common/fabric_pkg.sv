// fabric_pkg.sv -- Shared package for fabric RTL infrastructure.
//
// Provides common constants, types, and utility functions used across
// all fabric design modules.

package fabric_pkg;

  // ---------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------

  // Maximum number of ports on any switch in the fabric.
  localparam int unsigned MAX_SWITCH_PORTS = 32;

  // Maximum tag width supported by tagged interfaces.
  localparam int unsigned MAX_TAG_WIDTH = 16;

  // Configuration word width (matches config.bin serialization).
  localparam int unsigned CONFIG_WORD_WIDTH = 32;

  // ---------------------------------------------------------------
  // Types
  // ---------------------------------------------------------------

  // Configuration word type used by config bus interfaces.
  typedef logic [CONFIG_WORD_WIDTH-1:0] config_word_t;

  // ---------------------------------------------------------------
  // Utility Functions
  // ---------------------------------------------------------------

  // Ceiling of log-base-2.  Returns the number of bits needed to
  // represent values in the range [0, n).
  //   clog2(0) = 0
  //   clog2(1) = 0   (one value needs zero select bits)
  //   clog2(2) = 1
  //   clog2(5) = 3
  //
  // Many synthesis tools provide $clog2, but having a local pure
  // function avoids tool-version portability issues and allows use
  // in constant expressions inside packages.
  function automatic int unsigned clog2(input int unsigned n);
    int unsigned result;
    int unsigned value;
    result = 0;
    if (n > 1) begin : clog2_compute
      value = n - 1;
      while (value > 0) begin : clog2_shift
        value = value >> 1;
        result = result + 1;
      end : clog2_shift
    end : clog2_compute
    return result;
  endfunction : clog2

  // Ceiling of log-base-2, but returns at least 1.
  // Useful for port-index widths where a zero-width bus is illegal.
  function automatic int unsigned clog2_min1(input int unsigned n);
    int unsigned v;
    v = clog2(n);
    return (v == 0) ? 1 : v;
  endfunction : clog2_min1

endpackage : fabric_pkg
