#include "EDA/Adapters/OpenSource/Yosys.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace loom::eda::open_source;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectInvalid(llvm::StringRef test, llvm::Error error) {
  if (!error)
    fail(test, "accepted an invalid Yosys structure");
  llvm::consumeError(std::move(error));
}

template <typename T>
void expectInvalid(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invalid Yosys structure");
  llvm::consumeError(value.takeError());
}

const std::string kMappedJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": {
        "a": {"direction": "input", "bits": [2]},
        "b": {"direction": "input", "bits": [3]},
        "y": {"direction": "output", "bits": [4]}
      },
      "cells": {
        "_0_": {
          "type": "AND2X1",
          "port_directions": {"A": "input", "B": "input", "Y": "output"},
          "connections": {"A": [2], "B": [3], "Y": [4]}
        }
      },
      "netnames": {
        "a": {"bits": [2]},
        "b": {"bits": [3]},
        "y": {"bits": [4]}
      }
    }
  }
})json";

const std::string kConstantJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": {
        "a": {"direction": "input", "bits": [2]},
        "y": {"direction": "output", "bits": ["0"]}
      },
      "cells": {},
      "netnames": {
        "a": {"bits": [2]},
        "y": {"bits": ["0"]}
      }
    }
  }
})json";

const std::string kMultidriverJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": {
        "a": {"direction": "input", "bits": [2]},
        "b": {"direction": "input", "bits": [3]},
        "y": {"direction": "output", "bits": [4]}
      },
      "cells": {
        "_0_": {
          "type": "AND2X1",
          "port_directions": {"A": "input", "B": "input", "Y": "output"},
          "connections": {"A": [2], "B": [3], "Y": [4]}
        },
        "_1_": {
          "type": "AND2X1",
          "port_directions": {"A": "input", "B": "input", "Y": "output"},
          "connections": {"A": [2], "B": [3], "Y": [4]}
        }
      },
      "netnames": {
        "a": {"bits": [2]},
        "b": {"bits": [3]},
        "y": {"bits": [4]}
      }
    }
  }
})json";

const std::string kSelfInstantiationJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": {
        "a": {"direction": "input", "bits": [2]},
        "b": {"direction": "input", "bits": [3]},
        "y": {"direction": "output", "bits": [4]}
      },
      "cells": {
        "_0_": {
          "type": "AND2X1",
          "port_directions": {"A": "input", "B": "input", "Y": "output"},
          "connections": {"A": [2], "B": [3], "Y": [4]}
        },
        "_1_": {
          "type": "top",
          "port_directions": {"a": "input", "b": "input", "y": "output"},
          "connections": {"a": [2], "b": [3], "y": [5]}
        }
      },
      "netnames": {
        "a": {"bits": [2]},
        "b": {"bits": [3]},
        "y": {"bits": [4]},
        "extra": {"bits": [5]}
      }
    }
  }
})json";

const std::string kBoxWithCellsJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {
        "inner": {"type": "AND2X1", "port_directions": {}, "connections": {}}
      },
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": {
        "a": {"direction": "input", "bits": [2]},
        "b": {"direction": "input", "bits": [3]},
        "y": {"direction": "output", "bits": [4]}
      },
      "cells": {
        "_0_": {
          "type": "AND2X1",
          "port_directions": {"A": "input", "B": "input", "Y": "output"},
          "connections": {"A": [2], "B": [3], "Y": [4]}
        }
      },
      "netnames": {
        "a": {"bits": [2]},
        "b": {"bits": [3]},
        "y": {"bits": [4]}
      }
    }
  }
})json";

const std::string kBoxWithMemoryJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "memories": {"m": {}},
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": {
        "a": {"direction": "input", "bits": [2]},
        "y": {"direction": "output", "bits": ["0"]}
      },
      "cells": {},
      "netnames": {
        "a": {"bits": [2]},
        "y": {"bits": ["0"]}
      }
    }
  }
})json";

const std::string kPortsAsArrayJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": [],
      "cells": {},
      "netnames": {}
    }
  }
})json";

const std::string kInoutJson = R"json({
  "creator": "fixture",
  "modules": {
    "AND2X1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "B": {"direction": "input", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "netnames": {}
    },
    "BIDIRX1": {
      "attributes": {"blackbox": "00000000000000000000000000000001"},
      "ports": {
        "A": {"direction": "input", "bits": [0]},
        "IO": {"direction": "inout", "bits": [0]},
        "Y": {"direction": "output", "bits": [0]}
      },
      "cells": {},
      "netnames": {}
    },
    "top": {
      "attributes": {},
      "ports": {
        "a": {"direction": "input", "bits": [2]},
        "y": {"direction": "output", "bits": [4]},
        "io_free": {"direction": "inout", "bits": [5]},
        "io_used": {"direction": "inout", "bits": [6]}
      },
      "cells": {
        "_0_": {
          "type": "AND2X1",
          "port_directions": {"A": "input", "B": "input", "Y": "output"},
          "connections": {"A": [2], "B": [2], "Y": [4]}
        },
        "_1_": {
          "type": "BIDIRX1",
          "port_directions": {"A": "input", "IO": "inout", "Y": "output"},
          "connections": {"A": [2], "IO": [6], "Y": [7]}
        }
      },
      "netnames": {
        "a": {"bits": [2]},
        "y": {"bits": [4]},
        "io_free": {"bits": [5]},
        "io_used": {"bits": [6]},
        "extra": {"bits": [7]}
      }
    }
  }
})json";

std::string replaceAll(std::string text, llvm::StringRef from,
                       llvm::StringRef to) {
  std::size_t position = 0;
  while ((position = text.find(from.str(), position)) != std::string::npos) {
    text.replace(position, from.size(), to.str());
    position += to.size();
  }
  return text;
}

void driverBytesAreDeterministicAndExact() {
  const std::string driver = take(__func__, renderYosysSynthesisDriver("top"));
  const std::string expected = R"ys(read_verilog -sv inputs/design.sv
hierarchy -check -top top
proc
opt
check -assert -nolatches
write_json outputs/rtl-structure.json
synth -flatten -top top
dfflibmap -liberty inputs/library.lib
abc -liberty inputs/library.lib
read_liberty -lib inputs/library.lib
clean
check -assert -nolatches
write_verilog -noattr -nodec -simple-lhs outputs/netlist.v
design -reset
read_liberty -lib inputs/library.lib
read_verilog outputs/netlist.v
hierarchy -check -top top
proc
opt
check -assert -nolatches
write_json outputs/netlist-structure.json
)ys";
  require(__func__, driver == expected, "driver bytes changed");
  require(__func__,
          take(__func__, renderYosysSynthesisDriver("top")) == driver,
          "driver render is not deterministic");
  require(__func__,
          take(__func__, renderYosysSynthesisDriver("my_core_2")) !=
              driver,
          "driver ignores the exact top");
}

void rtlSourcesRemainIndependentCompilationUnits() {
  const std::vector<std::string> sources{
      "inputs/rtl/rtl/package.sv", "inputs/rtl/rtl/top module.sv"};
  const std::string driver = take(
      __func__, renderYosysSynthesisDriver(
                    "top", sources, "inputs/external/typical_cells.lib"));
  require(__func__,
          llvm::StringRef(driver).starts_with(
              "read_verilog -sv inputs/rtl/rtl/package.sv\n"
              "read_verilog -sv \"inputs/rtl/rtl/top module.sv\"\n"),
          "RTL sources were not emitted as independent compilation units");
  require(__func__,
          llvm::StringRef(driver).contains(
              "abc -liberty inputs/external/typical_cells.lib\n"),
          "Liberty path was not retained as one bare ABC-compatible token");
}

void unrepresentableDriverTokensAreRejected() {
  const std::vector<std::string> noSources;
  expectInvalid(__func__,
                renderYosysSynthesisDriver("top", noSources, "library.lib"));
  for (const std::string &source : {"quote\".sv", "back\\slash.sv"})
    expectInvalid(__func__, renderYosysSynthesisDriver(
                                "top", {source}, "library.lib"));
  for (llvm::StringRef liberty : {"quote\".lib", "back\\slash.lib",
                                  "has space.lib", "apostrophe's.lib"})
    expectInvalid(__func__, renderYosysSynthesisDriver(
                                "top", {"design.sv"}, liberty));
}

void unsafeTopsAreRejected() {
  for (llvm::StringRef top : {"", "9bad", "has space", "quo\"te", "semi;colon",
                              "new\nline", "carriage\r", "back\\slash",
                              "\\escaped", "unicode-\xc3\xa9"})
    expectInvalid(__func__, renderYosysSynthesisDriver(top));
  // `$` is legal inside the HDL identifier grammar and needs no quoting in a
  // Yosys command file.
  require(__func__,
          take(__func__, renderYosysSynthesisDriver("dollar$1"))
                  .find("-top dollar$1\n") != std::string::npos,
          "identifier grammar rejected a legal continuation dollar");
}

void malformedJsonIsRejected() {
  expectInvalid(__func__, parseYosysStructureFacts("not json"));
  expectInvalid(__func__, parseYosysStructureFacts("[1]"));
  expectInvalid(__func__, parseYosysStructureFacts("{}"));
  expectInvalid(__func__, parseYosysStructureFacts("{\"modules\": []}"));
}

void positiveStructuresAreAccepted() {
  const YosysStructureFacts mapped =
      take(__func__, parseYosysStructureFacts(kMappedJson));
  require(__func__, mapped.modules.size() == 2, "module inventory changed");
  require(__func__,
          mapped.modules.at("top").ports.at("y").direction ==
              YosysPortGeometry::Direction::Output,
          "top port direction changed");
  require(__func__,
          !validateYosysSynthesizedStructure(mapped, "top"),
          "mapped structure was rejected");

  const YosysStructureFacts constant =
      take(__func__, parseYosysStructureFacts(kConstantJson));
  require(__func__, constant.modules.at("top").cells.empty(),
          "constant-only design acquired cells");
  require(__func__,
          !validateYosysSynthesizedStructure(constant, "top"),
          "zero-cell constant-only design was rejected");

  // A bare externally driven top inout is not a required output, and a legal
  // inout connected through a matching declared inout cell is not a
  // multi-driver violation: the simple counter never proves tri-state
  // ownership.
  const YosysStructureFacts inout =
      take(__func__, parseYosysStructureFacts(kInoutJson));
  require(__func__, !validateYosysSynthesizedStructure(inout, "top"),
          "legal inout topology was rejected");
}

void adverseStructuresAreRejected() {
  const auto reject = [&](llvm::StringRef name, std::string json) {
    auto facts = parseYosysStructureFacts(json);
    if (!facts) {
      llvm::consumeError(facts.takeError());
      return;
    }
    expectInvalid(name, validateYosysSynthesizedStructure(*facts, "top"));
  };
  reject("blackbox-top", replaceAll(kMappedJson, "\"attributes\": {}",
                                    "\"attributes\": {\"blackbox\": \"1\"}"));
  reject("functional-module", replaceAll(kMappedJson, "\"blackbox\": "
                                         "\"00000000000000000000000000000001\"",
                                         "\"keep\": \"1\""));
  reject("residual-process",
         replaceAll(kMappedJson, "\"attributes\": {}",
                    "\"attributes\": {}, \"processes\": {\"p\": {}}"));
  reject("residual-memory",
         replaceAll(kMappedJson, "\"netnames\": {\n        \"a\"",
                    "\"memories\": {\"m\": {}}, \"netnames\": {\n        \"a\""));
  reject("generic-cell",
         replaceAll(kMappedJson, "\"type\": \"AND2X1\"", "\"type\": \"$and\""));
  reject("undeclared-cell",
         replaceAll(kMappedJson, "\"type\": \"AND2X1\"",
                    "\"type\": \"MYSTERY_X1\""));
  reject("undeclared-connection",
         replaceAll(kMappedJson, "\"A\": [2],", "\"Q\": [2],"));
  reject("undriven-output",
         replaceAll(kMappedJson, "\"Y\": [4]", "\"Y\": [9]"));
  {
    // The fixture must prove two top-level cells output-drive the required
    // net before the validator is consulted.
    YosysStructureFacts multidriver =
        take(__func__, parseYosysStructureFacts(kMultidriverJson));
    const auto &cells = multidriver.modules.at("top").cells;
    require(__func__, cells.size() == 2, "multidriver fixture lost a cell");
    for (const auto &[name, cell] : cells) {
      const auto &connection = cell.connections.at("Y");
      require(__func__,
              connection.size() == 1 &&
                  std::get<std::uint64_t>(connection.front().value) == 4,
              "multidriver fixture does not drive required net 4");
    }
    llvm::Error error = validateYosysSynthesizedStructure(multidriver, "top");
    require(__func__, !!error, "multidriver structure was accepted");
    const std::string message = llvm::toString(std::move(error));
    require(__func__, llvm::StringRef(message).contains("multiple drivers"),
            "wrong failure class: " + message);
  }
  reject("x-constant-output",
         replaceAll(kConstantJson, "\"bits\": [\"0\"]", "\"bits\": [\"x\"]"));

  expectInvalid(__func__,
                parseYosysStructureFacts(replaceAll(
                    kMappedJson, "\"direction\": \"input\"", "\"direction\": "
                    "\"sideways\"")));
  expectInvalid(__func__,
                parseYosysStructureFacts(
                    replaceAll(kMappedJson, "\"bits\": [2]", "\"bits\": []")));
  expectInvalid(__func__,
                parseYosysStructureFacts(
                    replaceAll(kMappedJson, "\"bits\": [2]", "\"bits\": [-1]")));
}

void structuralConsistencyIsEnforced() {
  const auto reject = [&](llvm::StringRef name, std::string json) {
    auto facts = parseYosysStructureFacts(json);
    if (!facts) {
      llvm::consumeError(facts.takeError());
      return;
    }
    expectInvalid(name, validateYosysSynthesizedStructure(*facts, "top"));
  };

  // A present-but-wrong-typed container must not read as absent.
  reject("ports-as-array", kPortsAsArrayJson);
  reject("attributes-as-string", replaceAll(kMappedJson, "\"attributes\": {}",
                                            "\"attributes\": \"none\""));
  reject("processes-as-array",
         replaceAll(kMappedJson, "\"attributes\": {}",
                    "\"attributes\": {}, \"processes\": []"));
  reject("netnames-as-string",
         replaceAll(kMappedJson, "\"netnames\": {\n        \"a\"",
                    "\"netnames\": \"x\", \"junk\": {\n        \"a\""));

  // Cell structural consistency.
  reject("empty-connection",
         replaceAll(kMappedJson, "\"A\": [2],", "\"A\": [],"));
  reject("direction-mismatch",
         replaceAll(kMappedJson, "\"port_directions\": {\"A\": \"input\"",
                    "\"port_directions\": {\"A\": \"output\""));
  reject("width-mismatch",
         replaceAll(kMappedJson, "\"Y\": [4]}", "\"Y\": [4, 5]}"));
  reject("self-instantiation", kSelfInstantiationJson);
  reject("box-with-process",
         replaceAll(kMappedJson,
                    "\"blackbox\": "
                    "\"00000000000000000000000000000001\"},\n      \"ports\"",
                    "\"blackbox\": "
                    "\"00000000000000000000000000000001\"},\n      "
                    "\"processes\": {\"p\": {}},\n      \"ports\""));
  reject("box-with-cells", kBoxWithCellsJson);
  reject("box-with-memory", kBoxWithMemoryJson);

  // Attribute values admit only Yosys's scalar encodings; anything else
  // fails closed instead of reading as false.
  expectInvalid(__func__,
                parseYosysStructureFacts(replaceAll(
                    kMappedJson, "\"attributes\": {}",
                    "\"attributes\": {\"blackbox\": {\"nested\": 1}}")));
  expectInvalid(__func__,
                parseYosysStructureFacts(replaceAll(
                    kMappedJson,
                    "\"blackbox\": \"00000000000000000000000000000001\"",
                    "\"blackbox\": [1]")));

  // A port object without a direction must fail typed, never crash.
  reject("missing-direction",
         replaceAll(kMappedJson, "\"direction\": \"input\", ", ""));
}

void portGeometryComparisonUsesCanonicalFacts() {
  const YosysStructureFacts pre =
      take(__func__, parseYosysStructureFacts(kMappedJson));
  require(__func__, !compareYosysTopPortGeometry(pre, pre, "top"),
          "identical geometry was rejected");
  const YosysStructureFacts rangeMetadata = take(
      __func__, parseYosysStructureFacts(replaceAll(
                    kMappedJson,
                    "\"y\": {\"direction\": \"output\", \"bits\": [4]}",
                    "\"y\": {\"direction\": \"output\", \"bits\": [4], "
                    "\"offset\": 7, \"upto\": 1, \"signed\": 1}")));
  require(__func__,
          !compareYosysTopPortGeometry(pre, rangeMetadata, "top"),
          "noncanonical Yosys range metadata changed descriptor port facts");
  expectInvalid(__func__,
                compareYosysTopPortGeometry(
                    pre,
                    take(__func__,
                         parseYosysStructureFacts(replaceAll(
                             kMappedJson, "\"bits\": [4]", "\"bits\": [4, 5]"))),
                    "top"));
  expectInvalid(__func__,
                compareYosysTopPortGeometry(
                    pre,
                    take(__func__,
                         parseYosysStructureFacts(
                             replaceAll(kMappedJson, "\"direction\": \"output\"",
                                        "\"direction\": \"inout\""))),
                    "top"));
  expectInvalid(__func__,
                compareYosysTopPortGeometry(
                    pre,
                    take(__func__,
                         parseYosysStructureFacts(replaceAll(
                             kMappedJson, "\"y\": {\"direction\"",
                             "\"z\": {\"direction\""))),
                    "top"));
}

constexpr llvm::StringLiteral syntheticLiberty = R"liberty(
library(probe) {
  delay_model : table_lookup;
  time_unit : "1ns";
  voltage_unit : "1V";
  current_unit : "1mA";
  pulling_resistance_unit : "1kohm";
  leakage_power_unit : "1nW";
  capacitive_load_unit(1,pf);
  default_cell_leakage_power : 0.0;
  default_fanout_load : 1.0;
  default_input_pin_cap : 0.01;
  default_inout_pin_cap : 0.01;
  default_output_pin_cap : 0.0;
  default_max_transition : 1.0;
  nom_process : 1.0;
  nom_temperature : 25.0;
  nom_voltage : 1.0;
  lu_table_template(fixed) {
    variable_1 : input_net_transition;
    variable_2 : total_output_net_capacitance;
    index_1("0.1");
    index_2("0.1");
  }
  cell(INVX1) {
    area : 1.0;
    pin(A) { direction : input; capacitance : 0.01; }
    pin(Y) { direction : output; function : "!A"; max_capacitance : 1.0;
      timing() { related_pin : "A"; timing_sense : negative_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } } }
  }
  cell(BUFX1) {
    area : 1.0;
    pin(A) { direction : input; capacitance : 0.01; }
    pin(Y) { direction : output; function : "A"; max_capacitance : 1.0;
      timing() { related_pin : "A"; timing_sense : positive_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } } }
  }
  cell(AND2X1) {
    area : 1.0;
    pin(A) { direction : input; capacitance : 0.01; }
    pin(B) { direction : input; capacitance : 0.01; }
    pin(Y) { direction : output; function : "A & B"; max_capacitance : 1.0;
      timing() { related_pin : "A"; timing_sense : positive_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } }
      timing() { related_pin : "B"; timing_sense : positive_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } } }
  }
  cell(NAND2X1) {
    area : 1.0;
    pin(A) { direction : input; capacitance : 0.01; }
    pin(B) { direction : input; capacitance : 0.01; }
    pin(Y) { direction : output; function : "!(A & B)"; max_capacitance : 1.0;
      timing() { related_pin : "A"; timing_sense : negative_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } }
      timing() { related_pin : "B"; timing_sense : negative_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } } }
  }
  cell(NOR2X1) {
    area : 1.0;
    pin(A) { direction : input; capacitance : 0.01; }
    pin(B) { direction : input; capacitance : 0.01; }
    pin(Y) { direction : output; function : "!(A | B)"; max_capacitance : 1.0;
      timing() { related_pin : "A"; timing_sense : negative_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } }
      timing() { related_pin : "B"; timing_sense : negative_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } } }
  }
  cell(OR2X1) {
    area : 1.0;
    pin(A) { direction : input; capacitance : 0.01; }
    pin(B) { direction : input; capacitance : 0.01; }
    pin(Y) { direction : output; function : "A | B"; max_capacitance : 1.0;
      timing() { related_pin : "A"; timing_sense : positive_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } }
      timing() { related_pin : "B"; timing_sense : positive_unate;
        cell_rise(fixed) { values("0.1"); } cell_fall(fixed) { values("0.1"); }
        rise_transition(fixed) { values("0.1"); }
        fall_transition(fixed) { values("0.1"); } } }
  }
}
)liberty";

const std::string kMappedDesign = "module top(input a, input b, output y);\n"
                                  "  assign y = a & b;\n"
                                  "endmodule\n";
const std::string kConstantDesign = "module top(input a, output y);\n"
                                    "  assign y = 1'b0;\n"
                                    "endmodule\n";
const std::string kDollarDesign = "module dollar$1(input a, output y);\n"
                                  "  assign y = a;\n"
                                  "endmodule\n";
const std::string kClosureHelper =
    "module helper(input a, input b, output y);\n"
    "  assign y = a & b;\n"
    "endmodule\n";
const std::string kClosureTop =
    "module top(input a, input b, output y);\n"
    "  helper helper_instance(.a(a), .b(b), .y(y));\n"
    "endmodule\n";

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream)
    fail("emit", "could not write " + path.string());
}

int emit(const std::string &root, const std::string &kind) {
  const std::filesystem::path base(root);
  std::filesystem::create_directories(base / "inputs");
  std::filesystem::create_directories(base / "outputs");
  std::string top = "top";
  std::optional<std::string> driver;
  if (kind == "mapped")
    writeFile(base / "inputs" / "design.sv", kMappedDesign);
  else if (kind == "constant")
    writeFile(base / "inputs" / "design.sv", kConstantDesign);
  else if (kind == "dollar") {
    writeFile(base / "inputs" / "design.sv", kDollarDesign);
    top = "dollar$1";
  } else if (kind == "closure") {
    std::filesystem::create_directories(base / "inputs" / "rtl");
    std::filesystem::create_directories(base / "inputs" / "external");
    writeFile(base / "inputs" / "rtl" / "helper.sv", kClosureHelper);
    writeFile(base / "inputs" / "rtl" / "top design.sv", kClosureTop);
    writeFile(base / "inputs" / "external" / "typical_cells.lib",
              syntheticLiberty);
    driver = take("emit", renderYosysSynthesisDriver(
                              top,
                              {"inputs/rtl/helper.sv",
                               "inputs/rtl/top design.sv"},
                              "inputs/external/typical_cells.lib"));
  } else
    fail("emit", "unknown design kind");
  if (!driver) {
    writeFile(base / "inputs" / "library.lib", syntheticLiberty);
    driver = take("emit", renderYosysSynthesisDriver(top));
  }
  writeFile(base / "top.txt", top);
  writeFile(base / "synthesize.ys", *driver);
  return EXIT_SUCCESS;
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream)
    fail("verify", "could not read " + path.string());
  return contents.str();
}

int verify(const std::string &root) {
  const std::filesystem::path base(root);
  const std::string top = readFile(base / "top.txt");
  const YosysStructureFacts pre = take(
      "verify", parseYosysStructureFacts(
                    readFile(base / "outputs" / "rtl-structure.json")));
  const YosysStructureFacts post = take(
      "verify", parseYosysStructureFacts(
                    readFile(base / "outputs" / "netlist-structure.json")));
  if (llvm::Error error = validateYosysSynthesizedStructure(post, top))
    fail("verify", llvm::toString(std::move(error)));
  if (llvm::Error error = compareYosysTopPortGeometry(pre, post, top))
    fail("verify", llvm::toString(std::move(error)));
  return EXIT_SUCCESS;
}

int compare(const std::string &lhs, const std::string &rhs) {
  for (const char *name : {"outputs/netlist.v", "outputs/netlist-structure.json",
                           "outputs/rtl-structure.json"})
    if (readFile(std::filesystem::path(lhs) / name) !=
        readFile(std::filesystem::path(rhs) / name))
      fail("compare", std::string("fresh-root output diverged: ") + name);
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 3 && llvm::StringRef(argv[1]) == "--verify")
    return verify(argv[2]);
  if (argc == 4 && llvm::StringRef(argv[1]) == "--compare")
    return compare(argv[2], argv[3]);
  if (argc == 3)
    return emit(argv[1], argv[2]);
  if (argc != 1)
    fail("main", "unexpected arguments");
  driverBytesAreDeterministicAndExact();
  rtlSourcesRemainIndependentCompilationUnits();
  unrepresentableDriverTokensAreRejected();
  unsafeTopsAreRejected();
  malformedJsonIsRejected();
  positiveStructuresAreAccepted();
  adverseStructuresAreRejected();
  structuralConsistencyIsEnforced();
  portGeometryComparisonUsesCanonicalFacts();
  return EXIT_SUCCESS;
}
