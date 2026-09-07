#include "MemoryDiagnostics.h"

#include "Common/DiagnosticVerbosity.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace loom::hardware::rtl::hierarchy {

void emitMemoryDiagnostics(mlir::OpBuilder &builder, mlir::Location location,
                           circt::hw::HWModulePortAccessor &accessor,
                           mlir::Value contextBase,
                           llvm::ArrayRef<EndpointPlan> endpoints,
                           llvm::ArrayRef<MemoryRowDiagnostics> rows) {
  llvm::SmallVector<mlir::Value> substitutions;
  auto use = [&](mlir::Value value) {
    assert(value && "diagnostic observes an existing typed signal");
    substitutions.push_back(value);
    return "{{" + std::to_string(substitutions.size() - 1) + "}}";
  };
  const std::string clock = use(accessor.getInput("clock"));
  const std::string reset = use(accessor.getInput("reset"));
  const std::string base = use(contextBase);
  const unsigned decision = static_cast<unsigned>(DiagnosticVerbosity::Decision);
  const unsigned detail = static_cast<unsigned>(DiagnosticVerbosity::Detail);
  auto output = [&](llvm::StringRef name) -> mlir::Value {
    // The accessor indexes output slots by their order, not PortInfo::argNum.
    for (auto [ordinal, port] :
         llvm::enumerate(accessor.getPortList().getOutputs()))
      if (port.getName() == name)
        return accessor.getOutputOperands()[ordinal];
    llvm_unreachable("typed memory endpoint output is absent");
  };
  std::string text;
  llvm::raw_string_ostream out(text);
  out << "`ifndef SYNTHESIS\n"
      << "integer loom_memory_verbose_level;\n"
      << "initial begin\n"
      << "  if (!$value$plusargs(\"" << diagnosticVerbosityEnvironment
      << "=%d\", loom_memory_verbose_level)) loom_memory_verbose_level = 0;\n"
      << "end\n"
      << "always @(posedge " << clock << ") begin\n"
      << "  if (!" << reset << ") begin\n";

  for (auto [ordinal, row] : llvm::enumerate(rows)) {
    const std::string context = use(row.context);
    const std::string active = use(row.active);
    const std::string write = use(row.write);
    const std::string request = use(row.requestValid);
    const std::string issued = use(row.issued);
    const std::string response = use(row.response);
    const std::string released = use(row.released);
    const std::string occupied = use(row.occupied);
    const std::string completed = use(row.completed);
    const std::string selected = use(row.selected);
    std::string busy = occupied + " || " + completed + " || " + request;
    std::vector<std::string> requiredBits, validBits, occupiedBits;
    for (const MemoryOperandDiagnostics &operand : row.operands) {
      const std::string present = use(operand.present);
      const std::string valid = use(operand.valid);
      const std::string queued = use(operand.occupied);
      const std::string enqueue = use(operand.enqueue);
      const std::string dequeue = use(operand.dequeue);
      const std::string internal = use(operand.internal);
      const std::string source = use(operand.endpoint);
      const std::string expectedTag = use(operand.tag);
      std::string inputValid = "1'b0", inputReady = "1'b0", inputTag = "'0";
      std::string inputData = "'0";
      for (const EndpointPlan &endpoint : endpoints) {
        if (endpoint.direction != fabric::FabricPortDirection::Input)
          continue;
        const std::string condition =
            source + " == " + std::to_string(endpoint.endpoint.ordinal);
        inputValid = "(" + condition + " ? " +
                     use(accessor.getInput(endpoint.valid.getName())) +
                     " : " + inputValid + ")";
        inputReady = "(" + condition + " ? " +
                     use(output(endpoint.ready.getName())) + " : " +
                     inputReady + ")";
        if (endpoint.tag)
          inputTag = "(" + condition + " ? " +
                     use(accessor.getInput(endpoint.tag->getName())) +
                     " : " + inputTag + ")";
        if (endpoint.data)
          inputData = "(" + condition + " ? " +
                      use(accessor.getInput(endpoint.data->getName())) +
                      " : " + inputData + ")";
      }
      requiredBits.push_back(present);
      validBits.push_back(valid);
      occupiedBits.push_back(queued);
      busy += " || (" + present + " && " + queued + ")";
      out << "    if ((loom_memory_verbose_level >= " << decision << " && ("
          << enqueue << " || " << dequeue
          << ")) || (loom_memory_verbose_level >= " << detail << " && "
          << active << " && " << write << " && " << present << " && !"
          << internal << ")) $display("
             "\"[loom][rtl][memory_owner] event=operand time=%0t "
             "context_base=%0h row="
          << ordinal << " context=%0h role="
          << static_cast<unsigned>(operand.role)
          << " present=%0b source_valid=%0b source_internal=%0b "
             "source_endpoint=%0d source_tag=%0h occupied=%0b "
             "enqueue=%0b dequeue=%0b data=%0h incoming=%0h "
             "input_valid=%0b input_ready=%0b input_tag=%0h input_data=%0h\", "
             "$realtime, "
          << base << ", " << context << ", " << present << ", " << valid
          << ", " << internal << ", " << source
          << ", " << expectedTag << ", " << queued << ", " << enqueue
          << ", " << dequeue << ", " << use(operand.data) << ", "
          << use(operand.incoming) << ", " << inputValid << ", "
          << inputReady << ", " << inputTag << ", " << inputData << ");\n";
    }
    auto bits = [](const std::vector<std::string> &values) {
      std::string result = "{ ";
      for (const std::string &value : llvm::reverse(values)) {
        if (result.size() != 2)
          result += ", ";
        result += value;
      }
      return values.empty() ? std::string("1'b0") : result + " }";
    };
    out << "    if ((loom_memory_verbose_level >= " << decision << " && ("
        << issued << " || " << response << " || " << released
        << ")) || (loom_memory_verbose_level >= " << detail << " && "
        << active << " && (" << busy << "))) $display("
           "\"[loom][rtl][memory_owner] event=row time=%0t "
           "context_base=%0h row="
        << ordinal << " context=%0h active=%0b write=%0b required=%0b source_valid=%0b "
           "queued=%0b request_valid=%0b issue=%0b response=%0b release=%0b "
           "result_occupied=%0b result_completed=%0b result_selected=%0b "
           "address=%0h data=%0h result=%0h result_next=%0h\", $realtime, "
        << base << ", " << context << ", " << active << ", " << write << ", "
        << bits(requiredBits) << ", " << bits(validBits) << ", "
        << bits(occupiedBits) << ", " << request << ", " << issued << ", "
        << response << ", " << released << ", " << occupied << ", "
        << completed << ", " << selected << ", " << use(row.address) << ", "
        << use(row.data) << ", " << use(row.result) << ", "
        << use(row.resultNext) << ");\n";
  }

  for (const EndpointPlan &endpoint : endpoints) {
    const bool input =
        endpoint.direction == fabric::FabricPortDirection::Input;
    auto signal = [&](llvm::StringRef name) {
      return use(input ? accessor.getInput(name) : output(name));
    };
    const std::string valid = signal(endpoint.valid.getName());
    const std::string ready = use(input ? output(endpoint.ready.getName())
                                       : accessor.getInput(endpoint.ready.getName()));
    out << "    if (" << valid << " && (loom_memory_verbose_level >= "
        << detail << " || (loom_memory_verbose_level >= " << decision
        << " && " << ready << "))) $display("
           "\"[loom][rtl][memory_owner] event=port time=%0t "
           "context_base=%0h direction="
        << (input ? "input" : "output") << " endpoint="
        << endpoint.endpoint.ordinal << " local_port=" << endpoint.localOrdinal
        << " data_bits=" << endpoint.dataPath.payloadWidthBits
        << " tag_bits=" << endpoint.dataPath.tagWidthBits
        << " valid=%0b ready=%0b data=%0h tag=%0h\", $realtime, "
        << base << ", " << valid << ", " << ready << ", "
        << (endpoint.data ? signal(endpoint.data->getName()) : "1'b0") << ", "
        << (endpoint.tag ? signal(endpoint.tag->getName()) : "1'b0") << ");\n";
  }
  out << "  end\nend\n`endif\n";
  out.flush();
  circt::sv::VerbatimOp::create(builder, location, builder.getStringAttr(text),
                               substitutions, builder.getArrayAttr({}));
}

} // namespace loom::hardware::rtl::hierarchy
