#include "TemporalPeDiagnostics.h"

#include "Common/DiagnosticVerbosity.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace loom::hardware::rtl::hierarchy {

void emitTemporalPeDiagnostics(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor, fabric::FabricEntityId pe,
    llvm::ArrayRef<EndpointPlan> endpoints,
    llvm::ArrayRef<TemporalPeOperandDiagnostics> operands,
    llvm::ArrayRef<TemporalPeFuInputDiagnostics> inputs,
    llvm::ArrayRef<TemporalPeResultDiagnostics> results,
    llvm::ArrayRef<TemporalPeFifoDiagnostics> fifos) {
  llvm::SmallVector<mlir::Value> substitutions;
  auto use = [&](mlir::Value value) {
    assert(value && "diagnostic observes an existing typed signal");
    substitutions.push_back(value);
    return "{{" + std::to_string(substitutions.size() - 1) + "}}";
  };
  auto data = [&](std::optional<mlir::Value> value) {
    return value ? use(*value) : std::string("1'b0");
  };
  auto output = [&](llvm::StringRef name) -> mlir::Value {
    for (auto [ordinal, port] :
         llvm::enumerate(accessor.getPortList().getOutputs()))
      if (port.getName() == name)
        return accessor.getOutputOperands()[ordinal];
    llvm_unreachable("typed TemporalPE endpoint output is absent");
  };
  const unsigned decision = static_cast<unsigned>(DiagnosticVerbosity::Decision);
  const unsigned detail = static_cast<unsigned>(DiagnosticVerbosity::Detail);
  std::string text;
  llvm::raw_string_ostream out(text);
  out << "`ifndef SYNTHESIS\ninteger loom_pe_verbose_level;\n"
      << "initial begin\n  if (!$value$plusargs(\""
      << diagnosticVerbosityEnvironment
      << "=%d\", loom_pe_verbose_level)) loom_pe_verbose_level = 0;\nend\n"
      << "always @(posedge " << use(accessor.getInput("clock"))
      << ") begin\n  if (loom_pe_verbose_level >= " << decision << " && !"
      << use(accessor.getInput("reset")) << ") begin\n";
  auto condition = [&](const std::string &commit, const std::string &observe) {
    out << "    if ((" << commit << ") || (loom_pe_verbose_level >= "
        << detail << " && (" << observe
        << "))) $display(\"[loom][rtl][temporal_pe] pe=" << pe;
  };
  for (const auto &operand : operands) {
    const auto active = use(operand.active), occupied = use(operand.occupied);
    const auto enqueue = use(operand.enqueue), selected = use(operand.selected);
    const auto ready = use(operand.ready);
    condition(enqueue + " || (" + selected + " && " + occupied + " && " +
                  ready + ")", active);
    out << " event=operand time=%0t context=" << operand.context
        << " fu=" << operand.fu << " input=" << operand.input
        << " active=%0b route=%0b target=%0d tag=%0h occupied=%0b "
           "enqueue_ready=%0b enqueue_grant=%0b enqueue=%0b selected=%0b "
           "fu_ready=%0b data=%0h\", $realtime, "
        << active << ", " << use(operand.route) << ", " << use(operand.target)
        << ", " << use(operand.tag) << ", " << occupied << ", "
        << use(operand.enqueueReady) << ", " << use(operand.enqueueGrant)
        << ", " << enqueue << ", " << selected << ", " << ready << ", "
        << data(operand.data) << ");\n";
  }
  for (const auto &input : inputs) {
    const auto valid = use(input.valid), ready = use(input.ready);
    const auto enabled = use(input.enabled);
    condition(enabled + " && " + valid + " && " + ready, enabled);
    out << " event=fu_input time=%0t fu=" << input.fu << " input=" << input.input
        << " context=%0d enabled=%0b valid=%0b ready=%0b data=%0h\", $realtime, "
        << use(input.context) << ", " << enabled << ", " << valid << ", "
        << ready << ", " << data(input.data) << ");\n";
  }
  for (const auto &result : results) {
    const auto valid = use(result.valid), ready = use(result.ready);
    const auto offer = use(result.offer);
    condition(valid + " && " + ready, offer + " || " + valid);
    out << " event=result time=%0t fu=" << result.fu << " output=" << result.output
        << " context=%0d requester=%0d offer=%0b valid=%0b ready=%0b presented=%0b "
           "active=%0b route=%0b discard=%0b target=%0d tag=%0h data=%0h\", "
           "$realtime, "
        << use(result.context) << ", " << use(result.requester) << ", "
        << offer << ", " << valid << ", " << ready << ", "
        << use(result.presented) << ", " << use(result.active) << ", "
        << use(result.route) << ", " << use(result.discard) << ", "
        << use(result.target) << ", " << use(result.tag) << ", "
        << data(result.data) << ");\n";
  }
  for (auto [ordinal, fifo] : llvm::enumerate(fifos)) {
    const auto valid = use(fifo.valid), ready = use(fifo.ready);
    condition(valid + " && " + ready, valid);
    out << " event=fifo time=%0t fifo=" << ordinal
        << " valid=%0b ready=%0b tag=%0h data=%0h\", $realtime, "
        << valid << ", " << ready << ", " << use(fifo.tag) << ", "
        << data(fifo.data) << ");\n";
  }
  for (const EndpointPlan &endpoint : endpoints) {
    const bool input = endpoint.direction == fabric::FabricPortDirection::Input;
    auto signal = [&](llvm::StringRef name) {
      return use(input ? accessor.getInput(name) : output(name));
    };
    const auto valid = signal(endpoint.valid.getName());
    const auto ready = use(input ? output(endpoint.ready.getName())
                                : accessor.getInput(endpoint.ready.getName()));
    condition(valid + " && " + ready, valid);
    out << " event=port time=%0t direction=" << (input ? "input" : "output")
        << " endpoint=" << endpoint.endpoint.ordinal
        << " local_port=" << endpoint.localOrdinal
        << " valid=%0b ready=%0b tag=%0h data=%0h\", $realtime, "
        << valid << ", " << ready << ", "
        << (endpoint.tag ? signal(endpoint.tag->getName()) : "1'b0") << ", "
        << (endpoint.data ? signal(endpoint.data->getName()) : "1'b0") << ");\n";
  }
  out << "  end\nend\n`endif\n";
  out.flush();
  circt::sv::VerbatimOp::create(builder, location, builder.getStringAttr(text),
                              substitutions, builder.getArrayAttr({}));
}

} // namespace loom::hardware::rtl::hierarchy
