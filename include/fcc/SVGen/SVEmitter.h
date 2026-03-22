#ifndef FCC_SVGEN_SVEMITTER_H
#define FCC_SVGEN_SVEMITTER_H

#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace fcc {
namespace svgen {

/// Port direction for SV module declarations.
enum class SVPortDir { Input, Output };

/// A single port in an SV module declaration.
struct SVPort {
  SVPortDir dir;
  std::string typeStr; // e.g. "logic [31:0]" or "logic"
  std::string name;
};

/// An instance connection: port name -> expression.
struct SVConnection {
  std::string portName;
  std::string expression;
};

/// Thin SystemVerilog emitter wrapping a raw_ostream. Centralizes module
/// headers/footers, wire/reg declarations, port formatting, instance
/// emission, and literal formatting. Handles MLIR-to-SV naming
/// sanitization and type mapping.
class SVEmitter {
public:
  explicit SVEmitter(llvm::raw_ostream &os) : os_(os), indentLevel_(0) {}

  /// Emit the standard auto-generated file header comment.
  void emitFileHeader(llvm::StringRef moduleName);

  /// Emit `module <name> #( ... parameters ... ) ( ... ports ... );`
  void emitModuleHeader(llvm::StringRef name,
                        const std::vector<std::string> &parameters,
                        const std::vector<SVPort> &ports);

  /// Emit `endmodule // <name>`
  void emitModuleFooter(llvm::StringRef name);

  /// Emit a wire declaration: `wire [W-1:0] name;`
  void emitWire(llvm::StringRef typeStr, llvm::StringRef name);

  /// Emit a reg declaration: `reg [W-1:0] name;`
  void emitReg(llvm::StringRef typeStr, llvm::StringRef name);

  /// Emit a localparam declaration.
  void emitLocalParam(llvm::StringRef typeStr, llvm::StringRef name,
                      llvm::StringRef value);

  /// Emit a module instantiation.
  void emitInstance(llvm::StringRef moduleName, llvm::StringRef instanceName,
                    const std::vector<std::string> &paramValues,
                    const std::vector<SVConnection> &connections);

  /// Emit a continuous assignment: `assign lhs = rhs;`
  void emitAssign(llvm::StringRef lhs, llvm::StringRef rhs);

  /// Emit a blank line.
  void emitBlankLine();

  /// Emit a comment line: `// <text>`
  void emitComment(llvm::StringRef text);

  /// Emit raw text at current indentation.
  void emitRaw(llvm::StringRef text);

  /// Emit raw text without indentation.
  void emitRawNoIndent(llvm::StringRef text);

  /// Increment/decrement indentation level.
  void indent() { ++indentLevel_; }
  void dedent() {
    if (indentLevel_ > 0)
      --indentLevel_;
  }

  // --- Naming and type utilities ---

  /// Sanitize an MLIR name to a legal SV identifier.
  /// Replaces dots, percent signs, colons with underscores.
  static std::string sanitizeName(llvm::StringRef mlirName);

  /// Convert an MLIR type to a SystemVerilog type string.
  /// fabric.bits<32> -> "logic [31:0]"
  /// i32 -> "logic [31:0]"
  /// f32 -> "logic [31:0]"
  /// fabric.tagged<bits<32>, i4> -> "logic [35:0]" (data + tag)
  static std::string typeToSV(mlir::Type type);

  /// Get the bit width of an MLIR type.
  static unsigned getTypeWidth(mlir::Type type);

  /// Get the data width portion (excluding tag) of an MLIR type.
  static unsigned getDataWidth(mlir::Type type);

  /// Get the tag width of a tagged type (0 for non-tagged).
  static unsigned getTagWidth(mlir::Type type);

  /// Format a bit range string: "[N-1:0]" or "" for single bit.
  static std::string bitRange(unsigned width);

  /// Get the underlying raw_ostream.
  llvm::raw_ostream &stream() { return os_; }

private:
  llvm::raw_ostream &os_;
  unsigned indentLevel_;

  void emitIndent();
};

} // namespace svgen
} // namespace fcc

#endif // FCC_SVGEN_SVEMITTER_H
