#include "Hardware/RTL/Providers/ScalarMathHyperbolic.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

enum class HyperbolicOperation : unsigned { Sinh, Cosh, Tanh };
using Format = detail::PortableFloatFormat;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  unsigned formatCode = 0;
};

constexpr std::array<std::int64_t, 64> kExpTable = {
    68719476736,  69467782776,  70224237334,  70988929142,  71761947898,
    72543384276,  73333329938,  74131877545,  74939120765,  75755154288,
    76580073833,  77413976163,  78256959093,  79109121506,  79970563360,
    80841385700,  81721690674,  82611581540,  83511162683,  84420539622,
    85339819026,  86269108727,  87208517730,  88158156225,  89118135606,
    90088568477,  91069568669,  92061251253,  93063732552,  94077130157,
    95101562938,  96137151061,  97184015999,  98242280549,  99312068844,
    100393506370, 101486719979, 102591837903, 103708989773, 104838306629,
    105979920939, 107133966613, 108300579022, 109479895006, 110672052900,
    111877192542, 113095455294, 114326984058, 115571923291, 116830419023,
    118102618875, 119388672076, 120688729477, 122002943576, 123331468528,
    124674460168, 126032076028, 127404475355, 128791819132, 130194270091,
    131611992741, 133045153377, 134493920110, 135958462879};

constexpr std::array<std::int64_t, 64> kNegativeExpTable = {
    68719476736, 67979231439, 67246960055, 66522576689, 65805996370,
    65097135046, 64395909566, 63702237678, 63016038014, 62337230084,
    61665734264, 61001471788, 60344364739, 59694336038, 59051309438,
    58415209512, 57785961645, 57163492029, 56547727647, 55938596271,
    55336026450, 54739947503, 54150289511, 53566983307, 52989960469,
    52419153314, 51854494886, 51295918952, 50743359990, 50196753185,
    49656034422, 49121140275, 48592008000, 48068575531, 47550781469,
    47038565079, 46531866276, 46030625627, 45534784335, 45044284239,
    44559067803, 44079078113, 43604258865, 43134554364, 42669909513,
    42210269811, 41755581341, 41305790770, 40860845337, 40420692850,
    39985281680, 39554560753, 39128479547, 38706988081, 38290036916,
    37877577144, 37469560383, 37065938773, 36666664969, 36271692138,
    35880973949, 35494464571, 35112118667, 34733891388};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_scalar_math_hyperbolic_invalid: " +
                                     message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

::fabric::ImplementationFamilyId familyId(HyperbolicOperation operation) {
  switch (operation) {
  case HyperbolicOperation::Sinh:
    return ::fabric::ImplementationFamilyId::ScalarMathSinh;
  case HyperbolicOperation::Cosh:
    return ::fabric::ImplementationFamilyId::ScalarMathCosh;
  case HyperbolicOperation::Tanh:
    return ::fabric::ImplementationFamilyId::ScalarMathTanh;
  }
  llvm_unreachable("unknown hyperbolic operation");
}

::dataflow::OperationSchemaId schemaId(HyperbolicOperation operation) {
  switch (operation) {
  case HyperbolicOperation::Sinh:
    return ::dataflow::OperationSchemaId::MathSinh;
  case HyperbolicOperation::Cosh:
    return ::dataflow::OperationSchemaId::MathCosh;
  case HyperbolicOperation::Tanh:
    return ::dataflow::OperationSchemaId::MathTanh;
  }
  llvm_unreachable("unknown hyperbolic operation");
}

llvm::StringRef operationName(HyperbolicOperation operation) {
  switch (operation) {
  case HyperbolicOperation::Sinh:
    return "sinh";
  case HyperbolicOperation::Cosh:
    return "cosh";
  case HyperbolicOperation::Tanh:
    return "tanh";
  }
  llvm_unreachable("unknown hyperbolic operation");
}

unsigned formatCode(const Format &format) {
  if (format == Format{5, 10})
    return 0;
  if (format == Format{8, 7})
    return 1;
  if (format == Format{8, 23})
    return 2;
  llvm_unreachable("unsupported hyperbolic format");
}

llvm::Expected<LoweredMode>
lowerMode(HyperbolicOperation operation,
          const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != schemaId(operation) || actor.type.getNumInputs() != 1 ||
      actor.type.getNumResults() != 1)
    return invalid("behavior is not the selected unary hyperbolic operation");
  if (actor.type.getResult(0) != actor.type.getInput(0) ||
      llvm::isa<mlir::VectorType>(actor.type.getInput(0)))
    return invalid("behavior does not have one uniform scalar floating type");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&actor.payload);
  if (!payload || payload->accuracy != SpecialMathAccuracyTier::Max4Ulp ||
      payload->flags != mlir::arith::FastMathFlags::afn)
    return invalid("behavior has the wrong typed special-math payload");
  auto format = detail::resolvePortableFloatFormat(actor.type.getInput(0));
  if (!format || !(*format == Format{5, 10} || *format == Format{8, 7} ||
                   *format == Format{8, 23}))
    return invalid("sealed relation contains an unsupported floating format");
  return LoweredMode{*format, formatCode(*format)};
}

bool hasSupportedBehavior(const ::fabric::ScalarSpecialMathParams &params) {
  const auto &behavior = params.behavior;
  return behavior.roundingModes.size() == 1 &&
         behavior.roundingModes.contains(
             mlir::arith::RoundingMode::to_nearest_even) &&
         behavior.nanBehaviors.size() == 1 &&
         behavior.nanBehaviors.contains(::fabric::FloatNaNBehavior::IEEE) &&
         behavior.subnormalBehaviors.size() == 1 &&
         behavior.subnormalBehaviors.contains(
             ::fabric::FloatSubnormalBehavior::Preserve) &&
         behavior.signedZeroBehaviors.size() == 1 &&
         behavior.signedZeroBehaviors.contains(
             ::fabric::FloatSignedZeroBehavior::Preserve) &&
         behavior.requiredFastMath == mlir::arith::FastMathFlags::afn;
}

std::string functionPrefix(HyperbolicOperation operation) {
  return ("loom_hyperbolic_" + operationName(operation)).str();
}

void emitTableFunction(llvm::raw_ostream &output, llvm::StringRef name,
                       const std::array<std::int64_t, 64> &table) {
  output << "function automatic signed [63:0] " << name
         << "(input [5:0] table_index);\n"
         << "  begin\n"
         << "    case (table_index)\n";
  for (std::size_t index = 0; index != table.size(); ++index)
    output << "      6'd" << index << ": " << name << " = 64'sd" << table[index]
           << ";\n";
  output << "      default: " << name << " = 64'sd" << table.front() << ";\n"
         << "    endcase\n"
         << "  end\n"
         << "endfunction\n\n";
}

std::string buildCoreFunctions(HyperbolicOperation operation) {
  const std::string prefix = functionPrefix(operation);
  const std::string multiply = prefix + "_mul_q36";
  const std::string positiveTable = prefix + "_exp_table";
  const std::string negativeTable = prefix + "_negative_exp_table";
  const std::string fixed = prefix + "_fixed";
  const std::string pack = prefix + "_pack";
  const std::string core = prefix + "_core";

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic signed [63:0] " << multiply
         << "(input signed [63:0] lhs, input signed [63:0] rhs);\n"
         << "  reg signed [127:0] wide_lhs;\n"
         << "  reg signed [127:0] wide_rhs;\n"
         << "  reg signed [127:0] product;\n"
         << "  reg signed [127:0] shifted_product;\n"
         << "  begin\n"
         << "    wide_lhs = {{64{lhs[63]}}, lhs};\n"
         << "    wide_rhs = {{64{rhs[63]}}, rhs};\n"
         << "    product = wide_lhs * wide_rhs;\n"
         << "    shifted_product = product >>> 36;\n"
         << "    " << multiply << " = shifted_product[63:0];\n"
         << "  end\n"
         << "endfunction\n\n";
  emitTableFunction(output, positiveTable, kExpTable);
  emitTableFunction(output, negativeTable, kNegativeExpTable);

  output << "function automatic [79:0] " << fixed
         << "(input signed [63:0] magnitude_input);\n"
         << "  reg signed [63:0] positive_remainder;\n"
         << "  reg signed [63:0] negative_remainder;\n"
         << "  reg signed [63:0] square;\n"
         << "  reg signed [63:0] cube;\n"
         << "  reg signed [63:0] polynomial;\n"
         << "  reg signed [63:0] negative_polynomial;\n"
         << "  reg signed [63:0] exponential_mantissa;\n"
         << "  reg signed [63:0] negative_exponential;\n"
         << "  reg signed [63:0] factor;\n"
         << "  reg signed [63:0] denominator;\n"
         << "  reg signed [63:0] reciprocal;\n"
         << "  reg signed [63:0] result_magnitude;\n"
         << "  integer positive_scale;\n"
         << "  integer negative_scale;\n"
         << "  integer positive_index;\n"
         << "  integer negative_index;\n"
         << "  integer result_scale;\n"
         << "  integer shift_index;\n"
         << "  integer iteration;\n"
         << "  begin\n"
         << "    positive_remainder = magnitude_input;\n"
         << "    positive_scale = 0;\n"
         << "    for (shift_index = 7; shift_index >= 0; "
            "shift_index = shift_index - 1) begin\n"
         << "      if (positive_remainder >= (64'sd47632711549 <<< "
            "shift_index)) begin\n"
         << "        positive_remainder = positive_remainder - "
            "(64'sd47632711549 <<< shift_index);\n"
         << "        positive_scale = positive_scale + (1 << shift_index);\n"
         << "      end\n"
         << "    end\n"
         << "    positive_index = 0;\n"
         << "    for (shift_index = 5; shift_index >= 0; "
            "shift_index = shift_index - 1) begin\n"
         << "      if (positive_remainder >= (64'sd744261118 <<< "
            "shift_index)) begin\n"
         << "        positive_remainder = positive_remainder - "
            "(64'sd744261118 <<< shift_index);\n"
         << "        positive_index = positive_index + (1 << shift_index);\n"
         << "      end\n"
         << "    end\n"
         << "    square = " << multiply
         << "(positive_remainder, positive_remainder);\n"
         << "    cube = " << multiply << "(square, positive_remainder);\n"
         << "    polynomial = 64'sd68719476736 + positive_remainder + "
            "(square >>> 1) + cube / 6;\n"
         << "    exponential_mantissa = " << multiply << "(" << positiveTable
         << "(positive_index[5:0]), polynomial);\n"
         << "    negative_remainder = magnitude_input <<< 1;\n"
         << "    negative_scale = 0;\n"
         << "    for (shift_index = 8; shift_index >= 0; "
            "shift_index = shift_index - 1) begin\n"
         << "      if (negative_remainder >= (64'sd47632711549 <<< "
            "shift_index)) begin\n"
         << "        negative_remainder = negative_remainder - "
            "(64'sd47632711549 <<< shift_index);\n"
         << "        negative_scale = negative_scale + (1 << shift_index);\n"
         << "      end\n"
         << "    end\n"
         << "    negative_index = 0;\n"
         << "    for (shift_index = 5; shift_index >= 0; "
            "shift_index = shift_index - 1) begin\n"
         << "      if (negative_remainder >= (64'sd744261118 <<< "
            "shift_index)) begin\n"
         << "        negative_remainder = negative_remainder - "
            "(64'sd744261118 <<< shift_index);\n"
         << "        negative_index = negative_index + (1 << shift_index);\n"
         << "      end\n"
         << "    end\n"
         << "    square = " << multiply
         << "(negative_remainder, negative_remainder);\n"
         << "    cube = " << multiply << "(square, negative_remainder);\n"
         << "    negative_polynomial = 64'sd68719476736 - "
            "negative_remainder + (square >>> 1) - cube / 6;\n"
         << "    negative_exponential = " << multiply << "(" << negativeTable
         << "(negative_index[5:0]), negative_polynomial);\n"
         << "    if (negative_scale >= 63)\n"
         << "      negative_exponential = 0;\n"
         << "    else\n"
         << "      negative_exponential = negative_exponential >>> "
            "negative_scale;\n";
  if (operation == HyperbolicOperation::Tanh) {
    output
        << "    denominator = 64'sd68719476736 + negative_exponential;\n"
        << "    reciprocal = 64'sd97015731863 - " << multiply
        << "(64'sd32338577288, denominator);\n"
        << "    for (iteration = 0; iteration < 4; iteration = iteration + 1)\n"
        << "      reciprocal = " << multiply << "(reciprocal, "
        << "64'sd137438953472 - " << multiply << "(denominator, reciprocal));\n"
        << "    result_magnitude = " << multiply
        << "(64'sd68719476736 - negative_exponential, reciprocal);\n"
        << "    result_scale = 0;\n";
  } else {
    output << "    factor = 64'sd68719476736 "
           << (operation == HyperbolicOperation::Sinh ? "-" : "+")
           << " negative_exponential;\n"
           << "    result_magnitude = " << multiply
           << "(exponential_mantissa, factor);\n"
           << "    result_scale = positive_scale - 1;\n";
  }
  output << "    " << fixed << "[63:0] = result_magnitude;\n"
         << "    " << fixed << "[79:64] = result_scale[15:0];\n"
         << "  end\n"
         << "endfunction\n\n";

  output << "function automatic [31:0] " << pack
         << "(input signed [63:0] magnitude, input signed [15:0] scale, "
            "input sign_result, input integer exponent_bits, "
            "input integer fraction_bits, input integer bias, "
            "input integer width);\n"
         << "  reg [63:0] rounded;\n"
         << "  reg guard;\n"
         << "  reg sticky;\n"
         << "  reg found;\n"
         << "  reg [31:0] result_value;\n"
         << "  reg [31:0] fraction_mask;\n"
         << "  reg [31:0] exponent_mask;\n"
         << "  integer leading_index;\n"
         << "  integer result_exponent;\n"
         << "  integer encoded_exponent;\n"
         << "  integer minimum_exponent;\n"
         << "  integer maximum_exponent;\n"
         << "  integer scale_value;\n"
         << "  integer distance;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    fraction_mask = (32'd1 << fraction_bits) - 1;\n"
         << "    exponent_mask = (32'd1 << exponent_bits) - 1;\n"
         << "    minimum_exponent = 1 - bias;\n"
         << "    maximum_exponent = bias;\n"
         << "    scale_value = $signed({{16{scale[15]}}, scale});\n"
         << "    result_value = 0;\n"
         << "    if (sign_result) result_value = 32'd1 << (width - 1);\n"
         << "    if (magnitude != 0) begin\n"
         << "      leading_index = 0;\n"
         << "      found = 1'b0;\n"
         << "      for (index = 63; index >= 0; index = index - 1) begin\n"
         << "        if (!found && magnitude[index]) begin\n"
         << "          leading_index = index;\n"
         << "          found = 1'b1;\n"
         << "        end\n"
         << "      end\n"
         << "      result_exponent = scale_value + leading_index - 36;\n"
         << "      if (result_exponent > maximum_exponent) begin\n"
         << "        result_value = (sign_result ? "
            "(32'd1 << (width - 1)) : 0) | (exponent_mask << fraction_bits);\n"
         << "      end else begin\n"
         << "        if (result_exponent >= minimum_exponent)\n"
         << "          distance = leading_index - fraction_bits;\n"
         << "        else\n"
         << "          distance = 36 + minimum_exponent - fraction_bits - "
            "scale_value;\n"
         << "        guard = 1'b0;\n"
         << "        sticky = 1'b0;\n"
         << "        if (distance <= 0) begin\n"
         << "          rounded = magnitude << (-distance);\n"
         << "        end else if (distance >= 64) begin\n"
         << "          rounded = 0;\n"
         << "          sticky = magnitude != 0;\n"
         << "        end else begin\n"
         << "          rounded = magnitude >> distance;\n"
         << "          guard = magnitude[distance - 1];\n"
         << "          for (index = 0; index < 64; index = index + 1)\n"
         << "            if (index < distance - 1) "
            "sticky = sticky | magnitude[index];\n"
         << "        end\n"
         << "        if (guard && (sticky || rounded[0]))\n"
         << "          rounded = rounded + 1;\n"
         << "        if (result_exponent >= minimum_exponent) begin\n"
         << "          if (rounded[fraction_bits + 1]) begin\n"
         << "            rounded = rounded >> 1;\n"
         << "            result_exponent = result_exponent + 1;\n"
         << "          end\n"
         << "          if (result_exponent > maximum_exponent) begin\n"
         << "            result_value = (sign_result ? "
            "(32'd1 << (width - 1)) : 0) | "
            "(exponent_mask << fraction_bits);\n"
         << "          end else begin\n"
         << "            encoded_exponent = result_exponent + bias;\n"
         << "            result_value = (sign_result ? "
            "(32'd1 << (width - 1)) : 0) | "
            "(encoded_exponent << fraction_bits) | "
            "(rounded[31:0] & fraction_mask);\n"
         << "          end\n"
         << "        end else if (rounded[fraction_bits]) begin\n"
         << "          result_value = (sign_result ? "
            "(32'd1 << (width - 1)) : 0) | "
            "(32'd1 << fraction_bits);\n"
         << "        end else begin\n"
         << "          result_value = (sign_result ? "
            "(32'd1 << (width - 1)) : 0) | "
            "(rounded[31:0] & fraction_mask);\n"
         << "        end\n"
         << "      end\n"
         << "    end\n"
         << "    " << pack << " = result_value;\n"
         << "  end\n"
         << "endfunction\n\n";

  output << "function automatic [31:0] " << core
         << "(input [31:0] raw_input, input [1:0] format_code);\n"
         << "  integer exponent_bits;\n"
         << "  integer fraction_bits;\n"
         << "  integer bias;\n"
         << "  integer width;\n"
         << "  integer exponent_value;\n"
         << "  integer shift_amount;\n"
         << "  reg [31:0] value;\n"
         << "  reg [31:0] value_mask;\n"
         << "  reg [31:0] fraction_mask;\n"
         << "  reg [31:0] exponent_mask;\n"
         << "  reg [31:0] infinity;\n"
         << "  reg [31:0] quiet_bit;\n"
         << "  reg [31:0] one;\n"
         << "  reg [31:0] exponent;\n"
         << "  reg [31:0] fraction;\n"
         << "  reg sign_input;\n"
         << "  reg signed [63:0] significand;\n"
         << "  reg signed [63:0] fixed_input;\n"
         << "  reg [79:0] fixed_result;\n"
         << "  begin\n"
         << "    case (format_code)\n"
         << "      2'd0: begin exponent_bits = 5; fraction_bits = 10; "
            "bias = 15; width = 16; end\n"
         << "      2'd1: begin exponent_bits = 8; fraction_bits = 7; "
            "bias = 127; width = 16; end\n"
         << "      default: begin exponent_bits = 8; fraction_bits = 23; "
            "bias = 127; width = 32; end\n"
         << "    endcase\n"
         << "    value_mask = width == 32 ? 32'hffff_ffff : 32'h0000_ffff;\n"
         << "    value = raw_input & value_mask;\n"
         << "    fraction_mask = (32'd1 << fraction_bits) - 1;\n"
         << "    exponent_mask = (32'd1 << exponent_bits) - 1;\n"
         << "    infinity = exponent_mask << fraction_bits;\n"
         << "    quiet_bit = 32'd1 << (fraction_bits - 1);\n"
         << "    one = bias << fraction_bits;\n"
         << "    sign_input = value[width - 1];\n"
         << "    exponent = (value >> fraction_bits) & exponent_mask;\n"
         << "    fraction = value & fraction_mask;\n"
         << "    " << core << " = infinity | quiet_bit;\n"
         << "    if (exponent == exponent_mask && fraction != 0) begin\n"
         << "      " << core << " = value | quiet_bit;\n"
         << "    end else if (exponent == exponent_mask) begin\n";
  if (operation == HyperbolicOperation::Tanh) {
    output << "      " << core
           << " = (sign_input ? (32'd1 << (width - 1)) : 0) | one;\n";
  } else {
    output << "      " << core << " = "
           << (operation == HyperbolicOperation::Sinh
                   ? "(sign_input ? (32'd1 << (width - 1)) : 0) | infinity"
                   : "infinity")
           << ";\n";
  }
  output << "    end else if (exponent == 0) begin\n"
         << "      " << core << " = "
         << (operation == HyperbolicOperation::Cosh ? "one" : "value") << ";\n"
         << "    end else begin\n"
         << "      exponent_value = $signed(exponent) - bias;\n"
         << "      if (exponent_value <= -13) begin\n"
         << "        " << core << " = "
         << (operation == HyperbolicOperation::Cosh ? "one" : "value") << ";\n";
  if (operation == HyperbolicOperation::Tanh) {
    output << "      end else if (exponent_value >= 4) begin\n"
           << "        " << core
           << " = (sign_input ? (32'd1 << (width - 1)) : 0) | one;\n";
  } else {
    output << "      end else if (exponent_value >= 7) begin\n"
           << "        " << core << " = "
           << (operation == HyperbolicOperation::Sinh
                   ? "(sign_input ? (32'd1 << (width - 1)) : 0) | infinity"
                   : "infinity")
           << ";\n";
  }
  output << "      end else begin\n"
         << "        significand = (64'sd1 << fraction_bits) | "
            "{{32{1'b0}}, fraction};\n"
         << "        shift_amount = exponent_value - fraction_bits + 36;\n"
         << "        fixed_input = significand <<< shift_amount;\n"
         << "        fixed_result = " << fixed << "(fixed_input);\n"
         << "        " << core << " = " << pack
         << "($signed(fixed_result[63:0]), "
            "$signed(fixed_result[79:64]), "
         << (operation == HyperbolicOperation::Cosh ? "1'b0" : "sign_input")
         << ", exponent_bits, fraction_bits, bias, width);\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildDispatchFunction(HyperbolicOperation operation,
                                  llvm::ArrayRef<LoweredMode> loweredModes,
                                  llvm::ArrayRef<Mode> modes,
                                  std::size_t inactiveMode,
                                  const ConfigurationFieldEncoding *field,
                                  const FiniteCodebookEncoding *codebook) {
  const std::string prefix = functionPrefix(operation);
  const std::string name = prefix + "_dispatch";
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [31:0] " << name << "(input [31:0] data_input";
  if (field)
    output << ", input [" << codebook->encodedBitCount - 1
           << ":0] configuration";
  output << ");\n"
         << "  reg [1:0] format_code;\n"
         << "  begin\n"
         << "    format_code = 2'd" << loweredModes[inactiveMode].formatCode
         << ";\n";
  if (field) {
    output << "    case (configuration)\n";
    for (std::size_t index = 0; index != modes.size(); ++index) {
      const llvm::APInt code = detail::decodePhysicalCode(
          modes[index].codebookEntry->physicalCode, codebook->encodedBitCount);
      output << "      " << codebook->encodedBitCount << "'d"
             << code.getZExtValue() << ": format_code = 2'd"
             << loweredModes[index].formatCode << ";\n";
    }
    output << "      default: format_code = 2'd"
           << loweredModes[inactiveMode].formatCode << ";\n"
           << "    endcase\n";
  }
  output << "    " << name << " = " << prefix
         << "_core(data_input, format_code);\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

mlir::Value callDispatch(mlir::OpBuilder &builder, mlir::Location location,
                         HyperbolicOperation operation, mlir::Value input,
                         mlir::Value configuration) {
  llvm::SmallVector<mlir::Value, 2> operands = {input};
  std::string expression = functionPrefix(operation) + "_dispatch({{0}}";
  if (configuration) {
    operands.push_back(configuration);
    expression += ", {{1}}";
  }
  expression += ")";
  return circt::sv::VerbatimExprOp::create(
      builder, location, builder.getI32Type(), expression, operands);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathHyperbolic(FabricOperationProviderRequest request,
                                        HyperbolicOperation operation) {
  const auto expectedFamily = familyId(operation);
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  const auto &descriptor = ::fabric::implementationFamily(expectedFamily);
  if (descriptor.familyId != expectedFamily ||
      ::fabric::capabilityParamsSchema(
          request.capability.parameterizedCapability) !=
          descriptor.capabilityParamsSchema)
    return invalid("capability does not match its generated family descriptor");
  const auto *params = std::get_if<::fabric::ScalarSpecialMathParams>(
      &request.capability.parameterizedCapability);
  if (!params)
    return invalid("capability has the wrong special-math parameter schema");
  if (request.capability.enabledOperationSchemas !=
          std::vector<::dataflow::OperationSchemaId>{schemaId(operation)} ||
      descriptor.admittedSchemas.size() != 1 ||
      descriptor.admittedSchemas.front() != schemaId(operation))
    return invalid("capability does not contain its exact registered schema");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return unsupported(request);

  if (params->accuracyGuarantee != SpecialMathAccuracyTier::Max4Ulp ||
      !hasSupportedBehavior(*params) ||
      params->formats.contains(::fabric::FloatFormat::F64))
    return unsupported(request);

  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       request.capability.physicalPorts) {
    if (port.reference.direction == fabric::FabricPortDirection::Input)
      inputs.push_back(&port);
    else if (port.reference.direction == fabric::FabricPortDirection::Output)
      outputs.push_back(&port);
    else
      return invalid("capability has a physical port with unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  if (inputs.size() != 1 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || outputs[0]->reference.ordinal != 0)
    return unsupported(request);
  if (inputs[0]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return invalid("capability has a zero-width physical data port");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto &domain = relation->finiteBehaviorDomain();
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free hyperbolic relation is not a singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured hyperbolic relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured hyperbolic capability requires one field");
    field = request.configurationAbi.findOperationField(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid("codebook does not exactly cover the behavior domain");
    if (codebook->encodedBitCount == 0 || codebook->encodedBitCount > 32)
      return unsupported(request);
    for (const auto &point : domain) {
      if (!point.semanticConfiguration)
        return invalid("configured hyperbolic behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook has no entry for an admitted semantic value");
      modes.push_back({point.representativeActor, entry});
    }
  }
  if (modes.empty())
    return invalid("sealed hyperbolic behavior relation is empty");

  std::size_t inactiveMode = 0;
  if (field) {
    const auto inactive = llvm::find_if(modes, [&](const Mode &mode) {
      return llvm::ArrayRef<std::uint8_t>(mode.codebookEntry->semanticValue)
          .equals(field->inactiveValue);
    });
    if (inactive == modes.end())
      return invalid("ABI inactive value is outside the behavior domain");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  std::vector<LoweredMode> loweredModes;
  std::set<unsigned> formats;
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(operation, mode.actor);
    if (!lowered)
      return lowered.takeError();
    if (!formats.insert(lowered->formatCode).second)
      return invalid("sealed relation contains a duplicate floating format");
    if (lowered->format.width() > inputs[0]->payloadWidthBits ||
        lowered->format.width() > outputs[0]->payloadWidthBits)
      return unsupported(request);
    loweredModes.push_back(*lowered);
  }

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        std::string declarations = buildCoreFunctions(operation);
        declarations += buildDispatchFunction(operation, loweredModes, modes,
                                              inactiveMode, field, codebook);
        circt::sv::VerbatimOp::create(bodyBuilder, location,
                                      bodyBuilder.getStringAttr(declarations));
        mlir::Value input = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"), 32);
        mlir::Value configuration;
        if (field)
          configuration = accessor.getInput(
              "config_" +
              std::to_string(
                  request.capability.configurationFieldSchema.front().ordinal));
        mlir::Value result = callDispatch(bodyBuilder, location, operation,
                                          input, configuration);
        accessor.setOutput("data_output_0", detail::resizeUnsigned(
                                                bodyBuilder, location, result,
                                                outputs[0]->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathSinh(FabricOperationProviderRequest request) {
  return materializePortableScalarMathHyperbolic(std::move(request),
                                                 HyperbolicOperation::Sinh);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathCosh(FabricOperationProviderRequest request) {
  return materializePortableScalarMathHyperbolic(std::move(request),
                                                 HyperbolicOperation::Cosh);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathTanh(FabricOperationProviderRequest request) {
  return materializePortableScalarMathHyperbolic(std::move(request),
                                                 HyperbolicOperation::Tanh);
}

} // namespace

llvm::Error registerPortableScalarMathHyperbolicProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry staged = registry;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathSinh,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathSinh}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathCosh,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathCosh}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathTanh,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathTanh}))
    return error;
  registry = std::move(staged);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
