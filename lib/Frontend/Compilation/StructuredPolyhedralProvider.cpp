#include "StructuredPolyhedralProvider.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/local_space.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::frontend::detail {
namespace {

constexpr unsigned long maximumIslOperations = 1'000'000;
constexpr std::uint64_t maximumProviderConstraints = 1'000'000;

template <typename T, T *(*Release)(T *)> struct IslDeleter final {
  void operator()(T *value) const {
    if (value)
      Release(value);
  }
};

template <typename T, T *(*Release)(T *)>
using IslOwner = std::unique_ptr<T, IslDeleter<T, Release>>;

struct IslContextDeleter final {
  void operator()(isl_ctx *context) const {
    if (context)
      isl_ctx_free(context);
  }
};

using IslContext = std::unique_ptr<isl_ctx, IslContextDeleter>;
using IslConstraint = IslOwner<isl_constraint, isl_constraint_free>;
using IslBasicSet = IslOwner<isl_basic_set, isl_basic_set_free>;
using IslSet = IslOwner<isl_set, isl_set_free>;
using IslBasicMap = IslOwner<isl_basic_map, isl_basic_map_free>;
using IslMap = IslOwner<isl_map, isl_map_free>;
using IslUnionSet = IslOwner<isl_union_set, isl_union_set_free>;
using IslUnionMap = IslOwner<isl_union_map, isl_union_map_free>;
using IslSchedule = IslOwner<isl_schedule, isl_schedule_free>;
using IslScheduleConstraints =
    IslOwner<isl_schedule_constraints, isl_schedule_constraints_free>;
using IslAff = IslOwner<isl_aff, isl_aff_free>;
using IslVal = IslOwner<isl_val, isl_val_free>;
using IslAstBuild = IslOwner<isl_ast_build, isl_ast_build_free>;
using IslAstExpression = IslOwner<isl_ast_expr, isl_ast_expr_free>;
using IslAstNode = IslOwner<isl_ast_node, isl_ast_node_free>;
using IslAstNodeList = IslOwner<isl_ast_node_list, isl_ast_node_list_free>;
using IslId = IslOwner<isl_id, isl_id_free>;

llvm::Error providerError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "polly_isl_provider_error: " + message);
}

class AstRepresentationUnavailable final
    : public llvm::ErrorInfo<AstRepresentationUnavailable> {
public:
  static char ID;

  void log(llvm::raw_ostream &stream) const override {
    stream << "the ISL AST exceeds the ordinary MLIR representation";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

char AstRepresentationUnavailable::ID = 0;

std::string tupleName(std::uint64_t statementOrdinal) {
  return "S" + std::to_string(statementOrdinal);
}

std::string parameterName(std::uint64_t parameterOrdinal) {
  return "p" + std::to_string(parameterOrdinal);
}

struct ParameterTable final {
  llvm::DenseMap<mlir::Value, unsigned> ordinals;
  std::vector<mlir::Value> values;
};

std::optional<PolyhedralScheduleProviderRefusalKind>
collectParameters(const mlir::affine::FlatAffineValueConstraints &relation,
                  ParameterTable &parameters) {
  if (relation.getNumLocalVars() != 0)
    return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
  const unsigned offset = relation.getNumDimVars();
  for (unsigned index = 0; index != relation.getNumSymbolVars(); ++index) {
    const unsigned position = offset + index;
    if (!relation.hasValue(position))
      return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
    mlir::Value value = relation.getValue(position);
    if (parameters.ordinals.count(value))
      continue;
    if (parameters.values.size() == std::numeric_limits<unsigned>::max())
      return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
    const unsigned ordinal = parameters.values.size();
    parameters.ordinals.try_emplace(value, ordinal);
    parameters.values.push_back(value);
  }
  return std::nullopt;
}

llvm::Expected<isl_space *> setSpace(isl_ctx *context, unsigned dimensions,
                                     const ParameterTable &parameters,
                                     llvm::StringRef tuple) {
  isl_space *space =
      isl_space_set_alloc(context, parameters.values.size(), dimensions);
  if (!space)
    return providerError("cannot allocate an ISL set space");
  for (unsigned index = 0; index != parameters.values.size(); ++index) {
    const std::string name = parameterName(index);
    space = isl_space_set_dim_name(space, isl_dim_param, index, name.c_str());
    if (!space)
      return providerError("cannot name an ISL set parameter");
  }
  space = isl_space_set_tuple_name(space, isl_dim_set, tuple.str().c_str());
  if (!space)
    return providerError("cannot name an ISL statement tuple");
  return space;
}

llvm::Expected<isl_space *>
mapSpace(isl_ctx *context, unsigned sourceDimensions,
         unsigned destinationDimensions, const ParameterTable &parameters,
         llvm::StringRef sourceTuple, llvm::StringRef destinationTuple) {
  isl_space *space = isl_space_alloc(context, parameters.values.size(),
                                     sourceDimensions, destinationDimensions);
  if (!space)
    return providerError("cannot allocate an ISL relation space");
  for (unsigned index = 0; index != parameters.values.size(); ++index) {
    const std::string name = parameterName(index);
    space = isl_space_set_dim_name(space, isl_dim_param, index, name.c_str());
    if (!space)
      return providerError("cannot name an ISL relation parameter");
  }
  space =
      isl_space_set_tuple_name(space, isl_dim_in, sourceTuple.str().c_str());
  if (!space)
    return providerError("cannot name an ISL source tuple");
  space = isl_space_set_tuple_name(space, isl_dim_out,
                                   destinationTuple.str().c_str());
  if (!space)
    return providerError("cannot name an ISL destination tuple");
  return space;
}

llvm::Expected<isl_constraint *> setCoefficient(isl_constraint *constraint,
                                                enum isl_dim_type type,
                                                unsigned position,
                                                std::int64_t coefficient) {
  IslConstraint owned(constraint);
  IslOwner<isl_val, isl_val_free> value(
      isl_val_int_from_si(isl_constraint_get_ctx(owned.get()), coefficient));
  if (!value)
    return providerError("cannot allocate an ISL coefficient");
  constraint = isl_constraint_set_coefficient_val(owned.release(), type,
                                                  position, value.release());
  if (!constraint)
    return providerError("cannot assign an ISL coefficient");
  return constraint;
}

llvm::Expected<isl_constraint *> setConstant(isl_constraint *constraint,
                                             std::int64_t constant) {
  IslConstraint owned(constraint);
  IslOwner<isl_val, isl_val_free> value(
      isl_val_int_from_si(isl_constraint_get_ctx(owned.get()), constant));
  if (!value)
    return providerError("cannot allocate an ISL constant");
  constraint =
      isl_constraint_set_constant_val(owned.release(), value.release());
  if (!constraint)
    return providerError("cannot assign an ISL constant");
  return constraint;
}

llvm::Expected<isl_constraint *>
translateConstraint(isl_local_space *localSpace,
                    const mlir::affine::FlatAffineValueConstraints &source,
                    llvm::ArrayRef<std::int64_t> row, bool equality,
                    unsigned sourceDimensions, unsigned destinationDimensions,
                    const ParameterTable &parameters) {
  IslConstraint constraint(equality
                               ? isl_constraint_alloc_equality(localSpace)
                               : isl_constraint_alloc_inequality(localSpace));
  if (!constraint)
    return providerError("cannot allocate an ISL constraint");
  if (source.getNumDimVars() != sourceDimensions + destinationDimensions ||
      row.size() != source.getNumCols())
    return providerError("a Presburger relation has inconsistent arity");

  for (unsigned index = 0; index != sourceDimensions; ++index) {
    auto updated =
        setCoefficient(constraint.release(), isl_dim_in, index, row[index]);
    if (!updated)
      return updated.takeError();
    constraint.reset(*updated);
  }
  for (unsigned index = 0; index != destinationDimensions; ++index) {
    const enum isl_dim_type type =
        sourceDimensions == 0 ? isl_dim_set : isl_dim_out;
    auto updated = setCoefficient(constraint.release(), type, index,
                                  row[sourceDimensions + index]);
    if (!updated)
      return updated.takeError();
    constraint.reset(*updated);
  }
  const unsigned symbolOffset = source.getNumDimVars();
  for (unsigned index = 0; index != source.getNumSymbolVars(); ++index) {
    const unsigned position = symbolOffset + index;
    if (!source.hasValue(position))
      return providerError("a relation symbol lost its MLIR value identity");
    auto found = parameters.ordinals.find(source.getValue(position));
    if (found == parameters.ordinals.end())
      return providerError(
          "a relation symbol is absent from the parameter table");
    auto updated = setCoefficient(constraint.release(), isl_dim_param,
                                  found->second, row[position]);
    if (!updated)
      return updated.takeError();
    constraint.reset(*updated);
  }
  auto updated = setConstant(constraint.release(), row.back());
  if (!updated)
    return updated.takeError();
  return *updated;
}

llvm::Expected<isl_set *>
translateDomain(isl_ctx *context, const PolyhedralStatementDomain &statement,
                const ParameterTable &parameters) {
  if (!statement.domain)
    return providerError("a statement has no exact domain");
  const auto &domain = *statement.domain;
  auto space = setSpace(context, domain.getNumDimVars(), parameters,
                        tupleName(statement.statementOrdinal));
  if (!space)
    return space.takeError();
  IslBasicSet result(isl_basic_set_universe(*space));
  if (!result)
    return providerError("cannot allocate an ISL statement domain");
  for (unsigned index = 0; index != domain.getNumInequalities(); ++index) {
    auto row = domain.getInequality64(index);
    isl_local_space *local =
        isl_local_space_from_space(isl_basic_set_get_space(result.get()));
    auto constraint = translateConstraint(local, domain, row, false,
                                          /*sourceDimensions=*/0,
                                          domain.getNumDimVars(), parameters);
    if (!constraint)
      return constraint.takeError();
    result.reset(isl_basic_set_add_constraint(result.release(), *constraint));
    if (!result)
      return providerError("cannot add an ISL domain inequality");
  }
  for (unsigned index = 0; index != domain.getNumEqualities(); ++index) {
    auto row = domain.getEquality64(index);
    isl_local_space *local =
        isl_local_space_from_space(isl_basic_set_get_space(result.get()));
    auto constraint = translateConstraint(local, domain, row, true,
                                          /*sourceDimensions=*/0,
                                          domain.getNumDimVars(), parameters);
    if (!constraint)
      return constraint.takeError();
    result.reset(isl_basic_set_add_constraint(result.release(), *constraint));
    if (!result)
      return providerError("cannot add an ISL domain equality");
  }
  return isl_set_from_basic_set(result.release());
}

llvm::Expected<isl_map *>
translateDependence(isl_ctx *context,
                    const PolyhedralDependenceRelation &dependence,
                    const ParameterTable &parameters) {
  auto space = mapSpace(context, dependence.sourceDimensionCount,
                        dependence.destinationDimensionCount, parameters,
                        tupleName(dependence.sourceStatementOrdinal),
                        tupleName(dependence.destinationStatementOrdinal));
  if (!space)
    return space.takeError();
  IslBasicMap result(isl_basic_map_universe(*space));
  if (!result)
    return providerError("cannot allocate an ISL dependence relation");
  if (!dependence.relation) {
    const unsigned commonDimensions = std::min(
        dependence.sourceDimensionCount, dependence.destinationDimensionCount);
    for (unsigned index = 0; index != commonDimensions; ++index) {
      IslConstraint constraint(isl_constraint_alloc_equality(
          isl_local_space_from_space(isl_basic_map_get_space(result.get()))));
      if (!constraint)
        return providerError("cannot allocate a precedence equality");
      auto source = setCoefficient(constraint.release(), isl_dim_in, index, -1);
      if (!source)
        return source.takeError();
      constraint.reset(*source);
      auto destination =
          setCoefficient(constraint.release(), isl_dim_out, index, 1);
      if (!destination)
        return destination.takeError();
      result.reset(
          isl_basic_map_add_constraint(result.release(), *destination));
      if (!result)
        return providerError("cannot add a precedence equality");
    }
    return isl_map_from_basic_map(result.release());
  }
  const auto &relation = *dependence.relation;
  for (unsigned index = 0; index != relation.getNumInequalities(); ++index) {
    auto row = relation.getInequality64(index);
    isl_local_space *local =
        isl_local_space_from_space(isl_basic_map_get_space(result.get()));
    auto constraint = translateConstraint(
        local, relation, row, false, dependence.sourceDimensionCount,
        dependence.destinationDimensionCount, parameters);
    if (!constraint)
      return constraint.takeError();
    result.reset(isl_basic_map_add_constraint(result.release(), *constraint));
    if (!result)
      return providerError("cannot add an ISL dependence inequality");
  }
  for (unsigned index = 0; index != relation.getNumEqualities(); ++index) {
    auto row = relation.getEquality64(index);
    isl_local_space *local =
        isl_local_space_from_space(isl_basic_map_get_space(result.get()));
    auto constraint = translateConstraint(
        local, relation, row, true, dependence.sourceDimensionCount,
        dependence.destinationDimensionCount, parameters);
    if (!constraint)
      return constraint.takeError();
    result.reset(isl_basic_map_add_constraint(result.release(), *constraint));
    if (!result)
      return providerError("cannot add an ISL dependence equality");
  }
  return isl_map_from_basic_map(result.release());
}

struct BandSummary final {
  std::uint64_t bands = 0;
  std::uint64_t dimensions = 0;
  std::uint64_t coincidentDimensions = 0;
  bool failed = false;
};

bool constraintLess(const StructuredPolyhedralConstraintView &lhs,
                    const StructuredPolyhedralConstraintView &rhs) {
  if (lhs.kind != rhs.kind)
    return lhs.kind < rhs.kind;
  return lhs.coefficients < rhs.coefficients;
}

bool divisionLess(const StructuredPolyhedralDivisionView &lhs,
                  const StructuredPolyhedralDivisionView &rhs) {
  if (lhs.denominator != rhs.denominator)
    return lhs.denominator < rhs.denominator;
  return lhs.numerator < rhs.numerator;
}

bool pieceLess(const StructuredPolyhedralSchedulePieceView &lhs,
               const StructuredPolyhedralSchedulePieceView &rhs) {
  if (lhs.sourceDimensionCount != rhs.sourceDimensionCount)
    return lhs.sourceDimensionCount < rhs.sourceDimensionCount;
  if (lhs.scheduleDimensionCount != rhs.scheduleDimensionCount)
    return lhs.scheduleDimensionCount < rhs.scheduleDimensionCount;
  if (lhs.parameterCount != rhs.parameterCount)
    return lhs.parameterCount < rhs.parameterCount;
  if (lhs.divisions != rhs.divisions)
    return std::lexicographical_compare(
        lhs.divisions.begin(), lhs.divisions.end(), rhs.divisions.begin(),
        rhs.divisions.end(), divisionLess);
  return std::lexicographical_compare(
      lhs.constraints.begin(), lhs.constraints.end(), rhs.constraints.begin(),
      rhs.constraints.end(), constraintLess);
}

bool readInteger(IslVal value, std::int64_t &result) {
  if (!value || isl_val_is_int(value.get()) != isl_bool_true)
    return false;
  isl_ctx *context = isl_val_get_ctx(value.get());
  const long integer = isl_val_get_num_si(value.get());
  if (isl_ctx_last_error(context) != isl_error_none)
    return false;
  result = static_cast<std::int64_t>(integer);
  return true;
}

llvm::Expected<std::optional<std::int64_t>> readAstInteger(IslVal value) {
  if (!value)
    return providerError("cannot read an ISL AST integer");
  const isl_bool isInteger = isl_val_is_int(value.get());
  if (isInteger < 0)
    return providerError("cannot classify an ISL AST integer");
  if (isInteger == isl_bool_false)
    return providerError("an ISL AST integer is not integral");
  const isl_size chunkCount =
      isl_val_n_abs_num_chunks(value.get(), sizeof(std::uint64_t));
  if (chunkCount < 0)
    return providerError("cannot measure an ISL AST integer");
  if (chunkCount > 1)
    return std::optional<std::int64_t>{};
  std::uint64_t magnitude = 0;
  if (isl_val_get_abs_num_chunks(value.get(), sizeof(magnitude), &magnitude) <
      0)
    return providerError("cannot read an ISL AST integer magnitude");
  if (isl_val_sgn(value.get()) < 0) {
    constexpr std::uint64_t minimumMagnitude = std::uint64_t{1} << 63;
    if (magnitude > minimumMagnitude)
      return std::optional<std::int64_t>{};
    if (magnitude == minimumMagnitude)
      return std::numeric_limits<std::int64_t>::min();
    return -static_cast<std::int64_t>(magnitude);
  }
  if (magnitude >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return std::optional<std::int64_t>{};
  return static_cast<std::int64_t>(magnitude);
}

bool readCoefficient(isl_constraint *constraint, enum isl_dim_type type,
                     unsigned position, std::int64_t &result) {
  return readInteger(
      IslVal(isl_constraint_get_coefficient_val(constraint, type, position)),
      result);
}

struct ConstraintFreezeContext final {
  StructuredPolyhedralSchedulePieceView *piece = nullptr;
  bool failed = false;
};

isl_stat freezeConstraint(isl_constraint *rawConstraint, void *opaque) {
  IslConstraint constraint(rawConstraint);
  auto &context = *static_cast<ConstraintFreezeContext *>(opaque);
  StructuredPolyhedralSchedulePieceView &piece = *context.piece;
  const std::uint64_t localCount = piece.divisions.size();
  const std::uint64_t coefficientCount = piece.sourceDimensionCount +
                                         piece.scheduleDimensionCount +
                                         piece.parameterCount + localCount + 1;
  if (coefficientCount > std::numeric_limits<std::size_t>::max()) {
    context.failed = true;
    return isl_stat_error;
  }
  StructuredPolyhedralConstraintView frozen;
  const isl_bool equality = isl_constraint_is_equality(constraint.get());
  if (equality < 0) {
    context.failed = true;
    return isl_stat_error;
  }
  frozen.kind = equality == isl_bool_true
                    ? StructuredPolyhedralConstraintKind::Equality
                    : StructuredPolyhedralConstraintKind::Inequality;
  frozen.coefficients.reserve(static_cast<std::size_t>(coefficientCount));
  const auto append = [&](enum isl_dim_type type, std::uint64_t count) {
    for (std::uint64_t index = 0; index != count; ++index) {
      std::int64_t coefficient = 0;
      if (index > std::numeric_limits<unsigned>::max() ||
          !readCoefficient(constraint.get(), type, static_cast<unsigned>(index),
                           coefficient))
        return false;
      frozen.coefficients.push_back(coefficient);
    }
    return true;
  };
  if (!append(isl_dim_in, piece.sourceDimensionCount) ||
      !append(isl_dim_out, piece.scheduleDimensionCount) ||
      !append(isl_dim_param, piece.parameterCount) ||
      !append(isl_dim_div, localCount)) {
    context.failed = true;
    return isl_stat_error;
  }
  std::int64_t constant = 0;
  if (!readInteger(IslVal(isl_constraint_get_constant_val(constraint.get())),
                   constant)) {
    context.failed = true;
    return isl_stat_error;
  }
  frozen.coefficients.push_back(constant);
  piece.constraints.push_back(std::move(frozen));
  return isl_stat_ok;
}

bool readScaledAffValue(isl_aff *affine, enum isl_dim_type type,
                        unsigned position, isl_val *denominator,
                        std::int64_t &result) {
  IslVal coefficient(isl_aff_get_coefficient_val(affine, type, position));
  if (!coefficient)
    return false;
  coefficient.reset(
      isl_val_mul(coefficient.release(), isl_val_copy(denominator)));
  return readInteger(std::move(coefficient), result);
}

bool freezeDivisions(isl_basic_map *basic,
                     StructuredPolyhedralSchedulePieceView &piece) {
  const isl_size localCount = isl_basic_map_dim(basic, isl_dim_div);
  if (localCount < 0)
    return false;
  piece.divisions.resize(static_cast<std::size_t>(localCount));
  for (isl_size local = 0; local != localCount; ++local) {
    IslAff division(isl_basic_map_get_div(basic, local));
    if (!division)
      return false;
    const isl_size affineDimensions = isl_aff_dim(division.get(), isl_dim_in);
    const isl_size affineParameters =
        isl_aff_dim(division.get(), isl_dim_param);
    const isl_size affineLocals = isl_aff_dim(division.get(), isl_dim_div);
    if (affineDimensions < 0 || affineParameters < 0 || affineLocals < 0 ||
        static_cast<std::uint64_t>(affineDimensions) !=
            piece.sourceDimensionCount + piece.scheduleDimensionCount ||
        static_cast<std::uint64_t>(affineParameters) != piece.parameterCount ||
        affineLocals != localCount)
      return false;
    IslVal denominator(isl_aff_get_denominator_val(division.get()));
    std::int64_t signedDenominator = 0;
    if (!readInteger(IslVal(isl_val_copy(denominator.get())),
                     signedDenominator) ||
        signedDenominator <= 0)
      return false;
    StructuredPolyhedralDivisionView &frozen =
        piece.divisions[static_cast<std::size_t>(local)];
    frozen.denominator = static_cast<std::uint64_t>(signedDenominator);
    const std::uint64_t numeratorCount =
        piece.sourceDimensionCount + piece.scheduleDimensionCount +
        piece.parameterCount + static_cast<std::uint64_t>(localCount) + 1;
    if (numeratorCount > std::numeric_limits<std::size_t>::max())
      return false;
    frozen.numerator.reserve(static_cast<std::size_t>(numeratorCount));
    for (std::uint64_t index = 0;
         index != piece.sourceDimensionCount + piece.scheduleDimensionCount;
         ++index) {
      std::int64_t coefficient = 0;
      if (index > std::numeric_limits<unsigned>::max() ||
          !readScaledAffValue(division.get(), isl_dim_in,
                              static_cast<unsigned>(index), denominator.get(),
                              coefficient))
        return false;
      frozen.numerator.push_back(coefficient);
    }
    for (std::uint64_t index = 0; index != piece.parameterCount; ++index) {
      std::int64_t coefficient = 0;
      if (index > std::numeric_limits<unsigned>::max() ||
          !readScaledAffValue(division.get(), isl_dim_param,
                              static_cast<unsigned>(index), denominator.get(),
                              coefficient))
        return false;
      frozen.numerator.push_back(coefficient);
    }
    for (isl_size index = 0; index != localCount; ++index) {
      std::int64_t coefficient = 0;
      if (!readScaledAffValue(division.get(), isl_dim_div, index,
                              denominator.get(), coefficient))
        return false;
      frozen.numerator.push_back(coefficient);
    }
    IslVal constant(isl_aff_get_constant_val(division.get()));
    if (!constant)
      return false;
    constant.reset(
        isl_val_mul(constant.release(), isl_val_copy(denominator.get())));
    std::int64_t frozenConstant = 0;
    if (!readInteger(std::move(constant), frozenConstant))
      return false;
    frozen.numerator.push_back(frozenConstant);
  }
  return true;
}

bool accumulate(std::int64_t &value, std::int64_t delta) {
  if ((delta > 0 && value > std::numeric_limits<std::int64_t>::max() - delta) ||
      (delta < 0 && value < std::numeric_limits<std::int64_t>::min() - delta))
    return false;
  value += delta;
  return true;
}

bool negate(std::int64_t value, std::int64_t &result) {
  if (value == std::numeric_limits<std::int64_t>::min())
    return false;
  result = -value;
  return true;
}

llvm::Expected<isl_constraint *>
reconstructConstraint(isl_local_space *localSpace,
                      const StructuredPolyhedralSchedulePieceView &piece,
                      llvm::ArrayRef<std::int64_t> row, bool equality) {
  const std::uint64_t expectedWidth =
      piece.sourceDimensionCount + piece.scheduleDimensionCount +
      piece.parameterCount + piece.divisions.size() + 1;
  if (row.size() != expectedWidth)
    return providerError("a frozen schedule row has inconsistent arity");
  IslConstraint constraint(equality
                               ? isl_constraint_alloc_equality(localSpace)
                               : isl_constraint_alloc_inequality(localSpace));
  if (!constraint)
    return providerError("cannot reconstruct an ISL schedule constraint");
  std::uint64_t offset = 0;
  const auto append = [&](enum isl_dim_type type, unsigned destinationOffset,
                          std::uint64_t count) -> llvm::Error {
    for (std::uint64_t index = 0; index != count; ++index) {
      if (index > std::numeric_limits<unsigned>::max() - destinationOffset)
        return providerError("a frozen schedule dimension exceeds ISL arity");
      auto updated = setCoefficient(
          constraint.release(), type,
          destinationOffset + static_cast<unsigned>(index), row[offset++]);
      if (!updated)
        return updated.takeError();
      constraint.reset(*updated);
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = append(isl_dim_in, 0, piece.sourceDimensionCount))
    return std::move(error);
  if (llvm::Error error = append(isl_dim_out, 0, piece.scheduleDimensionCount))
    return std::move(error);
  if (llvm::Error error = append(isl_dim_param, 0, piece.parameterCount))
    return std::move(error);
  if (piece.scheduleDimensionCount > std::numeric_limits<unsigned>::max())
    return providerError("a frozen schedule range exceeds ISL arity");
  if (llvm::Error error = append(
          isl_dim_out, static_cast<unsigned>(piece.scheduleDimensionCount),
          piece.divisions.size()))
    return std::move(error);
  auto updated = setConstant(constraint.release(), row[offset]);
  if (!updated)
    return updated.takeError();
  return *updated;
}

llvm::Expected<isl_map *>
reconstructSchedulePiece(isl_map *sourceMap,
                         const StructuredPolyhedralSchedulePieceView &piece) {
  if (piece.sourceDimensionCount > std::numeric_limits<unsigned>::max() ||
      piece.scheduleDimensionCount > std::numeric_limits<unsigned>::max() ||
      piece.divisions.size() > std::numeric_limits<unsigned>::max())
    return providerError("a frozen schedule piece exceeds ISL arity");
  IslOwner<isl_space, isl_space_free> space(isl_map_get_space(sourceMap));
  if (!space)
    return providerError("cannot read the original ISL schedule space");
  space.reset(
      isl_space_add_dims(space.release(), isl_dim_out, piece.divisions.size()));
  if (!space)
    return providerError("cannot extend a reconstructed ISL schedule space");
  IslBasicMap reconstructed(isl_basic_map_universe(space.release()));
  if (!reconstructed)
    return providerError("cannot allocate a reconstructed ISL schedule map");

  const auto addRow = [&](llvm::ArrayRef<std::int64_t> row,
                          bool equality) -> llvm::Error {
    auto constraint =
        reconstructConstraint(isl_local_space_from_space(
                                  isl_basic_map_get_space(reconstructed.get())),
                              piece, row, equality);
    if (!constraint)
      return constraint.takeError();
    reconstructed.reset(
        isl_basic_map_add_constraint(reconstructed.release(), *constraint));
    if (!reconstructed)
      return providerError("cannot add a reconstructed schedule constraint");
    return llvm::Error::success();
  };
  for (const StructuredPolyhedralConstraintView &constraint : piece.constraints)
    if (llvm::Error error = addRow(
            constraint.coefficients,
            constraint.kind == StructuredPolyhedralConstraintKind::Equality))
      return std::move(error);

  for (std::size_t local = 0; local != piece.divisions.size(); ++local) {
    const StructuredPolyhedralDivisionView &division = piece.divisions[local];
    const std::uint64_t rowWidth =
        piece.sourceDimensionCount + piece.scheduleDimensionCount +
        piece.parameterCount + piece.divisions.size() + 1;
    if (division.denominator == 0 ||
        division.denominator > static_cast<std::uint64_t>(
                                   std::numeric_limits<std::int64_t>::max()) ||
        division.numerator.size() != rowWidth)
      return providerError("a frozen schedule division is malformed");
    const std::size_t localOffset = static_cast<std::size_t>(
        piece.sourceDimensionCount + piece.scheduleDimensionCount +
        piece.parameterCount);
    if (llvm::any_of(
            llvm::ArrayRef<std::int64_t>(division.numerator)
                .slice(localOffset + local, piece.divisions.size() - local),
            [](std::int64_t coefficient) { return coefficient != 0; }))
      return providerError("a frozen schedule division is cyclic");
    std::vector<std::int64_t> lower = division.numerator;
    const std::int64_t denominator =
        static_cast<std::int64_t>(division.denominator);
    if (!accumulate(lower[localOffset + local], -denominator))
      return providerError("a frozen schedule division overflows");
    if (llvm::Error error = addRow(lower, false))
      return std::move(error);

    std::vector<std::int64_t> upper;
    upper.reserve(rowWidth);
    for (std::int64_t coefficient : division.numerator) {
      std::int64_t negated = 0;
      if (!negate(coefficient, negated))
        return providerError("a frozen schedule division overflows");
      upper.push_back(negated);
    }
    if (!accumulate(upper[localOffset + local], denominator) ||
        !accumulate(upper.back(), denominator - 1))
      return providerError("a frozen schedule division overflows");
    if (llvm::Error error = addRow(upper, false))
      return std::move(error);
  }

  IslMap result(isl_map_from_basic_map(reconstructed.release()));
  if (!result)
    return providerError("cannot construct a frozen ISL schedule piece");
  result.reset(
      isl_map_project_out(result.release(), isl_dim_out,
                          static_cast<unsigned>(piece.scheduleDimensionCount),
                          static_cast<unsigned>(piece.divisions.size())));
  if (!result)
    return providerError("cannot project frozen ISL schedule divisions");
  return result.release();
}

llvm::Error verifyFrozenScheduleMap(
    isl_map *original,
    const StructuredPolyhedralStatementScheduleView &schedule) {
  IslMap reconstructed;
  for (const StructuredPolyhedralSchedulePieceView &piece : schedule.pieces) {
    auto map = reconstructSchedulePiece(original, piece);
    if (!map)
      return map.takeError();
    if (!reconstructed) {
      reconstructed.reset(*map);
      continue;
    }
    reconstructed.reset(isl_map_union(reconstructed.release(), *map));
    if (!reconstructed)
      return providerError("cannot union reconstructed ISL schedule pieces");
  }
  if (!reconstructed)
    return providerError("a frozen statement schedule has no pieces");
  const isl_bool equal = isl_map_is_equal(original, reconstructed.get());
  if (equal < 0)
    return providerError("cannot compare a frozen ISL schedule map");
  if (equal == isl_bool_false)
    return providerError("a frozen ISL schedule map changed semantics");
  return llvm::Error::success();
}

llvm::Expected<isl_map *> reconstructStatementSchedule(
    isl_ctx *context, std::uint64_t parameterCount,
    const StructuredPolyhedralStatementScheduleView &schedule) {
  if (schedule.pieces.empty() ||
      parameterCount > std::numeric_limits<unsigned>::max() ||
      schedule.statementOrdinal > std::numeric_limits<unsigned>::max())
    return providerError("a frozen statement schedule has invalid arity");
  const StructuredPolyhedralSchedulePieceView &first = schedule.pieces.front();
  if (first.sourceDimensionCount > std::numeric_limits<unsigned>::max() ||
      first.scheduleDimensionCount > std::numeric_limits<unsigned>::max() ||
      first.parameterCount != parameterCount)
    return providerError("a frozen statement schedule has invalid dimensions");

  isl_space *rawSpace =
      isl_space_alloc(context, static_cast<unsigned>(parameterCount),
                      static_cast<unsigned>(first.sourceDimensionCount),
                      static_cast<unsigned>(first.scheduleDimensionCount));
  if (!rawSpace)
    return providerError("cannot allocate a frozen schedule space");
  IslOwner<isl_space, isl_space_free> space(rawSpace);
  for (unsigned index = 0; index != parameterCount; ++index) {
    const std::string name = parameterName(index);
    space.reset(isl_space_set_dim_name(space.release(), isl_dim_param, index,
                                       name.c_str()));
    if (!space)
      return providerError("cannot name a frozen schedule parameter");
  }
  const std::string statementName = tupleName(schedule.statementOrdinal);
  space.reset(isl_space_set_tuple_name(space.release(), isl_dim_in,
                                       statementName.c_str()));
  if (!space)
    return providerError("cannot name a frozen schedule statement");
  IslMap source(isl_map_universe(space.release()));
  if (!source)
    return providerError("cannot allocate a frozen schedule map");

  IslMap reconstructed;
  for (const StructuredPolyhedralSchedulePieceView &piece : schedule.pieces) {
    if (piece.sourceDimensionCount != first.sourceDimensionCount ||
        piece.scheduleDimensionCount != first.scheduleDimensionCount ||
        piece.parameterCount != parameterCount)
      return providerError("frozen schedule pieces disagree on their space");
    auto map = reconstructSchedulePiece(source.get(), piece);
    if (!map)
      return map.takeError();
    if (!reconstructed) {
      reconstructed.reset(*map);
      continue;
    }
    reconstructed.reset(isl_map_union(reconstructed.release(), *map));
    if (!reconstructed)
      return providerError("cannot union frozen statement schedule pieces");
  }
  return reconstructed.release();
}

llvm::Expected<isl_union_map *>
reconstructSchedule(isl_ctx *context,
                    const StructuredPolyhedralScheduleView &schedule) {
  if (schedule.statementSchedules.empty() ||
      schedule.parameterCount > std::numeric_limits<unsigned>::max())
    return providerError("a frozen schedule has invalid cardinality");
  IslUnionMap result(isl_union_map_empty_ctx(context));
  if (!result)
    return providerError("cannot allocate a frozen union schedule");
  std::vector<std::uint64_t> ordinals;
  ordinals.reserve(schedule.statementSchedules.size());
  std::optional<std::uint64_t> rangeDimensions;
  for (const StructuredPolyhedralStatementScheduleView &statement :
       schedule.statementSchedules) {
    if (llvm::is_contained(ordinals, statement.statementOrdinal))
      return providerError("a frozen schedule duplicates a statement");
    ordinals.push_back(statement.statementOrdinal);
    if (statement.pieces.empty())
      return providerError("a frozen statement schedule has no pieces");
    const std::uint64_t dimensions =
        statement.pieces.front().scheduleDimensionCount;
    if (rangeDimensions && *rangeDimensions != dimensions)
      return providerError("frozen statement schedules have unequal ranges");
    rangeDimensions = dimensions;
    auto map = reconstructStatementSchedule(context, schedule.parameterCount,
                                            statement);
    if (!map)
      return map.takeError();
    result.reset(isl_union_map_add_map(result.release(), *map));
    if (!result)
      return providerError("cannot extend the frozen union schedule");
  }
  const isl_bool singleValued = isl_union_map_is_single_valued(result.get());
  if (singleValued < 0)
    return providerError("cannot validate the frozen union schedule");
  if (singleValued == isl_bool_false)
    return providerError("a frozen schedule is not single-valued");
  return result.release();
}

llvm::Expected<isl_set *> reconstructStatementDomain(
    isl_ctx *context, std::uint64_t statementOrdinal,
    const StructuredPolyhedralSetView &domain,
    llvm::ArrayRef<StructuredEntityRef> globalParameters) {
  if (domain.dimensions.size() > std::numeric_limits<unsigned>::max() ||
      globalParameters.size() > std::numeric_limits<unsigned>::max())
    return providerError("a frozen statement domain exceeds ISL arity");
  isl_space *rawSpace = isl_space_set_alloc(
      context, static_cast<unsigned>(globalParameters.size()),
      static_cast<unsigned>(domain.dimensions.size()));
  if (!rawSpace)
    return providerError("cannot allocate a frozen statement domain space");
  IslOwner<isl_space, isl_space_free> space(rawSpace);
  for (unsigned index = 0; index != globalParameters.size(); ++index) {
    const std::string name = parameterName(index);
    space.reset(isl_space_set_dim_name(space.release(), isl_dim_param, index,
                                       name.c_str()));
    if (!space)
      return providerError("cannot name a frozen domain parameter");
  }
  const std::string statementName = tupleName(statementOrdinal);
  space.reset(isl_space_set_tuple_name(space.release(), isl_dim_set,
                                       statementName.c_str()));
  if (!space)
    return providerError("cannot name a frozen statement domain");
  IslBasicSet result(isl_basic_set_universe(space.release()));
  if (!result)
    return providerError("cannot allocate a frozen statement domain");

  const std::size_t expectedWidth =
      domain.dimensions.size() + domain.parameters.size() + 1;
  for (const StructuredPolyhedralConstraintView &row : domain.constraints) {
    if (row.coefficients.size() != expectedWidth)
      return providerError("a frozen domain row has inconsistent arity");
    IslConstraint constraint(
        row.kind == StructuredPolyhedralConstraintKind::Equality
            ? isl_constraint_alloc_equality(isl_local_space_from_space(
                  isl_basic_set_get_space(result.get())))
            : isl_constraint_alloc_inequality(isl_local_space_from_space(
                  isl_basic_set_get_space(result.get()))));
    if (!constraint)
      return providerError("cannot allocate a frozen domain constraint");
    for (unsigned index = 0; index != domain.dimensions.size(); ++index) {
      auto updated = setCoefficient(constraint.release(), isl_dim_set, index,
                                    row.coefficients[index]);
      if (!updated)
        return updated.takeError();
      constraint.reset(*updated);
    }
    for (unsigned index = 0; index != domain.parameters.size(); ++index) {
      auto global = llvm::find(globalParameters, domain.parameters[index]);
      if (global == globalParameters.end())
        return providerError(
            "a frozen domain parameter is absent from the global schedule");
      const std::size_t globalOrdinal = static_cast<std::size_t>(
          std::distance(globalParameters.begin(), global));
      if (globalOrdinal > std::numeric_limits<unsigned>::max())
        return providerError("a frozen domain parameter exceeds ISL arity");
      auto updated =
          setCoefficient(constraint.release(), isl_dim_param,
                         static_cast<unsigned>(globalOrdinal),
                         row.coefficients[domain.dimensions.size() + index]);
      if (!updated)
        return updated.takeError();
      constraint.reset(*updated);
    }
    auto updated = setConstant(constraint.release(), row.coefficients.back());
    if (!updated)
      return updated.takeError();
    result.reset(isl_basic_set_add_constraint(result.release(), *updated));
    if (!result)
      return providerError("cannot add a frozen statement-domain constraint");
  }
  return isl_set_from_basic_set(result.release());
}

llvm::Expected<isl_union_set *>
reconstructDomains(isl_ctx *context, const StructuredPolyhedralScopView &scop) {
  if (scop.parameters.size() != scop.schedule.parameterCount ||
      scop.statements.size() != scop.schedule.statementSchedules.size())
    return providerError("the frozen SCoP has inconsistent cardinality");
  IslUnionSet result(isl_union_set_empty_ctx(context));
  if (!result)
    return providerError("cannot allocate frozen statement domains");
  for (auto [ordinal, statement] : llvm::enumerate(scop.statements)) {
    auto domain = reconstructStatementDomain(context, ordinal, statement.domain,
                                             scop.parameters);
    if (!domain)
      return domain.takeError();
    result.reset(isl_union_set_add_set(result.release(), *domain));
    if (!result)
      return providerError("cannot extend frozen statement domains");
  }
  return result.release();
}

llvm::Expected<PolyhedralAstExpression>
convertAstExpression(isl_ast_expr *rawExpression) {
  IslAstExpression expression(rawExpression);
  if (!expression)
    return providerError("cannot read an ISL AST expression");
  switch (isl_ast_expr_get_type(expression.get())) {
  case isl_ast_expr_int: {
    auto integer =
        readAstInteger(IslVal(isl_ast_expr_get_val(expression.get())));
    if (!integer)
      return integer.takeError();
    if (!*integer)
      return llvm::make_error<AstRepresentationUnavailable>();
    return PolyhedralAstExpression{
        PolyhedralAstExpressionKind::Integer, **integer, {}, {}};
  }
  case isl_ast_expr_id: {
    IslId identifier(isl_ast_expr_get_id(expression.get()));
    const char *name = identifier ? isl_id_get_name(identifier.get()) : nullptr;
    if (!name)
      return providerError("an ISL AST identifier has no name");
    return PolyhedralAstExpression{
        PolyhedralAstExpressionKind::Identifier, 0, name, {}};
  }
  case isl_ast_expr_op:
    break;
  case isl_ast_expr_error:
    return providerError("an ISL AST expression is invalid");
  }

  const isl_ast_expr_op_type operation =
      isl_ast_expr_get_op_type(expression.get());
  PolyhedralAstExpressionKind kind;
  switch (operation) {
  case isl_ast_op_and:
    kind = PolyhedralAstExpressionKind::And;
    break;
  case isl_ast_op_and_then:
    kind = PolyhedralAstExpressionKind::AndThen;
    break;
  case isl_ast_op_or:
    kind = PolyhedralAstExpressionKind::Or;
    break;
  case isl_ast_op_or_else:
    kind = PolyhedralAstExpressionKind::OrElse;
    break;
  case isl_ast_op_max:
    kind = PolyhedralAstExpressionKind::Maximum;
    break;
  case isl_ast_op_min:
    kind = PolyhedralAstExpressionKind::Minimum;
    break;
  case isl_ast_op_minus:
    kind = PolyhedralAstExpressionKind::Negate;
    break;
  case isl_ast_op_add:
    kind = PolyhedralAstExpressionKind::Add;
    break;
  case isl_ast_op_sub:
    kind = PolyhedralAstExpressionKind::Subtract;
    break;
  case isl_ast_op_mul:
    kind = PolyhedralAstExpressionKind::Multiply;
    break;
  case isl_ast_op_div:
    kind = PolyhedralAstExpressionKind::Divide;
    break;
  case isl_ast_op_fdiv_q:
    kind = PolyhedralAstExpressionKind::FloorDivide;
    break;
  case isl_ast_op_pdiv_q:
    kind = PolyhedralAstExpressionKind::PositiveDivide;
    break;
  case isl_ast_op_pdiv_r:
    kind = PolyhedralAstExpressionKind::PositiveRemainder;
    break;
  case isl_ast_op_zdiv_r:
    kind = PolyhedralAstExpressionKind::ZeroRemainder;
    break;
  case isl_ast_op_cond:
    kind = PolyhedralAstExpressionKind::Conditional;
    break;
  case isl_ast_op_select:
    kind = PolyhedralAstExpressionKind::Select;
    break;
  case isl_ast_op_eq:
    kind = PolyhedralAstExpressionKind::Equal;
    break;
  case isl_ast_op_le:
    kind = PolyhedralAstExpressionKind::LessEqual;
    break;
  case isl_ast_op_lt:
    kind = PolyhedralAstExpressionKind::Less;
    break;
  case isl_ast_op_ge:
    kind = PolyhedralAstExpressionKind::GreaterEqual;
    break;
  case isl_ast_op_gt:
    kind = PolyhedralAstExpressionKind::Greater;
    break;
  case isl_ast_op_call:
    kind = PolyhedralAstExpressionKind::Call;
    break;
  case isl_ast_op_access:
  case isl_ast_op_member:
  case isl_ast_op_address_of:
    return llvm::make_error<AstRepresentationUnavailable>();
  case isl_ast_op_error:
    return providerError("an ISL AST operation is invalid");
  }
  const isl_size operandCount = isl_ast_expr_get_op_n_arg(expression.get());
  if (operandCount < 0)
    return providerError("cannot read ISL AST expression operands");
  std::vector<PolyhedralAstExpression> operands;
  operands.reserve(static_cast<std::size_t>(operandCount));
  for (isl_size index = 0; index != operandCount; ++index) {
    auto operand = convertAstExpression(
        isl_ast_expr_get_op_arg(expression.get(), static_cast<int>(index)));
    if (!operand)
      return operand.takeError();
    operands.push_back(std::move(*operand));
  }
  return PolyhedralAstExpression{kind, 0, {}, std::move(operands)};
}

llvm::Expected<PolyhedralAstNode> convertAstNode(isl_ast_node *rawNode) {
  IslAstNode node(rawNode);
  if (!node)
    return providerError("cannot read an ISL AST node");
  switch (isl_ast_node_get_type(node.get())) {
  case isl_ast_node_for: {
    auto iteratorExpression =
        convertAstExpression(isl_ast_node_for_get_iterator(node.get()));
    if (!iteratorExpression)
      return iteratorExpression.takeError();
    if (iteratorExpression->kind != PolyhedralAstExpressionKind::Identifier ||
        iteratorExpression->identifier.empty())
      return providerError("an ISL AST loop has no iterator identifier");
    auto initial = convertAstExpression(isl_ast_node_for_get_init(node.get()));
    auto condition =
        convertAstExpression(isl_ast_node_for_get_cond(node.get()));
    auto increment = convertAstExpression(isl_ast_node_for_get_inc(node.get()));
    auto body = convertAstNode(isl_ast_node_for_get_body(node.get()));
    if (!initial)
      return initial.takeError();
    if (!condition)
      return condition.takeError();
    if (!increment)
      return increment.takeError();
    if (!body)
      return body.takeError();
    return PolyhedralAstNode{PolyhedralAstFor{
        std::move(iteratorExpression->identifier), std::move(*initial),
        std::move(*condition), std::move(*increment),
        std::make_unique<PolyhedralAstNode>(std::move(*body))}};
  }
  case isl_ast_node_if: {
    auto condition = convertAstExpression(isl_ast_node_if_get_cond(node.get()));
    auto thenNode = convertAstNode(isl_ast_node_if_get_then(node.get()));
    if (!condition)
      return condition.takeError();
    if (!thenNode)
      return thenNode.takeError();
    std::unique_ptr<PolyhedralAstNode> elseNode;
    const isl_bool hasElse = isl_ast_node_if_has_else(node.get());
    if (hasElse < 0)
      return providerError("cannot inspect an ISL AST conditional");
    if (hasElse == isl_bool_true) {
      auto convertedElse = convertAstNode(isl_ast_node_if_get_else(node.get()));
      if (!convertedElse)
        return convertedElse.takeError();
      elseNode = std::make_unique<PolyhedralAstNode>(std::move(*convertedElse));
    }
    return PolyhedralAstNode{PolyhedralAstIf{
        std::move(*condition),
        std::make_unique<PolyhedralAstNode>(std::move(*thenNode)),
        std::move(elseNode)}};
  }
  case isl_ast_node_block: {
    IslAstNodeList children(isl_ast_node_block_get_children(node.get()));
    const isl_size childCount =
        children ? isl_ast_node_list_n_ast_node(children.get()) : -1;
    if (childCount < 0)
      return providerError("cannot inspect an ISL AST block");
    PolyhedralAstBlock block;
    block.children.reserve(static_cast<std::size_t>(childCount));
    for (isl_size index = 0; index != childCount; ++index) {
      auto child = convertAstNode(isl_ast_node_list_get_ast_node(
          children.get(), static_cast<int>(index)));
      if (!child)
        return child.takeError();
      block.children.push_back(std::move(*child));
    }
    return PolyhedralAstNode{std::move(block)};
  }
  case isl_ast_node_mark:
    return convertAstNode(isl_ast_node_mark_get_node(node.get()));
  case isl_ast_node_user: {
    auto call = convertAstExpression(isl_ast_node_user_get_expr(node.get()));
    if (!call)
      return call.takeError();
    if (call->kind != PolyhedralAstExpressionKind::Call)
      return providerError("an ISL AST user node is not a statement call");
    return PolyhedralAstNode{PolyhedralAstUser{std::move(*call)}};
  }
  case isl_ast_node_error:
    return providerError("an ISL AST node is invalid");
  }
  return providerError("an ISL AST node has an unknown kind");
}

struct BasicMapFreezeContext final {
  StructuredPolyhedralStatementScheduleView *statement = nullptr;
  std::uint64_t parameterCount = 0;
  bool failed = false;
};

isl_stat freezeBasicMap(isl_basic_map *rawBasic, void *opaque) {
  IslBasicMap basic(rawBasic);
  auto &context = *static_cast<BasicMapFreezeContext *>(opaque);
  const isl_size sourceDimensions = isl_basic_map_dim(basic.get(), isl_dim_in);
  const isl_size scheduleDimensions =
      isl_basic_map_dim(basic.get(), isl_dim_out);
  const isl_size parameters = isl_basic_map_dim(basic.get(), isl_dim_param);
  if (sourceDimensions < 0 || scheduleDimensions < 0 || parameters < 0 ||
      static_cast<std::uint64_t>(parameters) != context.parameterCount) {
    context.failed = true;
    return isl_stat_error;
  }
  StructuredPolyhedralSchedulePieceView piece{
      static_cast<std::uint64_t>(sourceDimensions),
      static_cast<std::uint64_t>(scheduleDimensions),
      context.parameterCount,
      {},
      {}};
  if (!freezeDivisions(basic.get(), piece)) {
    context.failed = true;
    return isl_stat_error;
  }
  ConstraintFreezeContext constraintContext{&piece, false};
  if (isl_basic_map_foreach_constraint(basic.get(), freezeConstraint,
                                       &constraintContext) < 0 ||
      constraintContext.failed) {
    context.failed = true;
    return isl_stat_error;
  }
  llvm::sort(piece.constraints, constraintLess);
  context.statement->pieces.push_back(std::move(piece));
  return isl_stat_ok;
}

struct ScheduleFreezeContext final {
  const std::map<std::string, std::uint64_t> *statementOrdinals = nullptr;
  const llvm::DenseMap<std::uint64_t, unsigned> *statementDimensions = nullptr;
  std::uint64_t parameterCount = 0;
  std::vector<StructuredPolyhedralStatementScheduleView> schedules;
  bool failed = false;
};

isl_stat freezeScheduleMap(isl_map *rawMap, void *opaque) {
  IslMap map(rawMap);
  auto &context = *static_cast<ScheduleFreezeContext *>(opaque);
  const char *tuple = isl_map_get_tuple_name(map.get(), isl_dim_in);
  if (!tuple) {
    context.failed = true;
    return isl_stat_error;
  }
  auto ordinal = context.statementOrdinals->find(tuple);
  if (ordinal == context.statementOrdinals->end() ||
      llvm::any_of(context.schedules, [&](const auto &schedule) {
        return schedule.statementOrdinal == ordinal->second;
      })) {
    context.failed = true;
    return isl_stat_error;
  }
  auto dimensions = context.statementDimensions->find(ordinal->second);
  const isl_size sourceDimensions = isl_map_dim(map.get(), isl_dim_in);
  if (dimensions == context.statementDimensions->end() ||
      sourceDimensions < 0 ||
      static_cast<unsigned>(sourceDimensions) != dimensions->second) {
    context.failed = true;
    return isl_stat_error;
  }
  StructuredPolyhedralStatementScheduleView schedule{ordinal->second, {}};
  BasicMapFreezeContext basicContext{&schedule, context.parameterCount, false};
  if (isl_map_foreach_basic_map(map.get(), freezeBasicMap, &basicContext) < 0 ||
      basicContext.failed || schedule.pieces.empty()) {
    context.failed = true;
    return isl_stat_error;
  }
  llvm::sort(schedule.pieces, pieceLess);
  if (llvm::Error error = verifyFrozenScheduleMap(map.get(), schedule)) {
    llvm::consumeError(std::move(error));
    context.failed = true;
    return isl_stat_error;
  }
  context.schedules.push_back(std::move(schedule));
  return isl_stat_ok;
}

bool matchesCanonicalSchedulePiece(
    const StructuredPolyhedralStatementScheduleView &statement,
    bool statementMajor, bool adjacentInterchange) {
  if (statement.pieces.size() != 1)
    return false;
  const StructuredPolyhedralSchedulePieceView &piece = statement.pieces.front();
  if (piece.sourceDimensionCount == 0 || !piece.divisions.empty() ||
      piece.scheduleDimensionCount != piece.sourceDimensionCount + 1 ||
      (adjacentInterchange && piece.sourceDimensionCount < 2) ||
      statement.statementOrdinal >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return false;
  const std::size_t width = static_cast<std::size_t>(
      piece.sourceDimensionCount + piece.scheduleDimensionCount +
      piece.parameterCount + 1);
  std::vector<StructuredPolyhedralConstraintView> expected;
  expected.reserve(static_cast<std::size_t>(piece.sourceDimensionCount + 1));
  for (std::uint64_t position = 0; position != piece.sourceDimensionCount;
       ++position) {
    std::vector<std::int64_t> row(width, 0);
    const std::uint64_t source =
        adjacentInterchange && position < 2 ? 1 - position : position;
    const std::uint64_t destination = statementMajor ? position + 1 : position;
    row[static_cast<std::size_t>(source)] = -1;
    row[static_cast<std::size_t>(piece.sourceDimensionCount + destination)] = 1;
    expected.push_back(
        {StructuredPolyhedralConstraintKind::Equality, std::move(row)});
  }
  std::vector<std::int64_t> statementRow(width, 0);
  const std::uint64_t statementPosition =
      statementMajor ? 0 : piece.sourceDimensionCount;
  statementRow[static_cast<std::size_t>(piece.sourceDimensionCount +
                                        statementPosition)] = 1;
  statementRow.back() = -static_cast<std::int64_t>(statement.statementOrdinal);
  expected.push_back(
      {StructuredPolyhedralConstraintKind::Equality, std::move(statementRow)});
  llvm::sort(expected, constraintLess);
  return expected == piece.constraints;
}

StructuredPolyhedralScheduleForm classifyFrozenScheduleForm(
    llvm::ArrayRef<StructuredPolyhedralStatementScheduleView> schedules) {
  struct CanonicalForm final {
    StructuredPolyhedralScheduleForm form;
    bool statementMajor = false;
    bool adjacentInterchange = false;
  };
  constexpr std::array<CanonicalForm, 4> forms = {{
      {StructuredPolyhedralScheduleForm::SourceOrder, false, false},
      {StructuredPolyhedralScheduleForm::AdjacentInterchange, false, true},
      {StructuredPolyhedralScheduleForm::StatementMajor, true, false},
      {StructuredPolyhedralScheduleForm::StatementMajorAdjacentInterchange,
       true, true},
  }};
  for (const CanonicalForm &candidate : forms)
    if (!schedules.empty() &&
        llvm::all_of(schedules, [&](const auto &schedule) {
          return matchesCanonicalSchedulePiece(schedule,
                                               candidate.statementMajor,
                                               candidate.adjacentInterchange);
        }))
      return candidate.form;
  return StructuredPolyhedralScheduleForm::General;
}

isl_bool summarizeBand(isl_schedule_node *node, void *opaque) {
  auto &summary = *static_cast<BandSummary *>(opaque);
  if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
    return isl_bool_true;
  const isl_size members = isl_schedule_node_band_n_member(node);
  if (members < 0) {
    summary.failed = true;
    return isl_bool_error;
  }
  ++summary.bands;
  summary.dimensions += static_cast<std::uint64_t>(members);
  for (isl_size index = 0; index != members; ++index) {
    const isl_bool coincident =
        isl_schedule_node_band_member_get_coincident(node, index);
    if (coincident < 0) {
      summary.failed = true;
      return isl_bool_error;
    }
    summary.coincidentDimensions += coincident == isl_bool_true;
  }
  return isl_bool_true;
}

} // namespace

llvm::Expected<PolyhedralScheduleProviderOutcome> computePinnedIslSchedule(
    llvm::ArrayRef<PolyhedralStatementDomain> statements,
    llvm::ArrayRef<PolyhedralDependenceRelation> dependences) {
  if (statements.empty() || statements.size() > maximumPinnedIslStatementCount)
    return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
  std::uint64_t constraintCount = 0;
  ParameterTable parameters;
  llvm::DenseMap<std::uint64_t,
                 const mlir::affine::FlatAffineValueConstraints *>
      statementDomains;
  for (const PolyhedralStatementDomain &statement : statements) {
    if (!statement.domain)
      return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
    if (!statementDomains
             .try_emplace(statement.statementOrdinal, statement.domain)
             .second)
      return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
    constraintCount += statement.domain->getNumConstraints();
    if (constraintCount > maximumProviderConstraints)
      return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
    if (auto refusal = collectParameters(*statement.domain, parameters))
      return *refusal;
  }
  const auto dimensionsMatch =
      [](const mlir::affine::FlatAffineValueConstraints &domain,
         const mlir::affine::FlatAffineValueConstraints &relation,
         unsigned relationOffset) {
        for (unsigned index = 0; index != domain.getNumDimVars(); ++index) {
          if (!domain.hasValue(index) ||
              !relation.hasValue(relationOffset + index) ||
              domain.getValue(index) !=
                  relation.getValue(relationOffset + index))
            return false;
        }
        return true;
      };
  const auto commonDimensionPrefixMatches =
      [](const mlir::affine::FlatAffineValueConstraints &source,
         const mlir::affine::FlatAffineValueConstraints &destination) {
        const unsigned common =
            std::min(source.getNumDimVars(), destination.getNumDimVars());
        for (unsigned index = 0; index != common; ++index)
          if (!source.hasValue(index) || !destination.hasValue(index) ||
              source.getValue(index) != destination.getValue(index))
            return false;
        return true;
      };
  for (const PolyhedralDependenceRelation &dependence : dependences) {
    const auto source =
        statementDomains.find(dependence.sourceStatementOrdinal);
    const auto destination =
        statementDomains.find(dependence.destinationStatementOrdinal);
    if (source == statementDomains.end() ||
        destination == statementDomains.end() ||
        dependence.sourceDimensionCount != source->second->getNumDimVars() ||
        dependence.destinationDimensionCount !=
            destination->second->getNumDimVars() ||
        (dependence.relation && dependence.relation->getNumDimVars() !=
                                    dependence.sourceDimensionCount +
                                        dependence.destinationDimensionCount) ||
        (dependence.relation &&
         (!dimensionsMatch(*source->second, *dependence.relation, 0) ||
          !dimensionsMatch(*destination->second, *dependence.relation,
                           dependence.sourceDimensionCount))) ||
        (!dependence.relation &&
         !commonDimensionPrefixMatches(*source->second, *destination->second)))
      return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
    constraintCount +=
        dependence.relation ? dependence.relation->getNumConstraints() : 0;
    if (constraintCount > maximumProviderConstraints)
      return PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted;
    if (dependence.relation)
      if (auto refusal = collectParameters(*dependence.relation, parameters))
        return *refusal;
  }

  IslContext context(isl_ctx_alloc());
  if (!context)
    return providerError("cannot allocate the pinned ISL context");
  const auto providerFailure = [&](const llvm::Twine &message)
      -> llvm::Expected<PolyhedralScheduleProviderOutcome> {
    if (isl_ctx_last_error(context.get()) == isl_error_quota)
      return PolyhedralScheduleProviderRefusalKind::OperationBudgetExhausted;
    return providerError(message);
  };
  if (isl_options_set_on_error(context.get(), ISL_ON_ERROR_CONTINUE) < 0)
    return providerFailure("cannot configure ISL error handling");
  isl_ctx_set_max_operations(context.get(), maximumIslOperations);
  if (isl_options_set_schedule_max_coefficient(context.get(), 64) < 0 ||
      isl_options_set_schedule_max_constant_term(context.get(), 64) < 0)
    return providerFailure("cannot configure bounded ISL scheduling");

  IslUnionSet domain(isl_union_set_empty_ctx(context.get()));
  if (!domain)
    return providerFailure("cannot allocate the ISL union domain");
  for (const PolyhedralStatementDomain &statement : statements) {
    auto translated = translateDomain(context.get(), statement, parameters);
    if (!translated) {
      if (isl_ctx_last_error(context.get()) == isl_error_quota) {
        llvm::consumeError(translated.takeError());
        return PolyhedralScheduleProviderRefusalKind::OperationBudgetExhausted;
      }
      return translated.takeError();
    }
    domain.reset(isl_union_set_add_set(domain.release(), *translated));
    if (!domain)
      return providerFailure("cannot extend the ISL union domain");
  }

  IslUnionMap validity(isl_union_map_empty_ctx(context.get()));
  IslUnionMap scalarProximity(isl_union_map_empty_ctx(context.get()));
  if (!validity || !scalarProximity)
    return providerFailure("cannot allocate the ISL dependence relations");
  for (const PolyhedralDependenceRelation &dependence : dependences) {
    auto translated =
        translateDependence(context.get(), dependence, parameters);
    if (!translated) {
      if (isl_ctx_last_error(context.get()) == isl_error_quota) {
        llvm::consumeError(translated.takeError());
        return PolyhedralScheduleProviderRefusalKind::OperationBudgetExhausted;
      }
      return translated.takeError();
    }
    if (!dependence.relation) {
      scalarProximity.reset(isl_union_map_add_map(scalarProximity.release(),
                                                  isl_map_copy(*translated)));
      if (!scalarProximity)
        return providerFailure("cannot extend scalar schedule proximity");
    }
    validity.reset(isl_union_map_add_map(validity.release(), *translated));
    if (!validity)
      return providerFailure("cannot extend the ISL validity relation");
  }
  validity.reset(isl_union_map_intersect_domain_union_set(
      validity.release(), isl_union_set_copy(domain.get())));
  if (!validity)
    return providerFailure("cannot restrict validity to statement domains");
  validity.reset(isl_union_map_intersect_range_union_set(
      validity.release(), isl_union_set_copy(domain.get())));
  if (!validity)
    return providerFailure("cannot restrict validity to destination domains");
  scalarProximity.reset(isl_union_map_intersect_domain_union_set(
      scalarProximity.release(), isl_union_set_copy(domain.get())));
  if (!scalarProximity)
    return providerFailure("cannot restrict scalar proximity domains");
  scalarProximity.reset(isl_union_map_intersect_range_union_set(
      scalarProximity.release(), isl_union_set_copy(domain.get())));
  if (!scalarProximity)
    return providerFailure("cannot restrict scalar proximity ranges");

  IslScheduleConstraints constraints(
      isl_schedule_constraints_on_domain(isl_union_set_copy(domain.get())));
  if (!constraints)
    return providerFailure("cannot allocate ISL schedule constraints");
  constraints.reset(isl_schedule_constraints_set_validity(
      constraints.release(), isl_union_map_copy(validity.get())));
  if (!constraints)
    return providerFailure("cannot attach exact validity relations");
  constraints.reset(isl_schedule_constraints_set_proximity(
      constraints.release(), scalarProximity.release()));
  if (!constraints)
    return providerFailure("cannot attach scalar schedule proximity");
  IslSchedule schedule(
      isl_schedule_constraints_compute_schedule(constraints.release()));
  if (!schedule) {
    const enum isl_error error = isl_ctx_last_error(context.get());
    if (error == isl_error_quota)
      return PolyhedralScheduleProviderRefusalKind::OperationBudgetExhausted;
    if (error == isl_error_none || error == isl_error_unknown)
      return PolyhedralScheduleProviderRefusalKind::ScheduleNotEstablished;
    return providerError("the pinned ISL scheduler failed internally");
  }

  IslUnionSet scheduledDomain(isl_schedule_get_domain(schedule.get()));
  if (!scheduledDomain)
    return providerFailure("cannot read the ISL schedule domain");
  const isl_bool equal =
      isl_union_set_is_equal(domain.get(), scheduledDomain.get());
  if (equal < 0)
    return providerFailure("cannot compare the ISL schedule domain");
  if (equal == isl_bool_false)
    return providerError("the ISL schedule changed the exact statement domain");

  IslUnionMap scheduleMap(isl_schedule_get_map(schedule.get()));
  if (!scheduleMap)
    return providerFailure("the ISL schedule has no schedule map");
  if (!dependences.empty()) {
    IslUnionMap ordered(
        isl_union_map_lex_lt_union_map(isl_union_map_copy(scheduleMap.get()),
                                       isl_union_map_copy(scheduleMap.get())));
    if (!ordered)
      return providerFailure("cannot construct the ISL schedule order");
    const isl_bool respects =
        isl_union_map_is_subset(validity.get(), ordered.get());
    if (respects < 0)
      return providerFailure("cannot verify the ISL schedule order");
    if (respects == isl_bool_false)
      return providerError("the ISL schedule violates an exact dependence");
  }

  BandSummary bands;
  if (isl_schedule_foreach_schedule_node_top_down(schedule.get(), summarizeBand,
                                                  &bands) < 0 ||
      bands.failed)
    return providerFailure("cannot inspect the ISL schedule bands");
  if (bands.bands == 0 || bands.dimensions == 0)
    return PolyhedralScheduleProviderRefusalKind::ScheduleNotEstablished;

  std::map<std::string, std::uint64_t> statementOrdinals;
  llvm::DenseMap<std::uint64_t, unsigned> statementDimensions;
  for (const PolyhedralStatementDomain &statement : statements) {
    statementOrdinals.try_emplace(tupleName(statement.statementOrdinal),
                                  statement.statementOrdinal);
    statementDimensions.try_emplace(statement.statementOrdinal,
                                    statement.domain->getNumDimVars());
  }
  ScheduleFreezeContext frozen{&statementOrdinals,
                               &statementDimensions,
                               parameters.values.size(),
                               {},
                               false};
  if (isl_union_map_foreach_map(scheduleMap.get(), freezeScheduleMap, &frozen) <
          0 ||
      frozen.failed)
    return providerFailure("cannot freeze the exact ISL schedule maps");
  llvm::sort(frozen.schedules, [](const auto &lhs, const auto &rhs) {
    return lhs.statementOrdinal < rhs.statementOrdinal;
  });
  if (frozen.schedules.size() != statements.size())
    return providerError("the frozen ISL schedule lost a statement map");
  const StructuredPolyhedralScheduleForm form =
      classifyFrozenScheduleForm(frozen.schedules);

  return PolyhedralScheduleProviderOutcome(PolyhedralScheduleProviderView{
      parameters.values.size(), form, bands.bands, bands.dimensions,
      bands.coincidentDimensions, std::move(frozen.schedules),
      std::move(parameters.values)});
}

llvm::Expected<PolyhedralAstBuildOutcome>
buildPinnedIslAst(const StructuredPolyhedralScopView &scop) {
  const StructuredPolyhedralScheduleView &schedule = scop.schedule;
  if (schedule.provider != StructuredPolyhedralProviderKind::PinnedPollyIsl)
    return providerError("the frozen schedule names an unknown provider");
  IslContext context(isl_ctx_alloc());
  if (!context)
    return providerError("cannot allocate the pinned ISL AST context");
  const auto astFailure = [&](const llvm::Twine &message)
      -> llvm::Expected<PolyhedralAstBuildOutcome> {
    if (isl_ctx_last_error(context.get()) == isl_error_quota)
      return StructuredScopRefusalKind::ProviderScheduleBudgetExhausted;
    return providerError(message);
  };
  const auto propagateAstFailure =
      [&](llvm::Error error) -> llvm::Expected<PolyhedralAstBuildOutcome> {
    if (isl_ctx_last_error(context.get()) == isl_error_quota) {
      llvm::consumeError(std::move(error));
      return StructuredScopRefusalKind::ProviderScheduleBudgetExhausted;
    }
    bool unavailable = false;
    llvm::Error remaining = llvm::handleErrors(
        std::move(error),
        [&](const AstRepresentationUnavailable &) { unavailable = true; });
    if (remaining)
      return std::move(remaining);
    if (unavailable)
      return StructuredScopRefusalKind::PolyhedralMaterializationUnavailable;
    return providerError("ISL AST conversion failed without an error");
  };
  if (isl_options_set_on_error(context.get(), ISL_ON_ERROR_CONTINUE) < 0)
    return astFailure("cannot configure ISL AST error handling");
  isl_ctx_set_max_operations(context.get(), maximumIslOperations);
  if (isl_options_set_ast_build_atomic_upper_bound(context.get(), 1) < 0 ||
      isl_options_set_ast_build_prefer_pdiv(context.get(), 1) < 0 ||
      isl_options_set_ast_build_detect_min_max(context.get(), 1) < 0 ||
      isl_options_set_ast_build_exploit_nested_bounds(context.get(), 1) < 0 ||
      isl_options_set_ast_build_group_coscheduled(context.get(), 1) < 0 ||
      isl_options_set_ast_build_separation_bounds(
          context.get(), ISL_AST_BUILD_SEPARATION_BOUNDS_EXPLICIT) < 0 ||
      isl_options_set_ast_build_allow_else(context.get(), 1) < 0 ||
      isl_options_set_ast_build_allow_or(context.get(), 0) < 0 ||
      isl_options_set_ast_build_scale_strides(context.get(), 1) < 0)
    return astFailure("cannot configure deterministic ISL AST generation");
  auto reconstructed = reconstructSchedule(context.get(), schedule);
  if (!reconstructed)
    return propagateAstFailure(reconstructed.takeError());
  IslUnionMap scheduleMap(*reconstructed);
  auto reconstructedDomains = reconstructDomains(context.get(), scop);
  if (!reconstructedDomains)
    return propagateAstFailure(reconstructedDomains.takeError());
  IslUnionSet domains(*reconstructedDomains);
  scheduleMap.reset(isl_union_map_intersect_domain_union_set(
      scheduleMap.release(), isl_union_set_copy(domains.get())));
  if (!scheduleMap)
    return astFailure("cannot restrict the frozen schedule to its domain");
  const isl_bool singleValued =
      isl_union_map_is_single_valued(scheduleMap.get());
  if (singleValued < 0)
    return astFailure("cannot check the reconstructed schedule function");
  if (singleValued == isl_bool_false)
    return providerError("the frozen schedule is not single-valued");
  IslUnionSet scheduledDomain(
      isl_union_map_domain(isl_union_map_copy(scheduleMap.get())));
  if (!scheduledDomain)
    return astFailure("cannot read the reconstructed schedule domain");
  const isl_bool exactDomain =
      isl_union_set_is_equal(domains.get(), scheduledDomain.get());
  if (exactDomain < 0)
    return astFailure("cannot compare the reconstructed schedule domain");
  if (exactDomain == isl_bool_false)
    return providerError("the frozen schedule does not cover its exact domain");

  if (schedule.parameterCount > std::numeric_limits<unsigned>::max())
    return providerError("the frozen schedule exceeds ISL parameter arity");
  IslOwner<isl_space, isl_space_free> parameterSpace(isl_space_params_alloc(
      context.get(), static_cast<unsigned>(schedule.parameterCount)));
  if (!parameterSpace)
    return astFailure("cannot allocate the ISL AST parameter space");
  for (unsigned index = 0; index != schedule.parameterCount; ++index) {
    const std::string name = parameterName(index);
    parameterSpace.reset(isl_space_set_dim_name(
        parameterSpace.release(), isl_dim_param, index, name.c_str()));
    if (!parameterSpace)
      return astFailure("cannot name an ISL AST parameter");
  }
  IslSet parameterContext(isl_set_universe(parameterSpace.release()));
  if (!parameterContext)
    return astFailure("cannot allocate the ISL AST parameter context");
  IslAstBuild build(isl_ast_build_from_context(parameterContext.release()));
  if (!build)
    return astFailure("cannot allocate the ISL AST builder");
  IslAstNode ast(
      isl_ast_build_node_from_schedule_map(build.get(), scheduleMap.release()));
  if (!ast) {
    if (isl_ctx_last_error(context.get()) == isl_error_quota)
      return StructuredScopRefusalKind::ProviderScheduleBudgetExhausted;
    return providerError("cannot derive an AST from the frozen schedule");
  }
  auto converted = convertAstNode(ast.release());
  if (!converted)
    return propagateAstFailure(converted.takeError());
  return PolyhedralAstBuildOutcome(std::move(*converted));
}

} // namespace loom::frontend::detail
