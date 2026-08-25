#include "StructuredPolyhedralProvider.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <isl/constraint.h>
#include <isl/ctx.h>
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
#include <cstdint>
#include <limits>
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

llvm::Error providerError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "polly_isl_provider_error: " + message);
}

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
                           dependence.sourceDimensionCount))))
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
  if (!validity)
    return providerFailure("cannot allocate the ISL validity relation");
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

  IslScheduleConstraints constraints(
      isl_schedule_constraints_on_domain(isl_union_set_copy(domain.get())));
  if (!constraints)
    return providerFailure("cannot allocate ISL schedule constraints");
  constraints.reset(isl_schedule_constraints_set_validity(
      constraints.release(), isl_union_map_copy(validity.get())));
  if (!constraints)
    return providerFailure("cannot attach exact validity relations");
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
  const isl_size mapCount = isl_union_map_n_map(scheduleMap.get());
  if (mapCount < 0)
    return providerFailure("cannot count the ISL schedule maps");
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

  return PolyhedralScheduleProviderOutcome(PolyhedralScheduleProviderView{
      parameters.values.size(), static_cast<std::uint64_t>(mapCount),
      bands.bands, bands.dimensions, bands.coincidentDimensions});
}

} // namespace loom::frontend::detail
