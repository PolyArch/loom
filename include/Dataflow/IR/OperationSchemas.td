//===- OperationSchemas.td - Canonical operation and family authority -----===//
//
// The one definition source of Loom's canonical operation authority. It owns
// exactly three related declarations:
//
//   * the closed typed semantic vocabulary a canonical actor projection may
//     use;
//   * every operation admitted as a Canonical Dataflow actor, with its stable
//     `OperationSchemaId`, the operation class that owns its source semantics,
//     its actor kind, and its semantic case; and
//   * the normative implementation-family registry, whose members are typed
//     references to the schemas declared above.
//
// One generator turns this source into the `OperationSchemaId` rows and the
// `ImplementationFamilyId` rows with their four-field descriptors. There is no
// second schema table, member list, family-shape switch, or operation-name
// table anywhere in the tree.
//
// A schema row states only the facts the canonical projection owns. The
// operation's own definition remains the owner of its source semantics and of
// its stable textual spelling, which is read from `OpClass::getOperationName()`
// rather than restated here. Physical ports, sharing, state, and timing are
// never schema facts.
//
// A family row states exactly one fact about hardware: its member schemas can
// be implemented by one real shared datapath family. It also selects the one
// closed typed `hw_params` record schema a concrete resource of that family
// may carry and the one typed admission rule that decides whether an exact
// actor point fits the family's physical organization. A family defines no
// semantics, carries no name field of its own, and never changes a schema's
// identity: the record name is both the C++ enumerator and the one keyword.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_OPERATION_SCHEMAS_TD
#define LOOM_OPERATION_SCHEMAS_TD

include "Dataflow/IR/OperationSchemaBase.td"

//===----------------------------------------------------------------------===//
// Closed typed semantic vocabulary
//===----------------------------------------------------------------------===//

// A case exists only when an actor carries an identity-critical semantic fact
// that its exact function type does not already state. Vector lane counts and
// element types, for example, are recovered from the type and therefore need
// no case of their own.
def NoSemanticPayload : SemanticsCase<0>;
def ArithFloatingPoint : SemanticsCase<1>;
def ArithIntegerOverflow : SemanticsCase<2>;
def ArithIntegerCompare : SemanticsCase<3>;
def ArithFloatCompare : SemanticsCase<4>;
def TypedConstantValue : SemanticsCase<5>;
def StreamRecurrence : SemanticsCase<6>;
def MemoryContract : SemanticsCase<7>;
def LLVMZeroPoison : SemanticsCase<8>;
def LLVMIntegerMinPoison : SemanticsCase<9>;
def LLVMAggregatePosition : SemanticsCase<10>;
def ArithExact : SemanticsCase<11>;
def ArithNonNegative : SemanticsCase<12>;
def VectorStaticPosition : SemanticsCase<13>;
def VectorShuffleMask : SemanticsCase<14>;

//===----------------------------------------------------------------------===//
// Closed typed capability-parameter record schemas
//===----------------------------------------------------------------------===//

def ScalarIntegerParams : CapabilityParamsSchema<0>;
def ScalarIntegerCompareMinMaxParams : CapabilityParamsSchema<1>;
def ScalarValueSelectParams : CapabilityParamsSchema<2>;
def ScalarIntegerCastParams : CapabilityParamsSchema<3>;
def ScalarBitReinterpretParams : CapabilityParamsSchema<4>;
def ScalarFloatParams : CapabilityParamsSchema<5>;
def ScalarFloatCompareMinMaxParams : CapabilityParamsSchema<6>;
def ScalarFloatWidthCastParams : CapabilityParamsSchema<7>;
def ScalarIntegerFloatConversionParams : CapabilityParamsSchema<8>;
def LoopStreamParams : CapabilityParamsSchema<9>;
def TokenPlaneParams : CapabilityParamsSchema<10>;

//===----------------------------------------------------------------------===//
// Closed typed admission providers
//===----------------------------------------------------------------------===//

def ScalarOrdinaryIntegerAdmission : TypedAdmissionProvider<0>;
def ScalarLogicIntegerAdmission : TypedAdmissionProvider<1>;
def ScalarIntegerCompareAdmission : TypedAdmissionProvider<2>;
def ScalarIntegerCastAdmission : TypedAdmissionProvider<3>;
def ScalarBitReinterpretAdmission : TypedAdmissionProvider<4>;
def ScalarValueSelectAdmission : TypedAdmissionProvider<5>;
def ScalarUniformFloatAdmission : TypedAdmissionProvider<6>;
def ScalarFloatCompareAdmission : TypedAdmissionProvider<7>;
def ScalarFloatCastAdmission : TypedAdmissionProvider<8>;
def ScalarIntegerFloatConversionAdmission : TypedAdmissionProvider<9>;
def StreamAdmission : TypedAdmissionProvider<10>;
def TokenPlaneAdmission : TypedAdmissionProvider<11>;

//===----------------------------------------------------------------------===//
// Standard integer, floating-point, and cast compute schemas
//===----------------------------------------------------------------------===//

def ArithConstant : ActorSchema<0, "::mlir::arith::ConstantOp", "Compute",
    TypedConstantValue>;
def ArithAddI : ActorSchema<1, "::mlir::arith::AddIOp", "Compute",
    ArithIntegerOverflow>;
def ArithSubI : ActorSchema<2, "::mlir::arith::SubIOp", "Compute",
    ArithIntegerOverflow>;
def ArithMulI : ActorSchema<3, "::mlir::arith::MulIOp", "Compute",
    ArithIntegerOverflow>;
def ArithShLI : ActorSchema<4, "::mlir::arith::ShLIOp", "Compute",
    ArithIntegerOverflow>;
def ArithShRSI : ActorSchema<5, "::mlir::arith::ShRSIOp", "Compute",
    ArithExact>;
def ArithShRUI : ActorSchema<6, "::mlir::arith::ShRUIOp", "Compute",
    ArithExact>;
def ArithAndI : ActorSchema<7, "::mlir::arith::AndIOp", "Compute",
    NoSemanticPayload>;
def ArithOrI : ActorSchema<8, "::mlir::arith::OrIOp", "Compute",
    NoSemanticPayload>;
def ArithXOrI : ActorSchema<9, "::mlir::arith::XOrIOp", "Compute",
    NoSemanticPayload>;
def ArithDivSI : ActorSchema<10, "::mlir::arith::DivSIOp", "Compute",
    ArithExact>;
def ArithDivUI : ActorSchema<11, "::mlir::arith::DivUIOp", "Compute",
    ArithExact>;
def ArithRemSI : ActorSchema<12, "::mlir::arith::RemSIOp", "Compute",
    NoSemanticPayload>;
def ArithRemUI : ActorSchema<13, "::mlir::arith::RemUIOp", "Compute",
    NoSemanticPayload>;
def ArithMinSI : ActorSchema<14, "::mlir::arith::MinSIOp", "Compute",
    NoSemanticPayload>;
def ArithMaxSI : ActorSchema<15, "::mlir::arith::MaxSIOp", "Compute",
    NoSemanticPayload>;
def ArithMinUI : ActorSchema<16, "::mlir::arith::MinUIOp", "Compute",
    NoSemanticPayload>;
def ArithMaxUI : ActorSchema<17, "::mlir::arith::MaxUIOp", "Compute",
    NoSemanticPayload>;
def ArithCmpI : ActorSchema<18, "::mlir::arith::CmpIOp", "Compute",
    ArithIntegerCompare>;
def ArithSelect : ActorSchema<19, "::mlir::arith::SelectOp", "Compute",
    NoSemanticPayload>;
def ArithExtSI : ActorSchema<20, "::mlir::arith::ExtSIOp", "Compute",
    NoSemanticPayload>;
def ArithExtUI : ActorSchema<21, "::mlir::arith::ExtUIOp", "Compute",
    ArithNonNegative>;
def ArithTruncI : ActorSchema<22, "::mlir::arith::TruncIOp", "Compute",
    ArithIntegerOverflow>;
def ArithIndexCast : ActorSchema<23, "::mlir::arith::IndexCastOp", "Compute",
    NoSemanticPayload>;
def ArithIndexCastUI : ActorSchema<24, "::mlir::arith::IndexCastUIOp",
    "Compute", ArithNonNegative>;
def ArithBitcast : ActorSchema<25, "::mlir::arith::BitcastOp", "Compute",
    NoSemanticPayload>;
def ArithAddF : ActorSchema<26, "::mlir::arith::AddFOp", "Compute",
    ArithFloatingPoint>;
def ArithSubF : ActorSchema<27, "::mlir::arith::SubFOp", "Compute",
    ArithFloatingPoint>;
def ArithMulF : ActorSchema<28, "::mlir::arith::MulFOp", "Compute",
    ArithFloatingPoint>;
def ArithDivF : ActorSchema<29, "::mlir::arith::DivFOp", "Compute",
    ArithFloatingPoint>;
def ArithRemF : ActorSchema<30, "::mlir::arith::RemFOp", "Compute",
    ArithFloatingPoint>;
def ArithNegF : ActorSchema<31, "::mlir::arith::NegFOp", "Compute",
    ArithFloatingPoint>;
def ArithMinimumF : ActorSchema<32, "::mlir::arith::MinimumFOp", "Compute",
    ArithFloatingPoint>;
def ArithMaximumF : ActorSchema<33, "::mlir::arith::MaximumFOp", "Compute",
    ArithFloatingPoint>;
def ArithMinNumF : ActorSchema<34, "::mlir::arith::MinNumFOp", "Compute",
    ArithFloatingPoint>;
def ArithMaxNumF : ActorSchema<35, "::mlir::arith::MaxNumFOp", "Compute",
    ArithFloatingPoint>;
def ArithCmpF : ActorSchema<36, "::mlir::arith::CmpFOp", "Compute",
    ArithFloatCompare>;
def ArithExtF : ActorSchema<37, "::mlir::arith::ExtFOp", "Compute",
    ArithFloatingPoint>;
def ArithTruncF : ActorSchema<38, "::mlir::arith::TruncFOp", "Compute",
    ArithFloatingPoint>;
def ArithSIToFP : ActorSchema<39, "::mlir::arith::SIToFPOp", "Compute",
    NoSemanticPayload>;
def ArithUIToFP : ActorSchema<40, "::mlir::arith::UIToFPOp", "Compute",
    ArithNonNegative>;
def ArithFPToSI : ActorSchema<41, "::mlir::arith::FPToSIOp", "Compute",
    NoSemanticPayload>;
def ArithFPToUI : ActorSchema<42, "::mlir::arith::FPToUIOp", "Compute",
    NoSemanticPayload>;

//===----------------------------------------------------------------------===//
// Elementary mathematical function schemas
//===----------------------------------------------------------------------===//

def MathAbsF : ActorSchema<43, "::mlir::math::AbsFOp", "Compute",
    ArithFloatingPoint>;
def MathAbsI : ActorSchema<44, "::mlir::math::AbsIOp", "Compute",
    NoSemanticPayload>;
def MathSin : ActorSchema<45, "::mlir::math::SinOp", "Compute",
    ArithFloatingPoint>;
def MathCos : ActorSchema<46, "::mlir::math::CosOp", "Compute",
    ArithFloatingPoint>;
def MathTan : ActorSchema<47, "::mlir::math::TanOp", "Compute",
    ArithFloatingPoint>;
def MathSinh : ActorSchema<48, "::mlir::math::SinhOp", "Compute",
    ArithFloatingPoint>;
def MathCosh : ActorSchema<49, "::mlir::math::CoshOp", "Compute",
    ArithFloatingPoint>;
def MathTanh : ActorSchema<50, "::mlir::math::TanhOp", "Compute",
    ArithFloatingPoint>;
def MathExp : ActorSchema<51, "::mlir::math::ExpOp", "Compute",
    ArithFloatingPoint>;
def MathExp2 : ActorSchema<52, "::mlir::math::Exp2Op", "Compute",
    ArithFloatingPoint>;
def MathExpM1 : ActorSchema<53, "::mlir::math::ExpM1Op", "Compute",
    ArithFloatingPoint>;
def MathLog : ActorSchema<54, "::mlir::math::LogOp", "Compute",
    ArithFloatingPoint>;
def MathLog2 : ActorSchema<55, "::mlir::math::Log2Op", "Compute",
    ArithFloatingPoint>;
def MathLog10 : ActorSchema<56, "::mlir::math::Log10Op", "Compute",
    ArithFloatingPoint>;
def MathLog1p : ActorSchema<57, "::mlir::math::Log1pOp", "Compute",
    ArithFloatingPoint>;
def MathFloor : ActorSchema<58, "::mlir::math::FloorOp", "Compute",
    ArithFloatingPoint>;
def MathCeil : ActorSchema<59, "::mlir::math::CeilOp", "Compute",
    ArithFloatingPoint>;
def MathRound : ActorSchema<60, "::mlir::math::RoundOp", "Compute",
    ArithFloatingPoint>;
def MathTrunc : ActorSchema<61, "::mlir::math::TruncOp", "Compute",
    ArithFloatingPoint>;
def MathRoundEven : ActorSchema<62, "::mlir::math::RoundEvenOp", "Compute",
    ArithFloatingPoint>;
def MathSqrt : ActorSchema<63, "::mlir::math::SqrtOp", "Compute",
    ArithFloatingPoint>;
def MathRsqrt : ActorSchema<64, "::mlir::math::RsqrtOp", "Compute",
    ArithFloatingPoint>;
def MathErf : ActorSchema<65, "::mlir::math::ErfOp", "Compute",
    ArithFloatingPoint>;
def MathFma : ActorSchema<66, "::mlir::math::FmaOp", "Compute",
    ArithFloatingPoint>;
def MathCountLeadingZeros : ActorSchema<93,
    "::mlir::math::CountLeadingZerosOp", "Compute", NoSemanticPayload>;

//===----------------------------------------------------------------------===//
// Fixed-vector structural schemas
//===----------------------------------------------------------------------===//

def VectorExtract : ActorSchema<94, "::mlir::vector::ExtractOp", "Compute",
    VectorStaticPosition>;
def VectorInsert : ActorSchema<95, "::mlir::vector::InsertOp", "Compute",
    VectorStaticPosition>;
def VectorShuffle : ActorSchema<96, "::mlir::vector::ShuffleOp", "Compute",
    VectorShuffleMask>;

//===----------------------------------------------------------------------===//
// Irreducible LLVM compute schemas
//
// An LLVM-dialect operation is registered only when no standard arith, math,
// or vector operation exactly represents it. Exact aliases such as llvm.add,
// llvm.icmp, llvm.trunc, llvm.select, and llvm.bitcast normalize to their
// standard schema during mechanical raising and therefore have no schema of
// their own: one basic computation has one schema identity, and a Fabric
// capability registry never lists a dialect alias. `llvm.intr.fmuladd` is
// deliberately absent because an unmaterialized fused-or-split decision may
// not be a canonical actor.
//
// The retained operations are irreducible for exactly one of two reasons:
//
//   * no standard operation spells the computation at all (`llvm.freeze`,
//     `llvm.fshl`, `llvm.bswap`, `llvm.usub.sat`); or
//   * the standard spelling exists but cannot state the operation's exact
//     poison contract: `math.ctlz` and `math.absi` are exactly the
//     flag-free `llvm.ctlz` and `llvm.abs` forms, so those forms normalize
//     away and only the poison-flagged form remains an LLVM actor, with the
//     flag projected through its closed typed case.
//
// `ub.poison` stands with them because no constant operation can spell a
// poison value. Its built-in optional attribute has one parameterless
// full-poison value; any other poison interface implementation or discardable
// state fails closed instead of becoming an open semantic payload.
//
// `llvm.extractvalue` and `llvm.insertvalue` have no exact standard-dialect
// spelling. Their aggregate position is projected through one closed typed
// case. `llvm.shufflevector` on a fixed vector with a constant mask is exactly
// `vector.shuffle`, so it has no schema of its own.
//===----------------------------------------------------------------------===//

def UBPoison : ActorSchema<67, "::mlir::ub::PoisonOp", "Compute",
    NoSemanticPayload>;
def LLVMFreeze : ActorSchema<68, "::mlir::LLVM::FreezeOp", "Compute",
    NoSemanticPayload>;
def LLVMFshl : ActorSchema<69, "::mlir::LLVM::FshlOp", "Compute",
    NoSemanticPayload>;
def LLVMByteSwap : ActorSchema<70, "::mlir::LLVM::ByteSwapOp", "Compute",
    NoSemanticPayload>;
def LLVMUSubSat : ActorSchema<71, "::mlir::LLVM::USubSat", "Compute",
    NoSemanticPayload>;
def LLVMCountLeadingZeros : ActorSchema<72,
    "::mlir::LLVM::CountLeadingZerosOp", "Compute", LLVMZeroPoison>;
def LLVMAbs : ActorSchema<73, "::mlir::LLVM::AbsOp", "Compute",
    LLVMIntegerMinPoison>;
def LLVMExtractValue : ActorSchema<74, "::mlir::LLVM::ExtractValueOp",
    "Compute", LLVMAggregatePosition>;
def LLVMInsertValue : ActorSchema<75, "::mlir::LLVM::InsertValueOp",
    "Compute", LLVMAggregatePosition>;

//===----------------------------------------------------------------------===//
// Canonical Dataflow control and memory actor schemas
//===----------------------------------------------------------------------===//

def DataflowStream : ActorSchema<76, "::dataflow::StreamOp", "Control",
    StreamRecurrence>;
def DataflowCarry : ActorSchema<77, "::dataflow::CarryOp", "Control",
    NoSemanticPayload>;
def DataflowInvariant : ActorSchema<78, "::dataflow::InvariantOp", "Control",
    NoSemanticPayload>;
def DataflowGate : ActorSchema<79, "::dataflow::GateOp", "Control",
    NoSemanticPayload>;
def DataflowConstant : ActorSchema<80, "::dataflow::ConstantOp", "Control",
    TypedConstantValue>;
def DataflowSync : ActorSchema<81, "::dataflow::SyncOp", "Control",
    NoSemanticPayload>;
def DataflowMux : ActorSchema<82, "::dataflow::MuxOp", "Control",
    NoSemanticPayload>;
def DataflowDemux : ActorSchema<83, "::dataflow::DemuxOp", "Control",
    NoSemanticPayload>;
def DataflowParallelize : ActorSchema<84, "::dataflow::ParallelizeOp",
    "Control", NoSemanticPayload>;
def DataflowSerialize : ActorSchema<85, "::dataflow::SerializeOp", "Control",
    NoSemanticPayload>;
def DataflowPack : ActorSchema<86, "::dataflow::PackOp", "Control",
    NoSemanticPayload>;
def DataflowUnpack : ActorSchema<87, "::dataflow::UnpackOp", "Control",
    NoSemanticPayload>;
def DataflowLoad : ActorSchema<88, "::dataflow::LoadOp", "Memory",
    MemoryContract>;
def DataflowStore : ActorSchema<89, "::dataflow::StoreOp", "Memory",
    MemoryContract>;
def DataflowAtomicRmw : ActorSchema<90, "::dataflow::AtomicRmwOp", "Memory",
    MemoryContract>;
def DataflowCmpXchg : ActorSchema<91, "::dataflow::CmpXchgOp", "Memory",
    MemoryContract>;
def DataflowFence : ActorSchema<92, "::dataflow::FenceOp", "Memory",
    MemoryContract>;

//===----------------------------------------------------------------------===//
// Normative scalar compute implementation families
//
// Every family below admits only scalar actor shapes. Widths, formats, and
// predicates are concrete-resource capability, never family identity.
//===----------------------------------------------------------------------===//

def ScalarIntegerAddSub : ImplementationFamily<0, [ArithAddI, ArithSubI],
    ScalarIntegerParams, ScalarOrdinaryIntegerAdmission>;
def ScalarIntegerLogic : ImplementationFamily<1,
    [ArithAndI, ArithOrI, ArithXOrI],
    ScalarIntegerParams, ScalarLogicIntegerAdmission>;
def ScalarIntegerShift : ImplementationFamily<2,
    [ArithShLI, ArithShRSI, ArithShRUI],
    ScalarIntegerParams, ScalarOrdinaryIntegerAdmission>;
def ScalarIntegerCompareMinMax : ImplementationFamily<3,
    [ArithCmpI, ArithMinSI, ArithMaxSI, ArithMinUI, ArithMaxUI],
    ScalarIntegerCompareMinMaxParams, ScalarIntegerCompareAdmission>;
def ScalarValueSelect : ImplementationFamily<4, [ArithSelect],
    ScalarValueSelectParams, ScalarValueSelectAdmission>;
def ScalarIntegerCast : ImplementationFamily<5,
    [ArithExtSI, ArithExtUI, ArithTruncI, ArithIndexCast, ArithIndexCastUI],
    ScalarIntegerCastParams, ScalarIntegerCastAdmission>;
def ScalarBitReinterpret : ImplementationFamily<6, [ArithBitcast],
    ScalarBitReinterpretParams, ScalarBitReinterpretAdmission>;
def ScalarFloatSign : ImplementationFamily<7, [ArithNegF, MathAbsF],
    ScalarFloatParams, ScalarUniformFloatAdmission>;
def ScalarFloatAddSub : ImplementationFamily<8, [ArithAddF, ArithSubF],
    ScalarFloatParams, ScalarUniformFloatAdmission>;
def ScalarFloatCompareMinMax : ImplementationFamily<9,
    [ArithCmpF, ArithMinimumF, ArithMaximumF, ArithMinNumF, ArithMaxNumF],
    ScalarFloatCompareMinMaxParams, ScalarFloatCompareAdmission>;
def ScalarFloatWidthCast : ImplementationFamily<10, [ArithExtF, ArithTruncF],
    ScalarFloatWidthCastParams, ScalarFloatCastAdmission>;
def ScalarIntegerToFloat : ImplementationFamily<11,
    [ArithSIToFP, ArithUIToFP],
    ScalarIntegerFloatConversionParams, ScalarIntegerFloatConversionAdmission>;
def ScalarFloatToInteger : ImplementationFamily<12,
    [ArithFPToSI, ArithFPToUI],
    ScalarIntegerFloatConversionParams, ScalarIntegerFloatConversionAdmission>;
def ScalarIntegerMultiply : ImplementationFamily<13, [ArithMulI],
    ScalarIntegerParams, ScalarOrdinaryIntegerAdmission>;
def ScalarFloatMultiply : ImplementationFamily<14, [ArithMulF],
    ScalarFloatParams, ScalarUniformFloatAdmission>;
def ScalarFloatFma : ImplementationFamily<15, [MathFma],
    ScalarFloatParams, ScalarUniformFloatAdmission>;

//===----------------------------------------------------------------------===//
// Normative loop-control implementation families
//
// These are four distinct physical families. Composing them in one FU is a
// Builder convenience and never makes them one family.
//===----------------------------------------------------------------------===//

def LoopStream : ImplementationFamily<16, [DataflowStream],
    LoopStreamParams, StreamAdmission>;
def LoopCarry : ImplementationFamily<17, [DataflowCarry],
    TokenPlaneParams, TokenPlaneAdmission>;
def LoopInvariant : ImplementationFamily<18, [DataflowInvariant],
    TokenPlaneParams, TokenPlaneAdmission>;
def LoopGate : ImplementationFamily<19, [DataflowGate],
    TokenPlaneParams, TokenPlaneAdmission>;

#endif // LOOM_OPERATION_SCHEMAS_TD
