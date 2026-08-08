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
def NoSemanticPayload : SemanticsCase<0, 0x4c530101>;
def ArithFloatingPoint : SemanticsCase<1, 0x4c530102>;
def ArithIntegerOverflow : SemanticsCase<2, 0x4c530103>;
def ArithIntegerCompare : SemanticsCase<3, 0x4c530104>;
def ArithFloatCompare : SemanticsCase<4, 0x4c530105>;
def TypedConstantValue : SemanticsCase<5, 0x4c530106>;
def StreamRecurrence : SemanticsCase<6, 0x4c530107>;
def MemoryContract : SemanticsCase<7, 0x4c530108>;
def LLVMZeroPoison : SemanticsCase<8, 0x4c530109>;
def LLVMIntegerMinPoison : SemanticsCase<9, 0x4c53010a>;
def LLVMAggregatePosition : SemanticsCase<10, 0x4c53010b>;
def ArithExact : SemanticsCase<11, 0x4c53010c>;
def ArithNonNegative : SemanticsCase<12, 0x4c53010d>;
def VectorStaticPosition : SemanticsCase<13, 0x4c53010e>;
def VectorShuffleMask : SemanticsCase<14, 0x4c53010f>;
def LLVMDisjoint : SemanticsCase<15, 0x4c530110>;
def LLVMRegisteredIntrinsic : SemanticsCase<16, 0x4c530111>;
def LLVMGetElementPtrSemantics : SemanticsCase<17, 0x4c530112>;
def SpecialMathAccuracy : SemanticsCase<18, 0x4c530113>;

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
def FixedVectorIntegerParams : CapabilityParamsSchema<11>;
def FixedVectorIntegerCompareMinMaxParams : CapabilityParamsSchema<12>;
def FixedVectorValueSelectParams : CapabilityParamsSchema<13>;
def FixedVectorFloatParams : CapabilityParamsSchema<14>;
def FixedVectorFloatCompareMinMaxParams : CapabilityParamsSchema<15>;
def FixedVectorAdapterParams : CapabilityParamsSchema<16>;
def PayloadCapacityParams : CapabilityParamsSchema<17>;
def RoutedTokenParams : CapabilityParamsSchema<18>;
def FixedVectorSliceAlignMergeParams : CapabilityParamsSchema<19>;
def FixedVectorShuffleParams : CapabilityParamsSchema<20>;
def ScalarSpecialMathParams : CapabilityParamsSchema<21>;

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
def FixedVectorOrdinaryIntegerAdmission : TypedAdmissionProvider<12>;
def FixedVectorLogicIntegerAdmission : TypedAdmissionProvider<13>;
def FixedVectorIntegerCompareAdmission : TypedAdmissionProvider<14>;
def FixedVectorValueSelectAdmission : TypedAdmissionProvider<15>;
def FixedVectorUniformFloatAdmission : TypedAdmissionProvider<16>;
def FixedVectorFloatCompareAdmission : TypedAdmissionProvider<17>;
def FixedVectorAdapterAdmission : TypedAdmissionProvider<18>;
def ConstantTokenAdmission : TypedAdmissionProvider<19>;
def SyncTokenAdmission : TypedAdmissionProvider<20>;
def MuxTokenAdmission : TypedAdmissionProvider<21>;
def DemuxTokenAdmission : TypedAdmissionProvider<22>;
def ScalarUnaryIntegerAdmission : TypedAdmissionProvider<23>;
def FixedVectorUnaryIntegerAdmission : TypedAdmissionProvider<24>;
def FixedVectorSliceAlignMergeAdmission : TypedAdmissionProvider<25>;
def FixedVectorShuffleAdmission : TypedAdmissionProvider<26>;
def ScalarSpecialMathAdmission : TypedAdmissionProvider<27>;

//===----------------------------------------------------------------------===//
// Standard integer, floating-point, and cast compute schemas
//===----------------------------------------------------------------------===//

def ArithConstant : ActorSchema<0, 0x4c530201, "::mlir::arith::ConstantOp", "Compute",
    TypedConstantValue>;
def ArithAddI : ElementwiseActorSchema<1, 0x4c530202, "::mlir::arith::AddIOp", "Compute",
    ArithIntegerOverflow>;
def ArithSubI : ElementwiseActorSchema<2, 0x4c530203, "::mlir::arith::SubIOp", "Compute",
    ArithIntegerOverflow>;
def ArithMulI : ElementwiseActorSchema<3, 0x4c530204, "::mlir::arith::MulIOp", "Compute",
    ArithIntegerOverflow>;
def ArithShLI : ElementwiseActorSchema<4, 0x4c530205, "::mlir::arith::ShLIOp", "Compute",
    ArithIntegerOverflow>;
def ArithShRSI : ElementwiseActorSchema<5, 0x4c530206, "::mlir::arith::ShRSIOp", "Compute",
    ArithExact>;
def ArithShRUI : ElementwiseActorSchema<6, 0x4c530207, "::mlir::arith::ShRUIOp", "Compute",
    ArithExact>;
def ArithAndI : ElementwiseActorSchema<7, 0x4c530208, "::mlir::arith::AndIOp", "Compute",
    NoSemanticPayload>;
def ArithOrI : ElementwiseActorSchema<8, 0x4c530209, "::mlir::arith::OrIOp", "Compute",
    NoSemanticPayload>;
def ArithXOrI : ElementwiseActorSchema<9, 0x4c53020a, "::mlir::arith::XOrIOp", "Compute",
    NoSemanticPayload>;
def ArithDivSI : ActorSchema<10, 0x4c53020b, "::mlir::arith::DivSIOp", "Compute",
    ArithExact>;
def ArithDivUI : ActorSchema<11, 0x4c53020c, "::mlir::arith::DivUIOp", "Compute",
    ArithExact>;
def ArithRemSI : ActorSchema<12, 0x4c53020d, "::mlir::arith::RemSIOp", "Compute",
    NoSemanticPayload>;
def ArithRemUI : ActorSchema<13, 0x4c53020e, "::mlir::arith::RemUIOp", "Compute",
    NoSemanticPayload>;
def ArithMinSI : ElementwiseActorSchema<14, 0x4c53020f, "::mlir::arith::MinSIOp", "Compute",
    NoSemanticPayload>;
def ArithMaxSI : ElementwiseActorSchema<15, 0x4c530210, "::mlir::arith::MaxSIOp", "Compute",
    NoSemanticPayload>;
def ArithMinUI : ElementwiseActorSchema<16, 0x4c530211, "::mlir::arith::MinUIOp", "Compute",
    NoSemanticPayload>;
def ArithMaxUI : ElementwiseActorSchema<17, 0x4c530212, "::mlir::arith::MaxUIOp", "Compute",
    NoSemanticPayload>;
def ArithCmpI : ElementwiseActorSchema<18, 0x4c530213, "::mlir::arith::CmpIOp", "Compute",
    ArithIntegerCompare>;
def ArithSelect : ElementwiseActorSchema<19, 0x4c530214, "::mlir::arith::SelectOp", "Compute",
    NoSemanticPayload>;
def ArithExtSI : ElementwiseActorSchema<20, 0x4c530215, "::mlir::arith::ExtSIOp", "Compute",
    NoSemanticPayload>;
def ArithExtUI : ElementwiseActorSchema<21, 0x4c530216, "::mlir::arith::ExtUIOp", "Compute",
    ArithNonNegative>;
def ArithTruncI : ElementwiseActorSchema<22, 0x4c530217, "::mlir::arith::TruncIOp", "Compute",
    ArithIntegerOverflow>;
def ArithIndexCast : ElementwiseActorSchema<23, 0x4c530218, "::mlir::arith::IndexCastOp", "Compute",
    NoSemanticPayload>;
def ArithIndexCastUI : ElementwiseActorSchema<24, 0x4c530219, "::mlir::arith::IndexCastUIOp",
    "Compute", ArithNonNegative>;
def ArithBitcast : ElementwiseActorSchema<25, 0x4c53021a, "::mlir::arith::BitcastOp", "Compute",
    NoSemanticPayload>;
def ArithAddF : ElementwiseActorSchema<26, 0x4c53021b, "::mlir::arith::AddFOp", "Compute",
    ArithFloatingPoint>;
def ArithSubF : ElementwiseActorSchema<27, 0x4c53021c, "::mlir::arith::SubFOp", "Compute",
    ArithFloatingPoint>;
def ArithMulF : ElementwiseActorSchema<28, 0x4c53021d, "::mlir::arith::MulFOp", "Compute",
    ArithFloatingPoint>;
def ArithDivF : ElementwiseActorSchema<29, 0x4c53021e, "::mlir::arith::DivFOp", "Compute",
    ArithFloatingPoint>;
def ArithRemF : ElementwiseActorSchema<30, 0x4c53021f, "::mlir::arith::RemFOp", "Compute",
    ArithFloatingPoint>;
def ArithNegF : ElementwiseActorSchema<31, 0x4c530220, "::mlir::arith::NegFOp", "Compute",
    ArithFloatingPoint>;
def ArithMinimumF : ElementwiseActorSchema<32, 0x4c530221, "::mlir::arith::MinimumFOp", "Compute",
    ArithFloatingPoint>;
def ArithMaximumF : ElementwiseActorSchema<33, 0x4c530222, "::mlir::arith::MaximumFOp", "Compute",
    ArithFloatingPoint>;
def ArithMinNumF : ElementwiseActorSchema<34, 0x4c530223, "::mlir::arith::MinNumFOp", "Compute",
    ArithFloatingPoint>;
def ArithMaxNumF : ElementwiseActorSchema<35, 0x4c530224, "::mlir::arith::MaxNumFOp", "Compute",
    ArithFloatingPoint>;
def ArithCmpF : ElementwiseActorSchema<36, 0x4c530225, "::mlir::arith::CmpFOp", "Compute",
    ArithFloatCompare>;
def ArithExtF : ElementwiseActorSchema<37, 0x4c530226, "::mlir::arith::ExtFOp", "Compute",
    ArithFloatingPoint>;
def ArithTruncF : ElementwiseActorSchema<38, 0x4c530227, "::mlir::arith::TruncFOp", "Compute",
    ArithFloatingPoint>;
def ArithSIToFP : ElementwiseActorSchema<39, 0x4c530228, "::mlir::arith::SIToFPOp", "Compute",
    NoSemanticPayload>;
def ArithUIToFP : ElementwiseActorSchema<40, 0x4c530229, "::mlir::arith::UIToFPOp", "Compute",
    ArithNonNegative>;
def ArithFPToSI : ElementwiseActorSchema<41, 0x4c53022a, "::mlir::arith::FPToSIOp", "Compute",
    NoSemanticPayload>;
def ArithFPToUI : ElementwiseActorSchema<42, 0x4c53022b, "::mlir::arith::FPToUIOp", "Compute",
    NoSemanticPayload>;

//===----------------------------------------------------------------------===//
// Elementary mathematical function schemas
//===----------------------------------------------------------------------===//

def MathAbsF : ElementwiseActorSchema<43, 0x4c53022c, "::mlir::math::AbsFOp", "Compute",
    ArithFloatingPoint>;
def MathAbsI : ElementwiseActorSchema<44, 0x4c53022d, "::mlir::math::AbsIOp", "Compute",
    NoSemanticPayload>;
def MathSin : ElementwiseActorSchema<45, 0x4c53022e, "::mlir::math::SinOp",
                                     "Compute", SpecialMathAccuracy>;
def MathCos : ElementwiseActorSchema<46, 0x4c53022f, "::mlir::math::CosOp",
                                     "Compute", SpecialMathAccuracy>;
def MathTan : ElementwiseActorSchema<47, 0x4c530230, "::mlir::math::TanOp",
                                     "Compute", SpecialMathAccuracy>;
def MathSinh : ElementwiseActorSchema<48, 0x4c530231, "::mlir::math::SinhOp",
                                      "Compute", SpecialMathAccuracy>;
def MathCosh : ElementwiseActorSchema<49, 0x4c530232, "::mlir::math::CoshOp",
                                      "Compute", SpecialMathAccuracy>;
def MathTanh : ElementwiseActorSchema<50, 0x4c530233, "::mlir::math::TanhOp",
                                      "Compute", SpecialMathAccuracy>;
def MathExp : ElementwiseActorSchema<51, 0x4c530234, "::mlir::math::ExpOp",
                                     "Compute", SpecialMathAccuracy>;
def MathExp2 : ElementwiseActorSchema<52, 0x4c530235, "::mlir::math::Exp2Op",
                                      "Compute", SpecialMathAccuracy>;
def MathExpM1 : ElementwiseActorSchema<53, 0x4c530236, "::mlir::math::ExpM1Op",
                                       "Compute", SpecialMathAccuracy>;
def MathLog : ElementwiseActorSchema<54, 0x4c530237, "::mlir::math::LogOp",
                                     "Compute", SpecialMathAccuracy>;
def MathLog2 : ElementwiseActorSchema<55, 0x4c530238, "::mlir::math::Log2Op",
                                      "Compute", SpecialMathAccuracy>;
def MathLog10 : ElementwiseActorSchema<56, 0x4c530239, "::mlir::math::Log10Op",
                                       "Compute", SpecialMathAccuracy>;
def MathLog1p : ElementwiseActorSchema<57, 0x4c53023a, "::mlir::math::Log1pOp",
                                       "Compute", SpecialMathAccuracy>;
def MathFloor : ElementwiseActorSchema<58, 0x4c53023b, "::mlir::math::FloorOp",
                                       "Compute", SpecialMathAccuracy>;
def MathCeil : ElementwiseActorSchema<59, 0x4c53023c, "::mlir::math::CeilOp",
                                      "Compute", SpecialMathAccuracy>;
def MathRound : ElementwiseActorSchema<60, 0x4c53023d, "::mlir::math::RoundOp",
                                       "Compute", SpecialMathAccuracy>;
def MathTrunc : ElementwiseActorSchema<61, 0x4c53023e, "::mlir::math::TruncOp",
                                       "Compute", SpecialMathAccuracy>;
def MathRoundEven
    : ElementwiseActorSchema<62, 0x4c53023f, "::mlir::math::RoundEvenOp",
                             "Compute", SpecialMathAccuracy>;
def MathSqrt : ElementwiseActorSchema<63, 0x4c530240, "::mlir::math::SqrtOp",
                                      "Compute", SpecialMathAccuracy>;
def MathRsqrt : ElementwiseActorSchema<64, 0x4c530241, "::mlir::math::RsqrtOp",
                                       "Compute", SpecialMathAccuracy>;
def MathErf : ElementwiseActorSchema<65, 0x4c530242, "::mlir::math::ErfOp",
                                     "Compute", SpecialMathAccuracy>;
def MathFma : ElementwiseActorSchema<66, 0x4c530243, "::mlir::math::FmaOp", "Compute",
    ArithFloatingPoint>;
def MathPowF : ElementwiseActorSchema<106, 0x4c53026b, "::mlir::math::PowFOp",
                                      "Compute", SpecialMathAccuracy>;
def MathCountLeadingZeros : ElementwiseActorSchema<93, 0x4c53025e,
    "::mlir::math::CountLeadingZerosOp", "Compute", NoSemanticPayload>;
def MathCountTrailingZeros : ElementwiseActorSchema<101, 0x4c530266,
    "::mlir::math::CountTrailingZerosOp", "Compute", NoSemanticPayload>;

//===----------------------------------------------------------------------===//
// Fixed-vector structural schemas
//===----------------------------------------------------------------------===//

def VectorExtract : ActorSchema<94, 0x4c53025f, "::mlir::vector::ExtractOp", "Compute",
    VectorStaticPosition>;
def VectorInsert : ActorSchema<95, 0x4c530260, "::mlir::vector::InsertOp", "Compute",
    VectorStaticPosition>;
def VectorShuffle : ActorSchema<96, 0x4c530261, "::mlir::vector::ShuffleOp", "Compute",
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
//     `llvm.fshl`, `llvm.bswap`, and the signed/unsigned saturating add/sub
//     intrinsics); or
//   * the standard spelling exists but cannot state the operation's exact
//     poison contract: `math.ctlz`, `math.cttz`, and `math.absi` are exactly
//     the flag-free LLVM forms, so those forms normalize away and only the
//     poison-flagged form remains an LLVM actor, with the flag projected
//     through its closed typed case.
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
//
// `llvm.call_intrinsic` is a generic carrier rather than one semantic
// operation. Only exact source-owned intrinsic IDs with registered disjoint
// selectors may enter Canonical Dataflow. Their overloaded spelling and
// signature are validated against LLVM's intrinsic registry.
//===----------------------------------------------------------------------===//

def UBPoison : ActorSchema<67, 0x4c530244, "::mlir::ub::PoisonOp", "Compute",
    NoSemanticPayload>;
def LLVMFreeze : ElementwiseActorSchema<68, 0x4c530245, "::mlir::LLVM::FreezeOp", "Compute",
    NoSemanticPayload>;
def LLVMFshl : ElementwiseActorSchema<69, 0x4c530246, "::mlir::LLVM::FshlOp", "Compute",
    NoSemanticPayload>;
def LLVMByteSwap : ElementwiseActorSchema<70, 0x4c530247, "::mlir::LLVM::ByteSwapOp", "Compute",
    NoSemanticPayload>;
def LLVMUSubSat : ElementwiseActorSchema<71, 0x4c530248, "::mlir::LLVM::USubSat", "Compute",
    NoSemanticPayload>;
def LLVMCountLeadingZeros : ElementwiseActorSchema<72, 0x4c530249,
    "::mlir::LLVM::CountLeadingZerosOp", "Compute", LLVMZeroPoison>;
def LLVMCountTrailingZeros : ElementwiseActorSchema<102, 0x4c530267,
    "::mlir::LLVM::CountTrailingZerosOp", "Compute", LLVMZeroPoison>;
def LLVMAbs : ElementwiseActorSchema<73, 0x4c53024a, "::mlir::LLVM::AbsOp", "Compute",
    LLVMIntegerMinPoison>;
def LLVMExtractValue : ActorSchema<74, 0x4c53024b, "::mlir::LLVM::ExtractValueOp",
    "Compute", LLVMAggregatePosition>;
def LLVMInsertValue : ActorSchema<75, 0x4c53024c, "::mlir::LLVM::InsertValueOp",
    "Compute", LLVMAggregatePosition>;
def LLVMOrDisjoint : ElementwiseActorSchema<97, 0x4c530262, "::mlir::LLVM::OrOp", "Compute",
    LLVMDisjoint>;
def LLVMSAddSat : ElementwiseActorSchema<98, 0x4c530263, "::mlir::LLVM::SAddSat", "Compute",
    NoSemanticPayload>;
def LLVMUAddSat : ElementwiseActorSchema<99, 0x4c530264, "::mlir::LLVM::UAddSat", "Compute",
    NoSemanticPayload>;
def LLVMSSubSat : ElementwiseActorSchema<100, 0x4c530265, "::mlir::LLVM::SSubSat", "Compute",
    NoSemanticPayload>;
def LLVMFPToSISat : ElementwiseActorSchema<103, 0x4c530268,
    "::mlir::LLVM::CallIntrinsicOp", "Compute", LLVMRegisteredIntrinsic,
    LLVMIntrinsicSelector<"fptosi_sat">>;
def LLVMFPToUISat : ElementwiseActorSchema<104, 0x4c530269,
    "::mlir::LLVM::CallIntrinsicOp", "Compute", LLVMRegisteredIntrinsic,
    LLVMIntrinsicSelector<"fptoui_sat">>;
def LLVMGetElementPtr : ActorSchema<105, 0x4c53026a,
    "::mlir::LLVM::GEPOp", "Compute", LLVMGetElementPtrSemantics>;

//===----------------------------------------------------------------------===//
// Canonical Dataflow control and memory actor schemas
//===----------------------------------------------------------------------===//

def DataflowStream : ActorSchema<76, 0x4c53024d, "::dataflow::StreamOp", "Control",
    StreamRecurrence>;
def DataflowCarry : ActorSchema<77, 0x4c53024e, "::dataflow::CarryOp", "Control",
    NoSemanticPayload>;
def DataflowInvariant : ActorSchema<78, 0x4c53024f, "::dataflow::InvariantOp", "Control",
    NoSemanticPayload>;
def DataflowGate : ActorSchema<79, 0x4c530250, "::dataflow::GateOp", "Control",
    NoSemanticPayload>;
def DataflowConstant : ActorSchema<80, 0x4c530251, "::dataflow::ConstantOp", "Control",
    TypedConstantValue>;
def DataflowSync : ActorSchema<81, 0x4c530252, "::dataflow::SyncOp", "Control",
    NoSemanticPayload>;
def DataflowMux : ActorSchema<82, 0x4c530253, "::dataflow::MuxOp", "Control",
    NoSemanticPayload>;
def DataflowDemux : ActorSchema<83, 0x4c530254, "::dataflow::DemuxOp", "Control",
    NoSemanticPayload>;
def DataflowParallelize : ActorSchema<84, 0x4c530255, "::dataflow::ParallelizeOp",
    "Control", NoSemanticPayload>;
def DataflowSerialize : ActorSchema<85, 0x4c530256, "::dataflow::SerializeOp", "Control",
    NoSemanticPayload>;
def DataflowPack : ActorSchema<86, 0x4c530257, "::dataflow::PackOp", "Control",
    NoSemanticPayload>;
def DataflowUnpack : ActorSchema<87, 0x4c530258, "::dataflow::UnpackOp", "Control",
    NoSemanticPayload>;
def DataflowLoad : ActorSchema<88, 0x4c530259, "::dataflow::LoadOp", "Memory",
    MemoryContract>;
def DataflowStore : ActorSchema<89, 0x4c53025a, "::dataflow::StoreOp", "Memory",
    MemoryContract>;
def DataflowAtomicRmw : ActorSchema<90, 0x4c53025b, "::dataflow::AtomicRmwOp", "Memory",
    MemoryContract>;
def DataflowCmpXchg : ActorSchema<91, 0x4c53025c, "::dataflow::CmpXchgOp", "Memory",
    MemoryContract>;
def DataflowFence : ActorSchema<92, 0x4c53025d, "::dataflow::FenceOp", "Memory",
    MemoryContract>;

//===----------------------------------------------------------------------===//
// Normative scalar compute implementation families
//
// Every family below admits only scalar actor shapes. Widths, formats, and
// predicates are concrete-resource capability, never family identity.
//===----------------------------------------------------------------------===//

def ScalarIntegerAddSub : ImplementationFamily<0,
    [ArithAddI, ArithSubI, LLVMGetElementPtr],
    ScalarIntegerParams, ScalarOrdinaryIntegerAdmission>;
def ScalarIntegerLogic : ImplementationFamily<1,
    [ArithAndI, ArithOrI, ArithXOrI, LLVMOrDisjoint],
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
    [ArithFPToSI, ArithFPToUI, LLVMFPToSISat, LLVMFPToUISat],
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

// Fixed-vector compute resources use the same canonical actor schemas as
// scalar compute, but a distinct physical family and typed shape admission.
def FixedVectorIntegerAddSub : ImplementationFamily<20,
    [ArithAddI, ArithSubI], FixedVectorIntegerParams,
    FixedVectorOrdinaryIntegerAdmission>;
def FixedVectorIntegerLogic : ImplementationFamily<21,
    [ArithAndI, ArithOrI, ArithXOrI, LLVMOrDisjoint], FixedVectorIntegerParams,
    FixedVectorLogicIntegerAdmission>;
def FixedVectorIntegerShift : ImplementationFamily<22,
    [ArithShLI, ArithShRSI, ArithShRUI], FixedVectorIntegerParams,
    FixedVectorOrdinaryIntegerAdmission>;
def FixedVectorIntegerCompareMinMax : ImplementationFamily<23,
    [ArithCmpI, ArithMinSI, ArithMaxSI, ArithMinUI, ArithMaxUI],
    FixedVectorIntegerCompareMinMaxParams,
    FixedVectorIntegerCompareAdmission>;
def FixedVectorValueSelect : ImplementationFamily<24, [ArithSelect],
    FixedVectorValueSelectParams, FixedVectorValueSelectAdmission>;
def FixedVectorIntegerMultiply : ImplementationFamily<25, [ArithMulI],
    FixedVectorIntegerParams, FixedVectorOrdinaryIntegerAdmission>;
def FixedVectorFloatSign : ImplementationFamily<26, [ArithNegF, MathAbsF],
    FixedVectorFloatParams, FixedVectorUniformFloatAdmission>;
def FixedVectorFloatAddSub : ImplementationFamily<27, [ArithAddF, ArithSubF],
    FixedVectorFloatParams, FixedVectorUniformFloatAdmission>;
def FixedVectorFloatCompareMinMax : ImplementationFamily<28,
    [ArithCmpF, ArithMinimumF, ArithMaximumF, ArithMinNumF, ArithMaxNumF],
    FixedVectorFloatCompareMinMaxParams, FixedVectorFloatCompareAdmission>;
def FixedVectorFloatMultiply : ImplementationFamily<29, [ArithMulF],
    FixedVectorFloatParams, FixedVectorUniformFloatAdmission>;
def FixedVectorFloatFma : ImplementationFamily<30, [MathFma],
    FixedVectorFloatParams, FixedVectorUniformFloatAdmission>;

// Adapter and token-control resources remain physically distinct. Their
// common parameter records describe capacity, not circuit sharing.
def FixedVectorPack : ImplementationFamily<31, [DataflowPack],
    FixedVectorAdapterParams, FixedVectorAdapterAdmission>;
def FixedVectorUnpack : ImplementationFamily<32, [DataflowUnpack],
    FixedVectorAdapterParams, FixedVectorAdapterAdmission>;
def FixedVectorParallelize : ImplementationFamily<33, [DataflowParallelize],
    FixedVectorAdapterParams, FixedVectorAdapterAdmission>;
def FixedVectorSerialize : ImplementationFamily<34, [DataflowSerialize],
    FixedVectorAdapterParams, FixedVectorAdapterAdmission>;
def TokenConstant : ImplementationFamily<35, [DataflowConstant],
    PayloadCapacityParams, ConstantTokenAdmission>;
def TokenSync : ImplementationFamily<36, [DataflowSync],
    RoutedTokenParams, SyncTokenAdmission>;
def TokenMux : ImplementationFamily<37, [DataflowMux],
    RoutedTokenParams, MuxTokenAdmission>;
def TokenDemux : ImplementationFamily<38, [DataflowDemux],
    RoutedTokenParams, DemuxTokenAdmission>;

// Special operations are singleton physical families unless quotient and
// remainder are two projections of the same signedness-specific divider.
def ScalarSignedIntegerDivRem : ImplementationFamily<39,
    [ArithDivSI, ArithRemSI], ScalarIntegerParams,
    ScalarOrdinaryIntegerAdmission>;
def ScalarUnsignedIntegerDivRem : ImplementationFamily<40,
    [ArithDivUI, ArithRemUI], ScalarIntegerParams,
    ScalarOrdinaryIntegerAdmission>;
def ScalarFloatDivide : ImplementationFamily<41, [ArithDivF],
    ScalarFloatParams, ScalarUniformFloatAdmission>;
def ScalarFloatRemainder : ImplementationFamily<42, [ArithRemF],
    ScalarFloatParams, ScalarUniformFloatAdmission>;
def ScalarMathSin : ImplementationFamily<43, [MathSin], ScalarSpecialMathParams,
                                         ScalarSpecialMathAdmission>;
def ScalarMathCos : ImplementationFamily<44, [MathCos], ScalarSpecialMathParams,
                                         ScalarSpecialMathAdmission>;
def ScalarMathTan : ImplementationFamily<45, [MathTan], ScalarSpecialMathParams,
                                         ScalarSpecialMathAdmission>;
def ScalarMathSinh
    : ImplementationFamily<46, [MathSinh], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathCosh
    : ImplementationFamily<47, [MathCosh], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathTanh
    : ImplementationFamily<48, [MathTanh], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathExp : ImplementationFamily<49, [MathExp], ScalarSpecialMathParams,
                                         ScalarSpecialMathAdmission>;
def ScalarMathExp2
    : ImplementationFamily<50, [MathExp2], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathExpM1
    : ImplementationFamily<51, [MathExpM1], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathLog : ImplementationFamily<52, [MathLog], ScalarSpecialMathParams,
                                         ScalarSpecialMathAdmission>;
def ScalarMathLog2
    : ImplementationFamily<53, [MathLog2], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathLog10
    : ImplementationFamily<54, [MathLog10], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathLog1p
    : ImplementationFamily<55, [MathLog1p], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathFloor
    : ImplementationFamily<56, [MathFloor], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathCeil
    : ImplementationFamily<57, [MathCeil], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathRound
    : ImplementationFamily<58, [MathRound], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathTrunc
    : ImplementationFamily<59, [MathTrunc], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathRoundEven
    : ImplementationFamily<60, [MathRoundEven], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathSqrt
    : ImplementationFamily<61, [MathSqrt], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathRsqrt
    : ImplementationFamily<62, [MathRsqrt], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def ScalarMathErf : ImplementationFamily<63, [MathErf], ScalarSpecialMathParams,
                                         ScalarSpecialMathAdmission>;
def ScalarIntegerSaturatingAddSub : ImplementationFamily<64,
    [LLVMSAddSat, LLVMUAddSat, LLVMSSubSat, LLVMUSubSat],
    ScalarIntegerParams, ScalarOrdinaryIntegerAdmission>;
def FixedVectorIntegerSaturatingAddSub : ImplementationFamily<65,
    [LLVMSAddSat, LLVMUAddSat, LLVMSSubSat, LLVMUSubSat],
    FixedVectorIntegerParams, FixedVectorOrdinaryIntegerAdmission>;
def ScalarIntegerCountZeros : ImplementationFamily<66,
    [MathCountLeadingZeros, MathCountTrailingZeros, LLVMCountLeadingZeros,
     LLVMCountTrailingZeros], ScalarIntegerParams,
    ScalarUnaryIntegerAdmission>;
def FixedVectorIntegerCountZeros : ImplementationFamily<67,
    [MathCountLeadingZeros, MathCountTrailingZeros, LLVMCountLeadingZeros,
     LLVMCountTrailingZeros], FixedVectorIntegerParams,
    FixedVectorUnaryIntegerAdmission>;
def ScalarMathPow
    : ImplementationFamily<68, [MathPowF], ScalarSpecialMathParams,
                           ScalarSpecialMathAdmission>;
def FixedVectorSliceAlignMerge : ImplementationFamily<69,
    [VectorExtract, VectorInsert], FixedVectorSliceAlignMergeParams,
    FixedVectorSliceAlignMergeAdmission>;
def FixedVectorShuffle : ImplementationFamily<70, [VectorShuffle],
    FixedVectorShuffleParams, FixedVectorShuffleAdmission>;

#endif // LOOM_OPERATION_SCHEMAS_TD
