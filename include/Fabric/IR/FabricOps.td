#ifndef FABRIC_OPS_TD
#define FABRIC_OPS_TD

include "Fabric/IR/FabricDialect.td"
include "Fabric/IR/FabricTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Properties.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Fabric_ScheduleAttr : I32EnumAttr<"Schedule",
    "fabric container schedule (spatial vs temporal time-share)",
    [
      I32EnumAttrCase<"Spatial",  0, "spatial">,
      I32EnumAttrCase<"Temporal", 1, "temporal">
    ]> {
  let cppNamespace = "::fabric";
}

def Fabric_BoundaryDirectionAttr : I32EnumAttr<"BoundaryDirection",
    "fabric.boundary direction (s2t / t2t / t2s)",
    [
      I32EnumAttrCase<"S2t", 0, "s2t">,
      I32EnumAttrCase<"T2t", 1, "t2t">,
      I32EnumAttrCase<"T2s", 2, "t2s">
    ]> {
  let cppNamespace = "::fabric";
}

def Fabric_TypeProp : Property<"::mlir::Type", "MLIR type"> {
  let convertFromAttribute = [{
    auto typeAttr = ::llvm::dyn_cast_if_present<::mlir::TypeAttr>($_attr);
    if (!typeAttr)
      return $_diag() << "expected type attribute";
    $_storage = typeAttr.getValue();
    return ::mlir::success();
  }];
  let convertToAttribute = [{
    return ::mlir::TypeAttr::get($_storage);
  }];
  let parser = [{
    if (::mlir::failed($_parser.parseType($_storage)))
      return ::mlir::failure();
  }];
  let printer = "$_printer.printType($_storage)";
  let readFromMlirBytecode = [{
    if (::mlir::failed($_reader.readType($_storage)))
      return ::mlir::failure();
  }];
  let writeToMlirBytecode = "$_writer.writeType($_storage)";
  let hashProperty = "::llvm::hash_value($_storage.getAsOpaquePointer())";
}

def Fabric_InnerInputTypesProp : ArrayProp<Fabric_TypeProp,
    "inner input type array"> {
  let defaultValue = "::llvm::ArrayRef<::mlir::Type>{}";
  let storageTypeValueOverride = "::llvm::SmallVector<::mlir::Type>{}";
}

// Temporal-PE FU configuration mode and operand buffer mode are stored
// as string-valued attributes ("per_instruction_fu_config" /
// "per_fu_config"; "per_instruction" / "per_input_port" /
// "all_fu_share"). The verifier validates the keyword set; the
// configuration generator interprets the keyword. We avoid a
// dedicated dialect attribute parser/printer here to keep the dialect
// initialization minimal.

def Fabric_MuxOp : Fabric_Op<"mux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric multiplexer with handshake";
  let description = [{
    Selects one of N (N > 1) inputs onto the single output. All inputs and the
    output share one `!fabric.bits<W>` type. No hardware parameters (the port
    width is fully described by the shared operand/result type).

    Software parameters (either all set or all unset):
      * `sel`        : i32, 0 <= sel < N (forced 0 when `disconnect` is true)
      * `discard`    : bool
      * `disconnect` : bool

    Unset means the op describes pure hardware (not programmed). Programmed
    combinations must satisfy: !(discard && disconnect).
  }];

  let arguments = (ins
      Variadic<Fabric_BitsType>:$inputs,
      OptionalAttr<I32Attr>:$sel,
      OptionalAttr<BoolAttr>:$discard,
      OptionalAttr<BoolAttr>:$disconnect
  );
  let results = (outs Fabric_BitsType:$output);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_DemuxOp : Fabric_Op<"demux", [Pure, SameOperandsAndResultType]> {
  let summary = "fabric demultiplexer with handshake";
  let description = [{
    Single input, N outputs (N > 1). All input/output ports share one
    `!fabric.bits<W>` type. No hardware parameters (the port width is fully
    described by the shared operand/result type). The op is the input/output
    dual of fabric.mux.

    Software parameters (either all set or all unset):
      * `sel`        : i32, 0 <= sel < N (forced 0 when `disconnect` is true)
      * `discard`    : bool
      * `disconnect` : bool

    Unset means the op describes pure hardware (not programmed). When
    programmed, `discard && disconnect` is forbidden. Untying valid/ready
    on the unselected outputs:

      * pure hardware: input.ready=0, all outputs.valid=0 (= disconnect)
      * sel only    : input <-> outputs[sel] passthrough; other outputs.valid=0
      * discard=T   : input.ready=1 (input drained); all outputs.valid=0
      * disconnect=T: input.ready=0; all outputs.valid=0
  }];

  let arguments = (ins
      Fabric_BitsType:$input,
      OptionalAttr<I32Attr>:$sel,
      OptionalAttr<BoolAttr>:$discard,
      OptionalAttr<BoolAttr>:$disconnect
  );
  let results = (outs Variadic<Fabric_BitsType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_FifoOp : Fabric_Op<"fifo", [Pure]> {
  let summary = "fabric FIFO with handshake";
  let description = [{
    Single-input single-output FIFO. Input and output share one Fabric port
    kind. Their widths may differ under the module-level low-bit-aligned
    truncation or zero-extension rule; the output type is the FIFO's internal
    storage width.

    Hardware parameters (always present, printed in `[...]`):
      * `max_depth`   : i32, must be > 0 (physical FIFO depth)
      * `bypassable`  : bool (whether hardware supports combinational bypass)

    Software parameter (only valid when `bypassable` is true, printed in `{}`):
      * `bypassed`    : bool (when true, the FIFO is configured in bypass mode;
                       data/valid/ready become a combinational passthrough)

    Unset `bypassed` describes pure hardware (not programmed). Setting
    `bypassed` when `bypassable` is false is illegal.
  }];

  let arguments = (ins
      Fabric_AnyType:$input,
      I32Attr:$max_depth,
      BoolAttr:$bypassable,
      OptionalAttr<BoolAttr>:$bypassed
  );
  let results = (outs Fabric_AnyType:$output);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_OpOp : Fabric_Op<"op", []> {
  let summary =
      "fabric module instance: a configurable hardware area that can implement "
      "one of a set of software ops";
  let description = [{
    Models a piece of fabric hardware (a "module") that can be
    configured to implement one of its typed `hw_params`. The required
    `op_list` mirrors the distinct operation identities in those modes.
    Members must either be a single op or all belong to one shared hardware-
    share group. Hardware ports are `fabric.bits<N>` only.

    Attributes:
      * `op_list`    : checked array of `FlatSymbolRefAttr`, e.g.
                       `[@arith.addi, @arith.subi]`. In a canonical FU its
                       distinct entries equal the operation identities in
                       `hw_params`.
      * `hw_params`  : an array of complete typed software modes owned
                       by this physical datapath. Each mode contains one
                       operation identity, exact function type, exact semantic
                       attributes, and ordered software-to-physical port maps.
                       FU semantic encodings select these modes by index. Each
                       mode must verify as the selected software operation;
                       mapped physical payload widths may differ and need only
                       be at least the corresponding software widths.
                       Legacy programmed Fabric may still use the historical
                       length-one allowed-set dictionary as an adapter form;
                       canonical FUs reject that shape.
      * `sw_configs` : `{ key = ... }` - a `DictionaryAttr` with
                       runtime configuration. Normalized `hw_params` modes are
                       selected only by `{mode = N}`. Legacy allowed-set input
                       retains its historical field selections. Canonical FUs
                       do not persist either programmed form.
  }];

  let arguments = (ins Variadic<Fabric_BitsType>:$inputs, ArrayAttr:$op_list,
      OptionalAttr<ArrayAttr>:$hw_params,
      OptionalAttr<DictionaryAttr>:$sw_configs);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);

  let assemblyFormat = [{
    $op_list `(` $inputs `)` attr-dict
      `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def Fabric_ModulePortType
    : AnyTypeOf<[Fabric_BitsType, Fabric_BitsTagType, AnyMemRef]>;

def Fabric_ModuleOp : Fabric_Op<"module", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">,
    Symbol,
    SymbolTable,
    DeclareOpInterfaceMethods<RegionKindInterface>
]> {
  let summary =
      "fabric module: top-level container of the fabric dialect";
  let description = [{
    Top-level container for the fabric dialect. Carries a required
    `sym_name`. The body is a single graph-region block whose
    terminator is `fabric.yield`.

    Module inputs are declared as the body's entry-block arguments
    (the op itself has zero SSA operands, like `func.func`). Module
    outputs are declared as result types on the op signature; the
    `fabric.yield` terminator supplies their values.

    Allowed input/output types: `!fabric.bits<W>`,
    `!fabric.bits_tag<W, T>`, `memref<...>`. No other MLIR types are
    accepted.

    The body accepts `fabric.pe`, `fabric.switch`, `fabric.mem`,
    `fabric.fifo`, nested `fabric.module`, `fabric.instantiate`, and
    `fabric.boundary`, plus the implicit `fabric.yield` terminator. See
    `docs/spec-fabric-module.md` for the full rules.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      // Optional per-module overrides for the global Loom constants
      // consumed by `fabric.mem` (and any future op that depends on the
      // address-space width or memory-bus width). When absent the
      // verifier falls back to ::loom::getDefaultLoomAddrBits() and
      // ::loom::getDefaultLoomMemBusWidth().
      OptionalAttr<I32Attr>:$loom_addr_bits,
      OptionalAttr<I32Attr>:$loom_mem_bus_width
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_PeOp : Fabric_Op<"pe", [
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>,
    HasParent<"::fabric::ModuleOp">,
    SymbolTable
]> {
  let summary =
      "fabric PE: a container of one or more fabric.fu ops with a schedule "
      "predicate (spatial or temporal)";
  let description = [{
    Processing element. Holds one or more `fabric.fu` ops in its body and
    carries a mandatory `schedule` predicate (`spatial` | `temporal`) that
    declares how the contained FUs are time-shared.

    The op exists in two disjoint syntactic forms by `sym_name` presence:

    Anonymous form (definition + use combined). Carries variadic SSA
    operands and variadic SSA results. The body has no terminator and
    inner FUs are not SSA-wired to the PE results.

      %r:2 = fabric.pe [spatial]
                 (%pa = %a : !fabric.bits<W>,
                  %pb = %b : !fabric.bits<W>)
                 -> (!fabric.bits<W>, !fabric.bits<W>) {
        fabric.fu(...) -> ...
      }

    Named template form (declaration only). Has zero SSA operands and zero
    SSA results in the host scope; the port signature is captured in the
    `function_type` attribute. The body's entry block carries the input
    port types as block arguments, and the body terminator
    `fabric.yield` supplies the values that match the function-type
    results. Actual usage of a named PE goes through `fabric.instantiate`.

      fabric.pe @ALU [spatial]
                (!fabric.bits<W>, !fabric.bits<W>) -> (!fabric.bits<W>) {
      ^bb0(%pa: !fabric.bits<W>, %pb: !fabric.bits<W>):
        fabric.fu(%fa = %pa : !fabric.bits<W>) -> !fabric.bits<W> { ... }
        fabric.yield %pa : !fabric.bits<W>
      }

    A temporal PE uses tagged boundary ports and carries the hardware and
    software configuration attributes declared below. Its verifier checks
    configuration completeness, instruction encoding, selector bounds, and
    FU compatibility. See `docs/spec-fabric-pe-temporal.md` for the complete
    temporal constraints.

    Constraints (spatial, both forms):
      * `K = numInputs() >= 1` and `L = numOutputs() >= 1`. For the
        anonymous form `K`/`L` are the operand/result counts; for the
        named form they are the function-type input/output counts.
      * Every PE port is `!fabric.bits<W>` with the same `W`. Mixing
        widths or non-`bits` fabric types on the PE boundary is rejected.
      * The body holds `fabric.fu` and `fabric.instantiate` ops and at
        least one such compute member. No `fabric.op`, `fabric.mux`,
        `fabric.demux`, `fabric.fifo`, or `fabric.pe` may appear
        directly in the body. `fabric.yield` may appear only in the
        named form.
      * For every inner FU `f`, `f.numInputs() <= K` and
        `f.numOutputs() <= L`. Every FU outer input port and every FU
        outer output port has type `!fabric.bits<W>` matching the PE's
        `W`. Inner FU block-arg types may be narrower
        (`!fabric.bits<F>` with `F <= W`), in which case the FU
        boundary truncates the high `W - F` bits.
      * The op must appear inside a `fabric.module` body.
  }];

  let arguments = (ins
      OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      Fabric_ScheduleAttr:$schedule,
      Variadic<Fabric_AnyType>:$inputs,
      // Temporal-only hardware parameters. All must be absent on a
      // spatial PE; presence/absence rules per-attribute are enforced
      // by the verifier (see docs/spec-fabric-pe-temporal.md).
      OptionalAttr<I32Attr>:$tag_width,
      OptionalAttr<I32Attr>:$num_instruction,
      OptionalAttr<I32Attr>:$num_reg_fifo,
      OptionalAttr<I32Attr>:$reg_fifo_depth,
      OptionalAttr<I32Attr>:$reg_fifo_ports,
      OptionalAttr<StrAttr>:$fu_config_mode,
      OptionalAttr<StrAttr>:$operand_buffer_mode,
      OptionalAttr<I32Attr>:$operand_buffer_size,
      // Temporal-only software configuration trio (all-or-nothing).
      OptionalAttr<BoolAttr>:$pe_enable,
      OptionalAttr<ArrayAttr>:$instruction_mem,
      OptionalAttr<ArrayAttr>:$per_fu_sw_configs);
  let results = (outs Variadic<Fabric_AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_FuOp : Fabric_Op<"fu", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>
]> {
  let summary =
      "fabric function unit (FU): a graph-region tile of fabric.op / mux / demux";
  let description = [{
    Container for one CGRA-style functional unit. The op exists in two
    disjoint syntactic forms by `sym_name` presence:

    Anonymous form (definition + use combined). Carries variadic SSA
    operands (with optional per-operand `to <inner-type>` clause for the
    FU input boundary's high-bit truncation) and variadic SSA results.
    The body's entry block argument types are the inner port types and
    the `fabric.yield` terminator supplies the values that match the
    FU's outer result types.

      %r = fabric.fu(%fa = %a : !fabric.bits<W> [to !fabric.bits<F>])
                 -> !fabric.bits<W> { ... fabric.yield %v : !fabric.bits<W> }

    Named template form (declaration only). Has zero SSA operands and
    zero SSA results; the port signature is captured in the
    `function_type` attribute. The body's entry block carries the input
    port types as block arguments and the `fabric.yield` terminator
    supplies values matching the function-type results. Actual usage of
    a named FU goes through `fabric.instantiate`.

      fabric.fu @F (!fabric.bits<W>, !fabric.bits<W>) -> !fabric.bits<W> {
      ^bb0(%fa: !fabric.bits<W>, %fb: !fabric.bits<W>):
        %v = fabric.op [@arith.muli] (%fa, %fb)
             : (!fabric.bits<W>, !fabric.bits<W>) -> !fabric.bits<W>
        fabric.yield %v : !fabric.bits<W>
      }

    Constraints:
      * The body must contain at least one `fabric.op`.
      * `fabric.fu` cannot be nested.
      * `fabric.fifo` is not permitted in the body.
      * The FU itself carries no `[hw_params]` / `{sw_configs}`;
        all configuration lives on the inner ops.
      * For each input port `i` (anonymous form only), the outer operand
        width must be greater than or equal to the inner block-arg width
        (`outer >= inner`). The named form has no outer operands.
      * Symmetric output-side boundary widening at the `fabric.yield`
        terminator (anonymous form): `fabric.yield %v : !fabric.bits<inner>
        to !fabric.bits<outer>` declares low-bit-aligned widening when
        `inner <= outer`. The high `(outer - inner)` bits are zero-filled
        at the FU boundary on each token; the outer type matches the
        FU's declared outer result type. Without the `to` clause, inner
        equals outer (strict).
  }];

  let arguments = (ins
      OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      OptionalAttr<ArrayAttr>:$valid_encodings,
      Variadic<Fabric_BitsType>:$inputs);
  let results = (outs Variadic<Fabric_BitsType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_InstantiateOp : Fabric_Op<"instantiate", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary =
      "fabric instantiation: bind a previously-defined fabric resource "
      "symbol into the current scope as a fresh hardware instance";
  let description = [{
    `fabric.instantiate` references a previously-defined fabric symbol
    (`fabric.module`, `fabric.pe`, `fabric.switch`, `fabric.mem`, or
    `fabric.fu`) and creates a fresh hardware instance with its own SSA
    inputs and results. The symbol target supplies the hardware shape
    (port count and per-port type); the instantiation site supplies the
    SSA inputs and consumer-side result SSA values.

    Allowed target kinds depend on the parent of the `fabric.instantiate`:
      * builtin top-level module: only `fabric.module` symbols.
      * `fabric.module` body: `fabric.module` (sibling top-level), or a
        named `fabric.pe`, `fabric.switch`, or `fabric.mem` defined inside
        the same module body.
      * `fabric.pe` body: a named `fabric.fu` defined inside the same
        pe body.

    Width relaxation at the input side mirrors the existing connection-
    point rule: the SSA operand type may differ in width from the target's
    declared input port type, expressed via the `to <inner-type>` clause.
    Differing destination types are retained in the ODS-owned typed
    `inner_input_types` property.
    `memref` operands must match exactly. Output direction is strict:
    each result type must equal the target's declared output port type.

    Forward and self-references are forbidden.
  }];

  let arguments = (ins
      FlatSymbolRefAttr:$callee,
      Variadic<Fabric_ModulePortType>:$inputs,
      Fabric_InnerInputTypesProp:$inner_input_types);
  let results = (outs Variadic<Fabric_ModulePortType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_BoundaryOp : Fabric_Op<"boundary", [Pure]> {
  let summary =
      "fabric boundary op: bridge between the spatial bits domain and the "
      "temporal bits_tag domain (direction = s2t | t2t | t2s)";
  let description = [{
    Single boundary op carrying a mandatory `direction` predicate that
    selects one of three transition shapes between the spatial
    `!fabric.bits<BW>` domain and the temporal `!fabric.bits_tag<BW, TW>`
    domain. The bracketed `[<direction>]` predicate is printed/parsed in
    the same style as `fabric.pe [spatial|temporal]`.

    See `docs/spec-fabric-boundary.md` for the full per-direction
    surface syntax (s2t general / s2t constant-tag / t2t LUT / t2s split
    / t2s drop-tag).

    Allowed only inside a `fabric.module` body. The direction-specific
    relationships between destination input-port and result widths remain
    exact. An incoming anonymous connection may use `source-type to
    destination-port-type` for same-kind `bits` or `bits_tag` width
    normalization; differing destination types are retained in the ODS-owned
    typed `inner_input_types` property. See
    `docs/spec-fabric-boundary.md` for the per-direction verifier rules and
    the t2t LUT config-mem encoding.
  }];

  let arguments = (ins
      Variadic<Fabric_AnyType>:$inputs,
      Fabric_BoundaryDirectionAttr:$direction,
      Fabric_InnerInputTypesProp:$inner_input_types,
      OptionalAttr<ArrayAttr>:$hw_params,
      OptionalAttr<DictionaryAttr>:$sw_configs
  );
  let results = (outs Variadic<Fabric_AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_SwitchOp : Fabric_Op<"switch", [
    DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>,
    Pure
]> {
  let summary =
      "fabric switch: configurable routing crossbar with a schedule predicate "
      "(spatial or temporal)";
  let description = [{
    Leaf-level routing op with variadic SSA inputs and variadic SSA results.
    Carries a mandatory `schedule` predicate (`spatial` | `temporal`) and a
    `connectivity_table` describing the static physical fabric. The op
    exists in two disjoint syntactic forms by `sym_name` presence:

    Anonymous form has variadic SSA operands and variadic SSA results. Its
    input type list accepts `source-type to destination-port-type` pairs for
    same-kind `bits` or `bits_tag` width normalization; differing destination
    types are retained in the ODS-owned typed `inner_input_types` property.
    Named template form has zero SSA operands and zero SSA results in the
    host scope; the port signature is captured in the `function_type`
    attribute. Hardware parameters are carried in `hw_params` (an
    ArrayAttr wrapping a single DictionaryAttr) printed in `[ ... ]`
    after the SSA operand list. Software configuration is carried in
    `sw_configs` (a DictionaryAttr) printed in `{ ... }` after the
    hardware parameters.

    See `docs/spec-fabric-switch.md` for the full per-schedule rules
    (connectivity_table format, MSB-on-left bit-string convention,
    route_table layout, tag-driven temporal trigger semantics, etc.).
  }];

  let arguments = (ins
      Variadic<Fabric_AnyType>:$inputs,
      OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      Fabric_ScheduleAttr:$schedule,
      Fabric_InnerInputTypesProp:$inner_input_types,
      OptionalAttr<ArrayAttr>:$hw_params,
      OptionalAttr<DictionaryAttr>:$sw_configs
  );
  let results = (outs Variadic<Fabric_AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_MemOp : Fabric_Op<"mem", [
    DeclareOpInterfaceMethods<Symbol, ["isOptionalSymbol"]>
]> {
  let summary = "fabric memory operation-engine hardware capability";
  let description = [{
    The accepted ABI describes one operation engine. Local memory service
    remains unsupported until its typed service contract is defined.

    Anonymous operands contain all manager/requester memrefs, followed by
    per-load `(addr, ctrl)` groups and per-store `(addr, data, ctrl)` groups.
    Results contain all subordinate/provider memrefs, followed by per-load
    `(data, done)` groups and per-store `done` results. Endpoint counts are
    derived from the signature and are not repeated as attributes. Incoming
    transport operands may use `source-type to destination-port-type` for
    same-kind width normalization.

    Named template form has zero SSA operands and zero SSA results in
    the host scope; the port signature is captured in `function_type`.
    `hw_params` is a length-one array containing the closed operation-engine
    hardware dictionary. It owns independent load/store port counts and
    operation data width, plus fixed request-source-to-manager dispatch
    eligibility. Temporal engines additionally own tag width and
    configured-row capacity. The registered schema is the closed canonical
    attribute set; the operation accepts no discardable attributes.

    See `docs/spec-fabric-mem.md` for the full per-schedule rules.
  }];

  let arguments = (ins
      Variadic<Fabric_ModulePortType>:$inputs,
      OptionalAttr<SymbolNameAttr>:$sym_name,
      OptionalAttr<TypeAttrOf<FunctionType>>:$function_type,
      Fabric_ScheduleAttr:$schedule,
      Fabric_InnerInputTypesProp:$inner_input_types,
      OptionalAttr<ArrayAttr>:$hw_params
  );
  let results = (outs Variadic<Fabric_ModulePortType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Fabric_SystemOp : Fabric_Op<"system", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">,
    Symbol,
    SymbolTable
]> {
  let summary = "fabric system: system-level heterogeneous SoC ADG container";
  let description = [{
    Top-level system ADG container. Carries a required symbol name and a
    required memory model. The body is a symbolic hardware graph made of
    `fabric.node`, `fabric.external_port`, and `fabric.link` operations.
    Connectivity is represented by explicit `fabric.link` endpoint fields,
    not by SSA values.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      StrAttr:$memory_model,
      OptionalAttr<StrAttr>:$model_name,
      OptionalAttr<DictionaryAttr>:$params
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name `memory_model` `=` $memory_model attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

def Fabric_NodeOp : Fabric_Op<"node", [
    HasParent<"::fabric::SystemOp">,
    Symbol
]> {
  let summary = "fabric system node";
  let description = [{
    Physical system-level hardware instance. The baseline implementation uses
    explicit string channel descriptors in the form `port.channel:direction`.
    Later structured port attributes must preserve the same endpoint identity
    and direction checks.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      StrAttr:$kind,
      ArrayAttr:$ports,
      OptionalAttr<FlatSymbolRefAttr>:$spatial,
      OptionalAttr<StrAttr>:$scalar,
      OptionalAttr<StrAttr>:$function,
      OptionalAttr<I64Attr>:$bytes,
      OptionalAttr<DictionaryAttr>:$params
  );

  let assemblyFormat = [{
    $sym_name `kind` `=` $kind `ports` `=` $ports attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

def Fabric_ExternalPortOp : Fabric_Op<"external_port", [
    HasParent<"::fabric::SystemOp">,
    Symbol
]> {
  let summary = "fabric system external port";
  let description = [{
    Intentional system-boundary protocol exposure. The baseline implementation
    uses the same `port.channel:direction` descriptors as `fabric.node`.
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      ArrayAttr:$ports,
      OptionalAttr<DictionaryAttr>:$params
  );

  let assemblyFormat = [{
    $sym_name `ports` `=` $ports attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

def Fabric_LinkOp : Fabric_Op<"link", [
    HasParent<"::fabric::SystemOp">
]> {
  let summary = "fabric system point-to-point channel link";
  let description = [{
    Directed point-to-point system link. The source endpoint must name an
    output channel, and the destination endpoint must name an input channel
    owned by a `fabric.node` or `fabric.external_port` in the same
    `fabric.system`.
  }];

  let arguments = (ins
      FlatSymbolRefAttr:$src,
      StrAttr:$src_port,
      StrAttr:$src_channel,
      FlatSymbolRefAttr:$dst,
      StrAttr:$dst_port,
      StrAttr:$dst_channel,
      OptionalAttr<StrAttr>:$protocol,
      OptionalAttr<I64Attr>:$latency,
      OptionalAttr<DictionaryAttr>:$params
  );

  let assemblyFormat = [{
    `src` `=` $src `src_port` `=` $src_port `src_channel` `=` $src_channel
    `dst` `=` $dst `dst_port` `=` $dst_port `dst_channel` `=` $dst_channel
    attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

def Fabric_YieldOp : Fabric_Op<"yield", [
    Terminator,
    Pure
]> {
  let summary = "terminator of a fabric.fu, fabric.module, or fabric.system region";
  let description = [{
    Closes the enclosing `fabric.fu`, `fabric.module`, or `fabric.system`
    body.

    Inside `fabric.fu`, `$values` supplies the values that leave the FU
    on its results; the number and types must match the parent op's
    result count and types exactly.

    Inside `fabric.module`, `$values` supplies the module's declared
    output values. The number and per-value types must match the
    module's declared result list under the width-relaxation rule
    documented in `docs/spec-fabric-module.md` (low-bit alignment, same
    fabric kind; memref must match exactly). The optional per-value
    `to <module-result-type>` clause makes the relaxation explicit:

      fabric.yield %v : !fabric.bits<32> to !fabric.bits<16>

    Inside `fabric.system`, the terminator carries no values. System
    connectivity is represented by `fabric.link` operations.
  }];

  // Yield operands are the SSA values being yielded. They must be one of
  // the allowed module port types when this yield closes a fabric.module
  // body, and !fabric.bits<W> when it closes a fabric.fu body. The op
  // accepts the union here; `verify` enforces the per-parent rule.
  let arguments = (ins Variadic<Fabric_ModulePortType>:$values);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // FABRIC_OPS_TD
