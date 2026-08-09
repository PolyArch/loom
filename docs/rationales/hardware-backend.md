# Hardware Backend Rationale

Normative contracts are owned by
[RTL Lowering](../spec-rtl-lowering.md),
[Hardware Implementation](../spec-hardware-implementation.md),
[EDA Tooling](../spec-eda-tooling.md), and
[FPA Evaluation](../spec-fpa-estimation.md). Shared ASIC or FPGA target identity
and technology-corner keys are owned by
[Implementation Platform](../spec-implementation-platform.md). Exact external
files and tool-bundled resources are owned by the provider binding that
consumes them, while local tool, input-path, runtime, and script binding is
owned by
[External Tool Invocation](../spec-external-tool-invocation.md).

## Why RTL Is Derived From Fabric

Fabric is the architecture and capability truth. RTL lowering implements an
exact Fabric artifact under a resolved provider and ConfigurationABI; it cannot
classify operations by string, invent unsupported behavior, or change timing,
capacity, state, buffering, arbitration, and clock/reset contracts.

Provider dispatch is keyed by the Fabric-owned implementation family and
resolved capability view. A backend-local operation list, `cfg_mode` enum, or
hard-coded semantic table would compete with OperationSchema, HSG, and the
concrete resource. Missing provider support is typed Unsupported and produces
no successful RTL artifact.

The sealed semantic-field relation is needed because family membership alone
does not say which admitted actor differences require physical control. If
each provider independently grouped operations, chose width-sensitive keys,
or numbered modes, Fabric verification, ConfigurationABI, portable RTL, and
vendor RTL could disagree while each looked locally consistent. Deriving field
need, behavior equivalence, domain, and codec once from the concrete Fabric
capability removes those competing tables. `None`, finite, and direct carriers
are the three essential cases; adding per-family selector mechanisms would only
rename them and enlarge the semantic surface.

The quotient keys and direct carriers belong to the sealed Fabric relation
because only the concrete intersection of registered actor semantics, HSG
eligibility, typed parameters, physical ports, and constraints can determine
physical-behavior equality. The generated HSG TableGen registry owns family
identity and member eligibility, not this concrete quotient. Giving TableGen a
second behavior-key or direct-bit table would either duplicate the normative
Fabric relation or ignore concrete physical narrowing. A backend-local codec
would duplicate both the quotient and ConfigurationABI's physical encoding.
Moving the three Direct carriers into a new standalone codec specification was
also rejected: it would add a document owner without separating an independent
concept from the family relation that already determines the carrier.
The specification therefore defines component order and validity once; the
Fabric resolver implements that contract, and every provider consumes the
sealed result.

Floating arithmetic uses exact formats rather than representation widths in
its quotient because equal-width formats can have different exponent and
significand behavior. Sign manipulation is an unconditional exception: negate
and absolute value only transform the sign bit. Compare/minmax has a
conditional exception when the complete reachable image abandons every
format-specific NaN observation. A mixed strict/`nnan` image already requires
its exact-format modes, so relaxed actors reuse those modes; adding a third
width mode would represent no circuit requirement. An all-`nnan` equal-width
image instead needs only one sign-magnitude mode. Fixed-vector shape remains an
admission fact because one configured vector datapath can serve every reachable
positive lane shape with the same element behavior.

Fast-math flags and accepted special-math accuracy are permissions on an actor,
while required fast-math and accuracy guarantee are fixed resource facts.
Encoding either side into a configuration key would create modes that no
resource selector chooses. Registered refinements therefore form a
deterministic cover: a relaxed behavior first reuses a compatible stronger mode
already required by the complete image, and only an uncovered relaxed behavior
is normalized. Actor-local normalization was rejected because compatibility of
partial semantics is not transitive; it can add a relaxed key beside two
already sufficient strict keys. A profile member with no actor-selected image
is rejected rather than retained as a phantom field, so parameter cardinality
cannot compete with the reachable behavior relation.

Encoding the complete actor projection as a floating key was rejected because
it would duplicate OperationSchema identity and preserve permissions that do
not change hardware behavior. Splitting the behavior profile into a distinct
parameter record for every family and mode was rejected because it would
multiply capability schemas and concrete Fabric resources without adding a
semantic distinction. A backend-owned floating mode registry was rejected
because it would let portable and vendor providers disagree with Fabric. The
single family-local quotient in the sealed relation is the only necessary
owner.

The direct carrier is intersected with exact physical ports for the same
reason. A constant field sized to a parameter maximum wider than its result
port would retain values that no admitted actor can emit. Slice and shuffle
validators therefore use constructive minimum geometries rather than a
width-only check. This keeps the validator equal to the projector image and
lets ConfigurationABI reject same-width spare codes without learning family
semantics.

Poison shuffle lanes canonicalize to ordinary selector zero. The software
contract permits any defined refinement there, so a dedicated poison selector
would create an unobservable configuration mode and a backend comparison that
has no semantic owner. The positive result-block count remains explicit for a
different reason: a trailing selector zero may be a real selection of source
block zero, while a trailing physical slot may be inactive output padding.
Those states have different observable hardware behavior and cannot be
recovered from the selector array. One count field is the minimum complete
carrier; a poison sentinel would not solve active trailing selector zero, and
a per-slot validity mask would duplicate the same prefix boundary. Likewise,
constant type tags remain in actor identity but not in the physical bit
carrier; equal emitted bits are one configured behavior.

Direct GEP support was deferred instead of assigning its unbounded static
layout and index tuple a backend-private encoding. Normalizing stable-integral
GEP to explicit DataLayout-derived integer address arithmetic uses existing
bounded relations. A future address-generation resource can add one typed
bound and one Direct carrier when that distinction is physically required.

These rules close invariants retained by `loom.fabric 4.0` rather than
preserving the incomplete implementation as a compatible behavior. The generic
relation already rejected a missing total projector and stated that poison
creates no mode. Consequently, a GEP capability with no bounded projector and
a shuffle poison sentinel were never specification-conforming artifacts whose
identity could be retained. The result-count component closes the already
required total projector: omitting it makes active trailing selector zero and
padding indistinguishable. A future incompatible carrier still requires a new
Fabric major version.

This boundary also exposes the backend's natural implementation parallelism.
Providers for independent implementation families or provider ecosystems share
only the stable capability-view, ConfigurationABI, RTL protocol, and recipe
contracts. Once those contracts have representative anchors, independent
providers can be developed and verified without splitting ownership of Fabric
semantics or the structural lowering core. A complex floating-point family may
therefore have a dedicated vendor-provider implementation while related simple
integer families share one provider, without creating per-operation semantic
registries.

## Why A Common CIRCT Skeleton Comes First

Fabric lowering needs one target-independent structural form before portable,
ASIC-library, or FPGA-device choices diverge. CIRCT HW, Comb, and Seq already
provide that form. Loom therefore lowers the exact Fabric topology, protocols,
state boundaries, and ordinary target-independent logic into an ephemeral
CIRCT skeleton, then specializes only the leaves whose implementation really
depends on a provider.

This is analogous to lowering source semantics to LLVM IR before target code
generation. The skeleton is not another semantic Artifact and does not own a
second operation catalog. An `hw.module.generated` declaration is only a typed
rendezvous point between one exact Fabric occurrence and its resolved
implementation-family binding. The selected provider replaces it with portable
logic, a vendor primitive, or a contract-bound external module. Every Loom
abstract generated leaf must be gone before SystemVerilog export.

The export-complete skeleton is System-rooted because only the System closes
physical occurrences, concrete clock/reset contracts, external interfaces,
and interconnect. Lowering a Module definition remains useful, but it is a
slot-parameterized internal fragment. Publishing it as a complete
implementation would force the backend to invent the concrete domains and
would alias state when the same definition is instantiated twice.

Reuse is therefore decided by one derived complete specialization key rather
than by Module identity alone or by pessimistically cloning every occurrence.
Two occurrences may share a definition only when their ABI projection, bound
domain contracts, recipes, external implementations, and memory choices are
equal. This preserves occurrence identity at the instances, avoids recompiling
genuinely equal implementations, and prevents a shared HDL definition from
hiding per-occurrence choices.

The reusable skeleton contains the complete Fabric configuration domain rather
than one workload's Mapping selection. Mapping verification separately proves
the selected combinational handshake closure before Deployment or mapped
execution, and the execution harness must retain that gate. Feeding Mapping
into RTL lowering would either make the implementation workload-specific or
create a second owner for route and selector choices; both would defeat the
Fabric and ConfigurationABI boundary.

Lowering the complete design through Handshake or DC-SC would add another
authority for scheduling, buffering, progress, and resource sharing that Fabric
already owns. Those dialects remain available when a future source contract
actually requires their semantics; they are not mandatory transit layers for
an already scheduled Fabric.

## Why Ordered Cardinality Uses One Shared Claim Boundary

`parallelize` and `serialize` are unusual because one logical firing can
publish several ordered token tuples. Treating each tuple as an ordinary
one-cycle firing would release the Fabric claim too early and would lose the
software firing identity. Keeping a provider-private continuation flag would
hide the same lifetime from Fabric, Mapping, and common backpressure logic.

The distilled boundary keeps the distinctions already present. Dataflow owns
the ordered production groups and their runtime multiplicity. The concrete
Fabric operation owns one claim envelope, adapter state, commit, and final
release. The common skeleton materializes the physical holding slot and retires
the Fabric-owned claim only after the provider identifies the Dataflow-derived
final group.
This composes three existing owners without a second actor or scheduler.

Two broader alternatives were rejected. Expanding each lane or terminal phase
into a separate Dataflow actor would change canonical firing identity and make
these implementation families redundant. A generic provider-managed
microtransaction engine would add another state-machine protocol even though
the registered production projection already supplies the only sequencing
needed. Both add conceptual surface without representing another observable
behavior.

Bits-only adapter ports also cannot discover poison or undef activity at
runtime. Three remedies were considered. A semantic-state sideband would widen
the entire Fabric transport model. Declaring both families permanently
unsupported would discard their defined behavior. The selected remedy keeps
Dataflow as the sole semantic owner: its derived activity-definedness proof is
the graph-only least fixed point of registered result transfer relations and
is required by TechMapping before choosing a bits-only adapter. It neither
depends on an invocation nor adds a persisted promise, and every backend
consumes the same proof boundary.

Ordered production groups and activity-definedness are derived views of the
existing Dataflow graph, Mapping persists `Intrinsic` release only for the
exact portable ordered-cardinality contract, and Fabric reuses the existing
ResourceContract fields. No artifact wire field, reference kind, or importer
accepted-language change is introduced. Previously authored one-cycle adapter
resources remain valid generic Fabric records but do not satisfy the portable
adapter provider; this is a support correction rather than an artifact schema
change.

## Why Non-Defined Values Need No RTL Sideband

Poison and Undef describe where the software semantic model no longer requires
one particular payload value. They are not runtime exceptions that every
physical datapath must detect. Requiring a poison bit, overlap checker, trap,
or stall protocol would change Fabric-visible ports and behavior and would
duplicate promises already owned by OperationSchema.

A total RTL circuit can instead choose one concrete bit value wherever the
software result is non-defined while remaining exact for every defined result.
The choice is local to the non-defined result lane, so a poisoned vector lane
does not relax its defined siblings. This is the smallest refinement relation
and applies uniformly to overflow, exactness, disjointness, and other
promise-bearing schemas.

This is why an ordinary floating-to-integer conversion and its matching
saturating conversion do not need separate configuration modes. The clamp
result required by the saturating schema is a valid concrete result wherever
the ordinary schema produces poison, while both schemas agree on every defined
ordinary result. The enabled schema set still determines whether clamp behavior
is required; it does not duplicate that fact in a selector key.

For the same reason, integer-to-floating and floating-to-integer resource
parameters contain no generic floating behavior profile. Their registered
schemas already fix rounding and exceptional-result semantics. Retaining a
second capability-level statement would create unreachable parameter values
and let two authorities disagree without representing another circuit choice.

The relation observes rather than rewrites Canonical Dataflow semantics. Undef
remains unconstrained until its owning observation or freeze, and a canonical
Defined result from that operation must still be reproduced exactly. Fabric
may declare an explicit checker or exception protocol when hardware
observability is genuinely required; a provider cannot invent one locally.

## Why Fake RTL Success Is Forbidden

An X-filled module, passthrough stub, unimplemented branch, or same-provider
software model can make a build or test pass without implementing the hardware.
Such output is more harmful than an explicit failure because downstream EDA
and comparison will treat it as real capability.

Leaf providers must implement the exact typed datapath and protocol. Structural
lowering then composes boundaries, FIFOs, temporal PEs, switches, memory
engines, backpressure, and adapters from their Fabric contracts. Unknown or
unrealizable resources stop publication.

## Why HardwareImplementation Is An Artifact

Generated RTL, synthesized netlist, physical layout, or FPGA image represents
an immutable implementation state with exact semantic dependencies, payload
roles, interfaces, activity points, memory macro bindings, external bindings,
and an optional target manifest. Each real implementation change produces
another HardwareImplementation.

The generator invocation is deliberately excluded from that state. A parent
implementation, recipe selection, search decision, and tool-flow configuration
explain how an output was produced, so `InvocationManifest` owns them. Keeping
them in HardwareImplementation as well would create a second lineage authority
and would assign different identities to byte-for-byte equivalent hardware
states reached through different paths. The output instead materializes every
fact needed by a consumer. Identical canonical states converge, while the
manifest may preserve every valid derivation.

A typed representation root is necessary because a flat representation tag,
payload bag, and caller-supplied top leave three parties able to disagree about
what the implementation actually represents. Making the variant own its root
locator and payload closure turns that into one invariant and lets finalization
reject incomplete or mismatched state. Downstream flows read those exact
BlobDigests through BlobStore so a work-directory path or duplicate RTL string
cannot silently substitute different hardware.

Opaque physical state needs one additional authored fact: a logical object
index that can be checked without understanding proprietary bytes. Three
remedies were considered:

* one provider-neutral canonical index payload and one physical descriptor
  shared by ASIC physical, FPGA physical, and FPGA image roots;
* a reserved filename or an overloaded `PhysicalDatabase` or `DeviceImage`
  payload that changes meaning when it happens to carry the index; and
* separate tool-named descriptors or index schemas for each physical provider.

The first remedy is selected. `RepresentationIndex` gives the index one exact
role and media type, while `indexed_physical` gives its grammar and validation
one semantic owner. The root variant and stage select the exact admission row,
so the format kind does not duplicate those state distinctions. The second
remedy would make a logical name or overloaded role into a hidden discriminator
and would leave media type ambiguous. The third would duplicate identical
closure, locator, and object rules across providers and make tool identity part
of representation semantics.

The index binds its format ref, root claim, own logical name, every other
payload descriptor, object catalog, and unresolved-definition set. It omits
only its own digest, which is necessarily the digest of the canonical index
bytes that contain those bindings. This breaks the self-reference cycle
without permitting undeclared payload state. Proprietary bytes remain opaque,
but every declared blob is still re-read and digest-verified. Appending the
role and format kind is a compatible schema extension, so
HardwareImplementation and the representation-format registry advance from
`2.0` to `2.1`; all existing numeric tags retain their meanings.

An implementation interface uses one role-bearing semantic reference rather
than a caller-authored key plus a separately authored role. The former is the
smallest complete fact: its closed alternative selects Data, Memory, Clock,
Reset, Configuration, or ExternalProtocol and embeds the exact Fabric or
ConfigurationABI owner. Keeping a free key or a second role would add identity
without adding meaning and would require reconciliation when they disagree.

HardwareImplementation cannot own a universal list of required interface
signals or activity points. Fabric owns hardware semantics, ConfigurationABI
owns programming units, and each provider or consumer owns its protocol and
required observation set. Projecting a mandatory global catalog would either
duplicate those owners or require HardwareImplementation to invent protocol
signal decomposition. The artifact therefore validates every declared typed
binding and locator, while the exact invocation or runtime contract checks the
subset it requires before execution. This also permits one physical locator to
serve several semantic references and one protocol to use several locators
without caller-authored grouping IDs.

## Why One Pinned Frontend Supplies HDL Representation Facts

RTL generation and representation indexing answer different questions. CIRCT
constructs and lowers hardware; the representation index proves what exact
objects exist in already materialized HDL, including unresolved external
definitions. The repository-pinned Slang AST retains those source-level facts,
whereas the public CIRCT import path can reject an unresolved definition before
it can expose a complete inventory. Making CIRCT IR the index in that state
would either reject legal black-box implementations or require inference from
diagnostic text.

Using Slang for unresolved definitions and CIRCT Moore for the rest would be
two authorities for hierarchy, names, widths, and object kinds. Any mismatch
would need precedence rules and reconciliation logic. One projection from the
Slang parse and elaboration result is the smaller contract: CIRCT remains the
lowering owner, Slang remains the sole indexing source, and the
HardwareImplementation finalizer compares the resulting facts with Fabric and
ConfigurationABI rather than allowing either parser to rewrite those semantic
owners. The frontend AST itself stays removable implementation state rather
than becoming another persistent catalog.

The versioned `RepresentationFormatDescriptorRef` is the semantic identity of
that contract. The exact Slang and CIRCT revisions and semantic build options
are derivation provenance in the producer build identity, not another consumer
selector. The descriptor fixes the options. Adding another implementation-
semantic token to the artifact would duplicate the descriptor version; if an
implementation change alters observable indexing behavior, the descriptor
version must change instead.

The exact top participates in indexing because reachability is top-dependent.
A file may contain test modules, helper definitions, and several possible
roots. Requiring closure for every unused definition would overstate the
implementation, while choosing the first or filename-matching module would
make source layout semantic. Exact-root reachability yields one deterministic
object catalog and only the unresolved definitions that the represented design
actually uses.

Object kinds deliberately follow stable HDL syntax instead of inferred
hardware intent. A port wins over the backing net or variable that Slang also
exposes at the same hierarchical name. A fixed-size unpacked variable is a
Memory and another fixed-size variable is a Register, but Register does not
promise a synthesized flip-flop. Proving storage implementation would require
procedural and synthesis analysis that neither locator validation nor
HardwareImplementation identity needs.

The initial locator grammar admits only explicitly named scalar hierarchy.
Inventing spellings for anonymous primitives, implicit generate blocks, or
array indices would add a second naming scheme that every importer and report
adapter must reproduce. Rejecting those cases in the initial grammar keeps
names equal to source-owned identifiers; a later grammar can add indexed
hierarchy only when it has one versioned canonical encoding.

The gate-netlist descriptor uses a conservative structural subset for the same
reason. A small whitelist of cells, nets, static elaboration, and wiring-only
continuous assignments composes into an inspectable connectivity graph.
The same wiring-expression grammar applies to continuous assignments, net
initializers, and cell actual connections. Otherwise behavioral logic rejected
in an assignment could be hidden directly in a port connection, forcing every
consumer to understand arbitrary expressions. One grammar closes that side
channel without creating per-connection rules or turning the read-only indexer
into a source transformer. Unknown cell definitions remain explicit Module
dependencies. Their occurrences and actual-expression shape can be checked
without guessing pin geometry. The current `BlackBoxContract` payload role is
an opaque content role and does not itself define a pin schema. If consumers
genuinely require pin facts, an approved versioned BlackBoxContract content
schema must own them. Exactly one external
binding closes each unresolved definition so two provider contracts cannot
both claim authority, while one library binding can close several definitions
without duplicating its dependency identity.

Language validity precedes subset admission. Retrying an invalid Verilog-2005
source as SystemVerilog, or scanning words such as `assert` and `property`,
would introduce a second language authority and would misclassify legal uses of
those words as identifiers. The selected profile and pinned frontend therefore
decide validity once; only a well-formed source can proceed to the structural
subset check and receive `Unsupported`.

Validity and admission are evaluated in two separated contexts of the same
frontend over the same parse trees. A single exact-top-forced compilation
cannot answer both questions: forcing the top makes the frontend report policy
artifacts - an unconnected top-level `ref` port, an `interface` or `program`
named as root, an unfollowed `include`, a module that needs a parameter
override - as errors, which conflates illegal HDL with legal HDL the
descriptor simply does not admit. The language-validation context applies no
such policy, so every error it reports is intrinsic to the source. The
alternative, cancelling the policy artifacts with a table of typed diagnostic
codes, would create a second, silently growable classification authority that
every frontend upgrade could perturb; one separated context is the smaller and
more durable contract. Classification is observable behavior, so the contract
moved from registry `1.0` to `2.0` without a compatibility path rather than
keeping two classification regimes.

QoR, pass/fail status, logs, and reports do not enter that artifact. They are
Evaluation observations or attempt material. This prevents a tool result from
changing implementation identity and lets several evaluations query the same
layout under different corners or requirements.

ImplementationPlatform separately owns the selected ASIC technology release or
FPGA ordering code and its typed corner keys. Exact library, macro, rule, IP,
and tool-bundled dependencies are owned by provider-specific bindings in the
implementation. The backend cannot hide a host filesystem path as portable
hardware truth.

## Why CIRCT And LLVM Are Pinned Together

CIRCT provides mature RTL and hardware IR infrastructure, so Loom should not
reimplement it. CIRCT's stable `firtool` releases are tested against an exact
LLVM revision. Loom therefore pins CIRCT to a selected stable release commit
and top-level LLVM to that revision's gitlink, builds only the top-level LLVM,
and leaves CIRCT's nested LLVM uninitialized.

This atomic pair avoids two incompatible MLIR/LLVM ABIs while retaining
unmodified upstream source. Build identity records the exact commits and
semantic options rather than following floating branches.

## Why Constraints Are Derived

Clock/reset domains, crossings, Fabric resource timing, generator bindings,
implementation interfaces, target and corner identity, and exact external
bindings already determine the SDC and verification harness. A handwritten or
backend-default constraint would be a second hardware contract. Generated
constraints and scripts are reproducible payloads or attempt inputs whose
derivation is recorded.

Implementation-only choices such as floorplan or tool flow are typed generator
inputs and produce a new HardwareImplementation. A choice that changes a
Fabric-visible timing or capacity fact must instead produce a new Fabric
candidate; the backend cannot hide it as an implementation option.

## Why EDA Generation And Evaluation Are Separate

Synthesis, placement, routing, extraction, and FPGA implementation create new
immutable hardware state, so they are Candidate Generators. Timing, power,
area, correctness, and physical checks observe one exact finalized state, so
they are Evaluations. Some tools emit both products in one run, but allowing
one importer to publish a half-finalized implementation and its reports as one
result would make tool completion the semantic owner of both hardware and
Evidence.

The generator therefore finalizes HardwareImplementation first. The baseline
then constructs a subsequent exact EvaluationRequest and prepares a separate
evaluation bundle over that immutable state. It deliberately does not adopt a
generation attempt's reports: when several derivations converge to one
HardwareImplementation, choosing one nonsemantic attempt would otherwise make
the same Request depend on hidden history. A later optimization may reuse a run
only after a versioned typed cross-attempt contract owns that choice. The
baseline still permits a lightweight evaluator to load the finalized database
rather than repeat placement or routing. This preserves adverse completed
Evidence, while tool crash, license failure, timeout, unsupported primitive,
and structurally incomplete output remain distinct failures that cannot create
either partial hardware or partial Evidence.

External generation and Evaluation use two necessary calls: preparation asks
the existing semantic owners to validate their portions and writes a
deterministic bundle; import validates the exact completed bundle and returns
the typed result to its central finalizer. The caller executes `run.sh` between
them. A synchronous callback would
make compiler workers own long-lived EDA execution, while a generic persistent
Job model would duplicate bundle completion, the execution journal, and site
schedulers. The two-call boundary provides parallelism and recovery without
either extra authority.

The completion record belongs only to the script attempt between those calls.
Extending it backward into preparation or forward into import would make one
record span separately owned API operations, or require import to rewrite
already completed attempt state. Typed preparation and import errors preserve
the same distinctions without another lifecycle record or mutable status.

Generated scripts are part of the compiler output because they make the exact
tool translation inspectable and reusable. The optional execution path merely
invokes the top-level script. Containers, modules, licenses, resource limits,
and schedulers retain their existing owners instead of being approximated by a
backend-specific process manager.

## Why Direct EDA Material Stays Local

The Artifact Store answers whether an object is a valid immutable semantic
result. Git answers whether bytes are suitable for public source distribution.
Conflating those decisions would either weaken Evidence semantics or publish
tool, library, IP, design, and host details that belong to licensed or private
environments. Normalized Evidence remains the one semantic evaluation result,
but direct EDA Evidence and all attempt material stay in an ignored local store.

The rule applies equally to open-source and commercial runs. That removes a
vendor classification table, prevents a mixed flow from changing disclosure
status halfway through, and keeps repository review mechanical. Parser tests
use small authored semantic fixtures rather than captured report snapshots.

Repository automation defaults local attempts to the ignored build tree
because it already owns generated output. An explicitly configured ignored
repository directory or external directory covers workstation and cluster
placement without adding another project-specific scratch convention.

Model parameters remain distinct semantic results, but semantic validity does
not make their bytes suitable for source publication. A disclosure-review
exception for `ModelParameterBundle` would add a second authority that must
decide whether weights reveal private training inputs and would make the rule
depend on an unrecorded review state. Keeping EDA-derived bundles local uses
one rule for raw Evidence, normalized Evidence, training corpora, and derived
weights without inventing a sanitized Evidence schema, public-weight
projection, or duplicate model format.

## Why Capabilities Compose Instead Of Ecosystem Flows

The five required tool ecosystems are coverage families, not five compiler
models. Simulation, synthesis, implementation, extraction, timing, power, and
physical verification have the same ownership boundaries regardless of vendor.
Representing each product suite as a monolithic flow would duplicate
HardwareImplementation finalization, invocation lineage, constraints, Evidence
normalization, failure classification, and script execution policy.

The central candidate-generator and evaluator descriptors already express the
real distinction: generators publish immutable implementation states, while
evaluators observe exact states. The open-source path supplies an inspectable
portable baseline. Synopsys and Cadence supply independent commercial ASIC
implementations and signoff-oriented observations. Vivado and Quartus Prime
supply the two static FPGA routes. New products extend one of those
capabilities instead of extending a global ecosystem enum or workflow language.

This also explains why every installed suite product is not mandatory in every
run. PrimeTime, Tempus, extraction, power, or physical-verification adapters
are useful only when their required implementation, target, and provider input
views exist.

Cerebrus is naturally a candidate search generator over typed flow decisions.
Stratus and Vitis HLS require an exact high-level input contract; sending
already generated RTL through HLS would add no capability. Virtuoso is useful
as a source of exact custom-cell and macro views without becoming a default
digital implementation stage.

Vendor arithmetic libraries and FPGA primitives remain explicit RTL recipes
because downstream inference is not reproducible implementation identity. An
explicit DesignWare, ChipWare, AMD/Xilinx, or Intel/Altera recipe records which
occurrence uses which provider contract and tool-bundled resource key. Portable
synthesizable RTL remains a distinct recipe, so open-source support does not
depend on proprietary IP and a commercial flow can deliberately choose the
portable implementation. No tool or adapter silently substitutes one for the
other.

OpenROAD observations are deliberately labeled by their actual implementation
state and model binding rather than promoted to generic signoff. Likewise, the
first VCS and Xcelium contract is functional simulation; timing annotation is
deferred until its payload and dependency contract has one semantic owner.
Completeness therefore means completing the declared capability scope, not
claiming that one tool name implies every signoff check.

## Why Provider Coverage Is Not A Cartesian Matrix

The full product-by-platform-by-recipe-by-operation matrix is neither necessary
nor affordable. Driver projection, output declaration, importer behavior, and
failure mapping can be tested without licenses. A real representative platform
then proves each baseline route, while scheduled broader runs establish the
declared process and device coverage. Evidence records exactly which checks ran,
so untested combinations never acquire implied support.

This structure exposes the available parallelism. One immutable RTL
implementation may fan out to independent simulation or implementation
bundles. Different occurrence recipe maps create independent generator
invocations from the same Fabric and normally materialize distinct RTL states;
equal canonical states converge without losing either manifest edge. Only true
data dependencies, such as synthesis before placement or extraction before
post-route timing, remain serial. License and machine scheduling stay outside
Loom, so adding concurrency does not require a shared mutable environment or
process manager.

## Why Version And Word Size Are Provider Binding Facts

Recent commercial tool generations replace or consolidate older products, but
a rolling wall-clock cutoff would make reproducibility depend on the day a
bundle is prepared. Provider maintenance therefore validates explicit current
releases and orders their exact module aliases ahead of generic site aliases.
The resolver still freezes the actual version and never guesses that a
directory name is newer. Explicit configuration remains highest priority but
cannot turn an incompatible executable into the selected provider.

Choosing a supported 64-bit launcher is likewise provider knowledge. It should
not be repeated in every user configuration or delegated to an unreliable
presentation `PATH`. Mandatory architecture tokens such as VCS `-full64` belong
to deterministic command projection and cannot be accidentally removed. This
keeps local installation paths private while making the generated script's
actual architecture choice inspectable.

## Why ImplementationPlatform Is A Target Manifest

The ASIC and FPGA conformance targets prove materially different technology and
device routes: educational and commercial ASIC releases, plus HBM-oriented and
DSP-oriented devices across two generations from each FPGA vendor. The shared
fact needed by every flow is small: which ASIC technology release or exact FPGA
ordering code is selected, and which typed technology-corner keys that target
defines. ImplementationPlatform owns only that manifest.

Putting libraries, macros, rule decks, user IP, or vendor databases into the
same object would turn a target identity into a content warehouse. It would
also make an unrelated view required by one provider perturb every consumer.
Keeping the manifest small gives each target fact one owner and lets a
target-independent RTL implementation omit Platform entirely.

## Why External Files Belong To Provider Bindings

The exact files required by synthesis, timing, extraction, simulation, or
physical verification differ by provider and capability. The consuming
provider contract is therefore the only place that can define the required
slots, roles, compatibility rules, and exact SHA-256 fingerprints. A
HardwareImplementation records the resolved bindings for dependencies that are
part of that implementation, such as a memory macro or black-box module.

Machine-local paths are only projections used to materialize an invocation
bundle. Two machines may map the same expected fingerprint to different paths
without changing semantic identity. A user-authored synthesizable RTL body is
an explicit generator input and becomes ordinary RTL payload. Encrypted or
black-box user IP remains an external binding with an explicit black-box
contract. Neither case broadens ImplementationPlatform.

There is no generic directory-import contract. When a provider genuinely needs
a logical directory, its typed input declares the exact relative ordinary-file
set, per-file fingerprints, and layout it consumes. The local file-tree
resolver scans only a user-selected root to prove that frozen requirement; it
cannot derive or broaden semantic membership. Loom never scans or hashes an
installation, PDK, or IP tree to discover semantic membership.

## Why Tool-Bundled Resources Use Provider Build Identity

DesignWare, ChipWare, and FPGA device databases are released as part of a tool
provider and are normally resolved internally by that tool. Copying or hashing
the complete installation tree would be expensive, license-sensitive, and less
precise than the actual dependency. Their semantic identity is the stable
provider build plus the exact resource, library, primitive, or device key used
by the recipe.

The local resolver still finds the executable through explicit configuration,
the current environment, or module discovery. That machine-local mechanism does
not own the provider build identity and does not change which bundled resource
the semantic binding selected.

## Why Functional Oracles Must Be Independent

A generated checker from the same backend can validate protocol and ABI, but
it cannot independently prove the backend's functional implementation. Mapped
RTL execution must compare requested terminal observations with an independent
DFG or CGRA execution under compatible workload and service contracts.

Raw waveforms and reports are useful diagnostics but remain attempt or scratch
material until an exact raw-bundle owner exists. Human summaries are removable
projections of typed Artifacts and Evidence.

## Why Low- And High-Fidelity FPA Share Metrics

An analytical architecture model and an EDA-backed model differ in method,
uncertainty, cost, and accuracy, not in what frequency, area, or power means.
Both therefore publish ordinary shared MetricResults for limiting frequency,
total area, dynamic power, and leakage power. Model-owned coefficients or EDA
report parsers stay behind their descriptor; neither creates a private FPA
record. This lets calibration compare like quantities while preserving the
lower-confidence model as a fast, complete early-stage estimate.

Low confidence means inaccurate absolute values, not an incomplete question.
The early model still estimates frequency, area, dynamic power, leakage power,
and runtime with coherent relative scaling. Omitting a dimension would force
frontend DSE to use a separate ad hoc score and would make later EDA calibration
change the optimization data model instead of only improving its evidence.

Rail analysis needs a different physical quantity, but not a provider-specific
record. Maximum voltage drop is comparable across static and dynamic methods
when it means the worst delivered-voltage deficit over the complete analyzed
network of the exact case. Method, activity, network coverage, and uncertainty
remain model facts. One shared `MaximumVoltageDrop` MetricKind therefore lets
Voltus and future providers answer the same question without turning a vendor
report field or severity label into an Evaluation authority.

The initial shared rail contract deliberately has no second provider-config
authority. Fixed method, activity-basis kind, complete-network coverage, and
uncertainty are recovered from the exact model config view; process corner,
applied voltage, and activity values are recovered from the same validated
Request. The Voltus adapter only translates that closed projection and binds
its exact PGV tree. Keeping the first model to one global supply and activity
clock avoids inventing a premature power-domain-to-layout schema while making
the limitation observable as typed `Unsupported`. Multi-domain or dynamic
analysis can add an exact model when its physical correspondence is owned,
without weakening the existing whole-case metric.
