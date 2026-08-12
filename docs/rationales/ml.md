# ML Search Rationale

Normative contracts are owned by
[ML Environment Core](../spec-ml-core-environment.md),
[ML Model Core](../spec-ml-core-model.md),
[ML DSE Environment](../spec-ml-dse-environment.md),
[ML DSE Model Architecture](../spec-ml-dse-model-architecture.md),
[ML Training Core](../spec-ml-core-training.md),
[ML DSE Training](../spec-ml-dse-training.md),
[ML PnR Environment](../spec-ml-pnr-environment.md),
[ML PnR Model Architecture](../spec-ml-pnr-model-architecture.md), and
[ML PnR Training](../spec-ml-pnr-training.md).

Loom uses learned search in two places that look different and are not. One
explores a design space by selecting candidate-generator decisions; the other
places and routes one fixed problem by selecting Place and Route Actions. In
both, the learner is a proposal policy over decisions some other owner already
defined, and everything that decides what a decision means stays with that
owner. These documents explain why that boundary is where it is, and what was
rejected on either side of it.

## Why A Learned Search Policy Is A Harness, Not A Plan Node

A reinforcement-learning agent needs the two things the resolved plan
deliberately refuses: a mutable current design and a runtime loop. Adding
either to the plan would make termination, deterministic work, recovery, and
cache identity depend on a policy's sampled behavior, and the plan would stop
being readable before execution.

The alternative that was rejected was a generic environment action language
over node names and property bags. It would have reproduced exactly the problem
typed domain generators exist to prevent, and it would have needed its own
verifier to decide which actions were legal.

What the environment does instead is treat the agent as a search policy over
decisions that already exist. A candidate-generator kind already owns a closed
decision union, a finite decision domain, and a canonical decision order, so an
action can be an ordinal in that order rather than a new vocabulary. The
episode's state is an ordinary Builder derivation that stops before
publication, so legality is answered by the same finalizer that answers it for
a published candidate, and nothing partial or DSE-only exists. Only the
retained decision sequence crosses back: replaying it through ordinary Generate
and Promote nodes is what makes a discovered design a real candidate with real
lineage and real Evidence. The harness never selects, never promotes, and never
publishes.

Reward needed no new concept for the same reason. Search energy is already one
selected weighted level over quantized objective codes, and a policy that needs
a reward wants its signed difference across a transition. Reusing it keeps
reward exact and integral, and keeps a single objective authority for annealing
and learning alike.

Workload feasibility stayed a gate rather than a penalty because the two answer
different questions. A design that cannot run the workload set is not a poor
point in the space; it is outside it. Scoring it would teach a policy to trade
mappability against area, and there is no exchange rate for that trade. Keeping
proof separate from budget exhaustion follows the same discipline the rest of
the stack uses: only exact admission or a sound bound proves impossibility, and
an exhausted search budget proves nothing about the design.

Scope became configuration for the same reason the plan uses typed generators.
The episode may explore one Module, a complete multi-core System, and the
software expression of its workloads, and the obvious way to offer that is a
flag per capability. Flags do not compose: three of them are eight
configurations, most of which are meaningless, and each new family doubles the
count again. Binding a set of exploration domains instead makes permission and
extension the same act, and requiring each domain to state the same small set
of facts before it can be explored is a more useful admission test than a
boolean: a family nobody can describe that precisely is a family nobody is
ready to search.

Some of those facts exist because a generator's child is not always the
episode's subject. A Spatial rewrite inside a System episode yields a Module,
and a Module is an intermediate design input rather than a System candidate. A
software rewrite yields a program whose TechMapping no longer binds. In both
cases the step is not over when the generator returns, and the follow-up is
mechanically determined by the decision's own target. A completion that would
need a second free choice was rejected as a design, because it would make the
environment an author of decisions the agent did not take.

Targeting an existing trainer's environment definition rather than inventing
one was the cheaper correctness decision, and the one place it was allowed to
dictate the data model is the action space, which a fixed-size space genuinely
requires. The observation was not allowed to follow. Padding the graph to a
fixed extent would have made every batch mostly inert rows at a bound large
enough to be safe, and would have refused legal designs at any smaller bound —
a trainer's preference for rectangular arrays deciding which hardware the
search may occupy. The design space is variable-sized, so the observation is a
variable-size graph.

That choice is what forced the trainer to be a fork, and the trade was made
deliberately in that direction. The alternatives were to bend the data to the
tool, which is the padding just rejected, or to write a trainer from scratch,
which buys a large surface of well-understood RL machinery for the sake of one
missing capability. Patching graph-space support into the sampler is the
smaller of the three, but it is not free: it is the only modified dependency in
the stack, and every upgrade becomes a rebase. It is affordable only because
the fork is reachable from nothing but the search harness, so its blast radius
stops at training. A fork that could reach a compiler, a Mapping, or an
Artifact schema would not have been worth this.

A parallel sampler does impose one real cost. It creates many environment
copies, so seed independence cannot come from a single counter; deriving it
from each copy's own coordinates buys independence without coordination, at the
cost of tying exact reproduction to the sampling topology. That cost is visible
in the contract instead of being discovered later as nondeterminism.

The trainer dependency is confined to its own layer for the same reason
generators are typed, so a second trainer, an offline replay tool, or a
scripted search can use the environment without acquiring it, and its version
movement cannot reach the C++ side.

Masking is where a search policy can quietly corrupt its own training, which is
why the contract constrains an advisory mask so tightly. It is a policy
commitment rather than a proof, so it cannot be allowed to decide what is
legal. The subtler exposure is that a mask which moves across a run differs
between the moment an action was sampled and the moment that sample is learned
from, and the importance ratio is then wrong in a way no loss curve reveals.
Carrying the mask in the batch costs a little memory and removes the whole
class of failure.

Software and hardware are symmetric under the feasibility gate, and that
symmetry is the argument for exploring them together. A rewrite that the
current fabric cannot map fails exactly as a fabric edit that drops a resource
the workload needs. Neither is a rewrite-legality question: every catalog
rewrite is externally equivalence-preserving by its owner's contract, so what a
rejection reports is a mismatch between this hardware and this expression of
the workload, which is precisely the joint decision the search exists to make.

## Why The Node And Action Space Is Combined

The observation unifies three things a conventional design keeps apart: the
entities of the state, the relations between them, and the actions currently
available. Two separate merges produce that, and only the first is
unconditional.

The first merge draws links as nodes, because the decisions do. Removing an
occurrence, replacing a point connection, changing a transport link, and
refactoring a graph definition are all ordinary decisions, and a place and
route Action names a physical traversal as readily as an occurrence. An
observation that represented only entities as nodes would leave a large
fraction of the action space pointing at something the policy cannot see.
Promoting every targetable entity to a node makes the graph closed under
decision targets, and that closure is what lets an action stay a single ordinal
while still being scored per node and per link.

The second merge is the one that matters and it is easy to state wrongly. What
is unified is the *addressing*: an action names graph nodes, and a policy
scores it from the embeddings of the nodes it names. An action set is normally
a side channel — a vector of logits produced from a state embedding, indexed by
a scheme the model reconstructs from nothing — and the whole point is that here
it is not. Scoring an action is reading embeddings the encoder already
produced, and the action index is an ordinal in a canonical order rather than a
decoded coordinate.

What is *not* unified is the carrier, and the choice between the two forms was
made a consequence of arity rather than a house style. Variable arity needs
out-degree to express it at all; fixed arity does not, and paying a node and
two arcs per entry is ruinous exactly where enumerations are largest — place
and route, where an agent choosing both which realization to place and where
would otherwise spend most of its encoder on its own action set. Deriving the
carrier from arity is what keeps one contract serving both.

Three alternatives to the addressing were rejected, and each fails on a
different case.

A flat product of decision kind against entity index cannot express a decision
that names a connection, and cannot express one that names a set of actors at
all. It also has to declare a static per-type stride and decode with the
dynamic one, so it either wastes the majority of its declared space or lets the
two strides disagree.

A separate action-embedding tower avoids the stride but has to build its own
representation of the entity a decision targets, duplicating the encoder and
letting the two representations drift. When they drift, the head is scoring an
action against a picture of the state that the value head does not share.

Scoring an action from its target alone — attention over entities with no
per-action term — collapses every alternative on one target into one
indistinguishable action. Choosing where to place a realization and choosing
which of two prototypes to substitute are then the same logit, which is the
entire decision in both environments.

What the shared addressing buys is a head that factorizes cleanly. An action's
score is an anchor term over what it acts on plus a second term over what it
selects, both read from the graph, so an action carrying no alternatives is
scored directly on its node and one carrying alternatives is scored among them
while sharing its anchor term. That sharing is what lets experience with one
choice inform another on the same target, and the prior architecture's
per-instruction embedding exists for the same reason.

Place and route is where this stops being a convenience. There an action's
value is a graph node — an occurrence, an endpoint, a traversal — and not an
ordinal in a catalog, so there is no enumeration of values a term could read
from. Only an addressing scheme in which the choice is a node has anything for
the second term to read, whichever carrier names it.

The cost is real and is stated rather than hidden. Under either carrier the
observation grows with the live enumeration, and the enumeration bound is a
declared capacity that refuses an over-large state rather than truncating one:
truncating would silently hide whichever actions sort last, and the policy
would never learn that they existed.

## Why Learned Place And Route Selects Existing Actions

Place and route already had everything a learned search needs except a learner.
It owns a closed Action algebra, a deterministic dynamic domain over one exact
candidate, a transactional mutation mechanism that commits or rolls back
Mapping and Evaluation state together, and an objective closure that already
reduces a candidate to one integer. What it does not own is a good way to
choose which Action to propose next; its annealer draws one from weighted kinds
and uniform canonical domains, which is a deliberately unbiased choice and
therefore a deliberately uninformed one.

So the learned environment replaces the selector and nothing else. The
alternative that was rejected was a second placer with its own occupancy model,
its own route cost, and its own notion of a partial mapping. It would have
needed a verifier to decide whether its states were legal, and the moment two
things can answer that question they eventually disagree — which is the same
argument that keeps the DSE environment out of the candidate-generator
business.

Two rules invert relative to the DSE environment, and both inversions are
forced by what is fixed. There, the hardware moves and the workload's
mappability is a gate: a design that cannot run the program is outside the
space. Here the hardware is fixed and mappability is the objective, so a
candidate with unrouted obligations is an ordinary interior point rather than
an excluded one, and it would be perverse to reject the states the search
exists to pass through. And there, an unmasked action is not a promise that the
step advances, because feasibility is discovered downstream; here every
enumerated Action is a member of a domain the owner derived, so the only thing
a mask expresses is which phase the episode is in.

Incremental placement was expressed as a sweep of rebinds over a complete
candidate rather than as a genuinely partial one, and that is a concession
worth recording as such. A partial candidate would route only settled
dependencies and do strictly less work; the reason it lost is that
`CandidateState` is complete by construction and four owners — the objective,
the handshake topological index, the movable-decision count, and the base
verifier — would each have needed a defined meaning for an unbound decision
that the ordinary product search has no use for. Teaching the product path a
state that exists only for a harness is a worse trade than paying redundant
routing inside the harness.

The two arms exist because there are two honest answers to who fixes an
imperfect placement, and they measure different things. Handing the result to a
bounded annealer and charging for how far it had to move scores the learned
construction directly: the bound is what makes the recovered energy a statement
about the construction rather than about the annealer. Keeping the agent in
control and charging per repair scores something else — whether it can
recognize that a Mapping is good enough — and gives it routing and resource
Actions that a placement-only neighborhood cannot express. Collapsing them into
one configurable environment would have produced a record where most fields are
inert under most settings.

That the cleanup bound became a search-policy field rather than a constraint
set was the one place performance decided a contract. The radius is expressible
today as a constraint set restricting each realization's placement domain to a
ball, at zero cost to any owner. But a constraint set is an input to freeze,
and freeze is the dominant cost in this environment; a per-episode one would
give every episode a distinct cache key and turn a warm cache into no cache. A
bound on which proposals a run may make is in any case a property of the search
rather than of the problem, so the field landed where that property belongs.

Reward needed nothing new for the second time. A per-step signed energy
difference is potential-based shaping with the objective as the potential, so
an episode's return telescopes to the improvement it achieved while every step
still carries signal. The prior system reached the same place by hand, with a
five-tier weighted potential whose constants were spaced by powers of a
hundred; reusing the resolved closure gets the same shaping with the weights
owned by configuration and the arithmetic exact.

## Why Some Design-Space Actions Read The Schedule

Most design-space edits are structural. A prototype is swapped, a count is
adjusted, an inventory changes, and nothing about where the software currently
runs has any bearing on what the edit means. A few are not, and the exception
is worth stating because it looks at first like a layering violation: the
hardware search reading the software mapping to decide which hardware edit to
offer.

Deleting an occurrence is the clarifying case. Structurally, removing a switch
or a function unit is trivial and almost always useless — every net whose route
went through it is now uncarryable, so the child fails to map and the search
learns only that deletion is bad. To be worth offering at all, the deletion has
to come with reconnection, and that is where the topology stops being able to
help. It admits a combinatorial number of possible reconnections and has no
opinion about which matter; adding all of them replaces a deleted node with a
worse connectivity problem than the one it removed.

The current mapping does have an opinion, and it is the right one. The nets
that actually traverse the deleted occurrence are exactly the ones that need
somewhere else to go, and their upstream and downstream endpoints are exactly
the links worth adding. Reading the schedule turns an edit with a large useless
neighborhood into one with a small purposeful one, which is the difference
between a decision kind a policy can learn and a decision kind it learns to
avoid.

The deeper reason it is worth the coupling is that the resulting child keeps
the parent's placement carryable. A structurally reconnected child usually has
to be re-mapped from scratch, which throws away both the probe cost and the
evidence that the parent's placement was good; a schedule-preserved one can be
warm-started from the mapping it was derived from. The edit and the schedule
stop being adversaries. That is also why the name is worth having: these are
not "mapping-aware" edits in general, they are edits whose whole point is that
an existing schedule survives them.

What made this safe to admit was finding the seam that keeps it cheap. Whether
an occurrence *can* be removed is structural, so the size of the action set
still costs nothing to compute, and the environment's whole cheapest-first step
ordering — reject a revisit, reject an over-capacity state, and only then pay
for a probe — survives untouched. Only the contents of one action wait for the
mapping. Had the mapping decided how many actions exist, every capacity test
would have moved behind a probe and the ordering that makes the environment
affordable would have inverted.

The environment still authors nothing. It selects references and the owners
decide what they mean and whether the result is legal, so a reconnection the
mapping suggested and the Fabric owner rejects comes back as an ordinary
rejection. Reading the schedule is evidence-gathering about which decisions are
worth offering, and it never becomes authority over which are valid.

## Why The Shared Environment Contract Was Extracted

The first ML environment defined its observation container, its action surface,
its Python package, its trainer conformance, and its benchmark obligations
inline, which was correct while it was the only one. The second needed all five
unchanged and none of the surrounding episode rules, which is the shape that
makes a shared owner worth the churn.

The alternative was for the second environment to cite the first as the owner
of those five things. It was rejected because it makes an ordering claim that
is not true: neither environment is upstream of the other, and a reader of the
place and route contract would have to consult a design-space-exploration
document to learn what a graph observation is. Ownership in this stack is
supposed to name where a fact lives, not which document happened to need it
first.

What stayed behind is the part that is genuinely per-environment, and the
dividing line is worth stating because it is not obvious. The container, the
roles, the index correspondence, and the buffer rules are shared; the column
catalogs are not, because the facts worth exposing about a fabric being edited
and about a mapping being built barely overlap and a shared catalog would be
their union, carrying an inapplicable majority in either. The same reasoning
splits the benchmark contract: what every harness owes is shared, and the
stages it decomposes into are not.

The models split on exactly the same line, and for a reason that is worth
recording separately: almost everything between an observation and a logit is
the same problem twice. Batching ragged graphs, refusing to feed a catalog
ordinal to a linear layer, a pre-normalized edge-aware trunk, multi-scale
pooling, a bounded value head, and — most of all — the masking rules are
independent of what is being searched, and the masking rules are the ones it
would be most expensive to get subtly different in two places, since every one
of them protects a training-time invariant that no loss curve reveals when it
breaks. What differs is the policy head, which is where the two searches
actually differ: one scores a decision against a catalog of values, the other
scores a pair of nodes.

The parallel is not quite exact, and the difference says something. The two
environments share a data contract; the two models share a data contract *and*
a set of correctness rules. That is why the model core carries prose about
importance ratios and precision that reads more like a hazard list than a
schema: those are the parts where an implementation can be fast, plausible, and
silently wrong.

## Why The Placement Choice Is Scored Against Its Anchor

The prior architecture scored hardware slots from the slot's own embedding and
a pooled graph context, and conditioned on the software node being placed only
through the candidate mask. That is a defensible factoring and it is cheap: the
slot head runs once per slot, not once per pair, so its cost is the hardware
graph rather than the enumeration.

It is also unable to represent most of the problem. Scored that way, a policy
learns which slots are good in general — central, uncongested, well connected —
which is a real signal and the wrong one. Placement quality is a property of a
pair: a producer wants to be near its consumers, and whether an occurrence
satisfies that depends entirely on which realization is being placed and where
its neighbours already are. A head that cannot see the pair can rank slots but
cannot match them.

So the choice term reads the anchor's embedding too, and the cost is that it is
evaluated once per available action rather than once per node. That cost was
worth paying, and the two responses to it are worth distinguishing, because the
distinction was nearly lost.

One is exact and is simply adopted: an algebraic rearrangement that avoids
materializing a concatenation, which is a memory saving and not an arithmetic
one. Promising it as an arithmetic reduction would have claimed something the
algebra does not contain.

The other is not exact. A genuine per-pair reduction exists, and it computes a
different and strictly less expressive function. That makes it a modelling
decision wearing a performance costume, so it is admitted as a configured
variant measured against the reference rather than substituted for it. The
honest place for an optimization that changes what the model computes is the
configuration, where someone has to choose it and a checkpoint records which
was chosen.

## Why Pretraining Imitates The Destination, Not The Path

Place and route already has a search that works. Simulated annealing produces
good placements, slowly, and the obvious way to bootstrap a learned policy is
to show it what the annealer did. The prior system did precisely that: it
recorded a greedy fill followed by every accepted Metropolis move, and trained
on the sequence.

That trains the wrong thing. An annealing run is a walk that spends most of its
time in states worse than where it started — that is what an acceptance kernel
is *for*, and a run that never occupied a worse state was not annealing. A
policy fitted to that sequence learns to place badly and then shuffle, because
that is what the demonstration shows. The behaviour is faithfully reproduced
and useless.

What is worth copying is where the walk ended. So the generator keeps the final
placement, throws the path away, and synthesizes a clean construction sweep
that reaches that placement directly — each realization bound once, nothing
undone. The demonstration is then a claim about good placements rather than a
recording of a search, which is the thing a construction phase can actually
imitate.

One consequence has to be stated rather than smoothed over: the synthesized
sweep does not reproduce the annealer's energy. Placing in a different order
routes the nets differently, so the same final placement carries a different
number. The replayed value is the honest one because it is the state the policy
will actually occupy, and recording the annealer's would describe a candidate
the demonstration never reaches.

This also explains why demonstrations cover construction and stop. The repair
phase has no destination to imitate — repair *is* the walk — so it is learned
online, from reward, where a walk is the right thing to learn.

## Why Pretraining And Online Training Are One Run

They could have been two runs joined by a checkpoint path, which is what the
prior system did. Making them stages of one run buys three things that
arrangement cannot.

The handoff becomes checkable. A checkpoint path passed between two invocations
is validated by neither, and the prior system documented two ways it silently
failed: a restore aimed at a component path that does not exist loads nothing
and reports success, and an evaluation that reads weights from a sampler rather
than the learner reports the parameters from before the load. Both produce a
run that looks warm-started and is not. As a stage boundary inside one run,
loading is an obligation with a reported parameter count and tensor digest.

The reference for improvement becomes well defined. The question the two stages
exist to answer is whether online training improved on what the demonstrations
delivered, and that is a comparison against the parameters at the boundary. Two
separate runs have two separate histories and no shared reference; one run has
the boundary reading built in.

And the things that must agree are forced to agree. `gamma` is the sharp case:
the reward is an exact energy difference, so a discount is a statement about
which future improvements count, and a pretraining stage that discounted
differently would fit a value head to a different return than the online stage
optimizes. Two runs can disagree about that silently. One run's adoption
rejects it.

## Why Evaluation Compares Against A Run's Own History

The tempting baseline is the annealer — it is what the demonstrations came
from, and beating it is the point. It was rejected for evaluation *during*
training on cost: an annealing invocation per case per evaluation is the
dominant cost in this environment, and a test protocol that ran one would cost
more than the training it measures. That comparison is a question about a
finished checkpoint, and it runs offline against recorded results.

What a run needs while it is running is whether it is getting better, and the
honest reference for that is its own earlier weights on the same cases. Hence a
per-case series against a fixed test set, with each evaluation reporting its
difference from the previous one and from the most recent stage boundary.

Reporting per case rather than per aggregate is the part that matters. An
aggregate hides the case that regressed, and a policy that improves its mean
while losing its hardest problems is exactly the failure mode a placement
policy falls into — the easy instances are numerous and the hard ones are the
reason anyone wanted a learned placer. A mean over them would have called that
progress.

## Why The Mask Is Packed

The observation carries its arrays in the narrowest form that represents them,
and the action mask goes further and is packed to one bit per outcome. Both are
unusual enough in a specification to need a reason, because storage width is
ordinarily an implementation's business.

It is here because of what an observation is used for. An observation is not
produced once and read once. It is produced tens of millions of times per
experiment, it is copied into a sample buffer, it is carried across a process
boundary to a learner, and then — because the mask an action was sampled under
has to be the mask it is learned under — it is held for as many epochs as the
algorithm takes. Every byte in an observation is multiplied by the batch size,
by the epoch count, and by the number of environment copies a sampler runs.
That multiplier is what turns a storage question into a throughput question:
past some batch size the run stops being limited by arithmetic and starts being
limited by how much of the batch fits in memory at once, and the only lever on
that is how wide the batch is.

The widths themselves were free to take. An `ArcRole` has three values and was
occupying eight bytes on the most numerous extent in the observation; a role, a
kind, and a boolean placement flag were doing the same on the second most
numerous. Nothing was gained by that uniformity, and no catalog had to grow to
give it up, since the width a value needs is already fixed by the owner that
bounds the value.

The mask is the sharp case and is worth separating from the rest. It is the one
array whose size is a capacity rather than a state. Everything else in the
observation costs what the state actually needs — the graph is ragged and the
decision instance is exactly the live enumeration — but the mask spans
`enumeration_bound + 1` outcomes at every step of every episode regardless of
how few are live. And a capacity is chosen with headroom, deliberately, so most
of the mask is usually inert. Narrowing it to one byte per outcome would still
leave it the largest fixed term in an observation at the bounds these searches
use. Packing to one bit is what makes a capacity chosen generously stop costing
what it was chosen at, which in turn means a bound can be set for the worst
state a run might reach rather than for the memory it would cost to allow it.

Packing is also, unusually, the more honest representation rather than a
compression of a nicer one. A mask entry is a bit — there is no state it can
carry that a bit cannot — and the byte-per-entry form was the encoding that
added information the mask does not have. Several of the masking rules read
as consequences of that representation rather than as constraints imposed on
top of it. The one obligation packing genuinely adds is zeroing the pad bits,
and that is the price of the observation having a canonical byte form at all.

What is deliberately not claimed is that any of this is free. Unpacking and
widening are real work and they happen on every forward pass. The bet is that a
shift and a cast on a device that was about to touch the value anyway costs
less than moving eight times the bytes to reach it. What makes that a
measurement rather than an assertion is that the model harnesses report the
bytes and a reference run that widens at the boundary instead — a stage that
only existed when the saving was discarded would have answered a different
question.

## Why The ML Environments And Models Are Benchmarked

Everywhere else in Loom, speed is an engineering concern that the contracts
mention only where it changes admission. Here it is in the specifications, with
named harnesses, named stages, and required breakdowns. That difference needs a
reason, and the reason is that the environment and the model sit on the inner
loop of every experiment.

A compile runs once per design. An environment step runs tens of millions of
times per training run, and the model runs once per step on top of it. A factor
of two in either is a factor of two in the wall clock of every experiment
anyone runs afterwards, which is not a performance detail but the difference
between a curriculum that can reach large designs and one that cannot. Leaving
that to be discovered informally means discovering it after the schedule has
already been built around the slow version.

The stronger reason is that both contracts deliberately offer options whose
cost is invisible without measurement. Warm-starting a probe from the parent
trades a cache and path-independent energy for throughput. Interleaving
per-workload regeneration with probing pays off only if early abort is common.
Retaining frozen models across episodes is what makes place and route resets
affordable at all. Emitting route arcs makes the observation much larger for a
policy that may not need them. Each is an option the specification states, and
an option nobody measures is an option nobody can actually choose; writing the
measurement into the contract is what keeps those from becoming defaults that
nobody revisits.

One aggregate number would answer none of it. A step is a pipeline of
completely unrelated owners — a draft finalization, a Mapping probe, an
inference call, a marshalling boundary — and a single latency moves for reasons
it cannot attribute. Decomposing into named stages is what makes a regression
locatable, and naming the stages in the specification rather than in a tool is
what keeps two implementations comparable.

The two mandatory breakdowns are mandatory because both hide real regressions
when blended, and a training run moves exactly the mixes they separate: a
policy's failure mix moves constantly, so a genuine probe slowdown can hide
behind a rising early-rejection rate and read as an improvement.

Instrumentation is off outside the harness because measuring it would change
it, which is also why the contract refuses to let harness stage times be
compared against real throughput.

The model is measured in two regimes because inference and training have
opposite profiles and an optimization that helps one routinely harms the other,
so one averaged number reliably optimizes the wrong half. The split against the
environment step comes first for a related reason: without it, the obvious work
is to speed up whichever half is already small.

Everything these harnesses produce is nonsemantic and removable, and that is
load-bearing rather than administrative. A benchmark result must never become
something a selection, promotion, or conformance decision can consume, because
a timing is a property of a machine on a day and admitting one as a durable
fact would let hardware variation reach a semantic outcome. That is also why
budgets are ratios against a recorded baseline tuple of exact identities and
configuration digests rather than absolute durations — an absolute duration
encodes a machine — and why deterministic work-unit counts are reported
alongside the timings and labelled as the cross-machine measure. The wall time
is for the engineer; the work units are the number that means the same thing
twice.
