# Evaluation ToolRunner

This specification defines Loom Evaluation's synchronous local-process
execution primitive. The typed C++ API in `include/Evaluation/ToolRunner.h` is
the single source of truth for the implemented value model.

## Ownership

`runTool(const ToolInvocation &) -> llvm::Expected<ToolRunOutcome>` executes
one already-resolved local tool invocation. It is shared by external
Evaluation model adapters, but it is not an EDA semantic layer, workflow
engine, scheduler, lease manager, artifact store, or remote execution backend.

The synchronous caller owns these prerequisites before calling `runTool`:

- select the exact tool and non-secret tool binding identity;
- allocate the scratch working directory;
- materialize every input artifact and bind its Common `ArtifactIdentity` to
  an absolute local path;
- acquire every resource and license lease;
- keep the real leases alive until `runTool` returns;
- keep any state captured by the cancellation query alive until `runTool`
  returns.

`ToolInvocation` carries only non-secret resource and license lease binding
identities. Those identities are provenance facts, not lease handles, and
`runTool` neither proves nor changes lease ownership.

## Invocation Contract

`ToolInvocation` contains:

- one exact non-secret tool binding identity;
- an absolute executable path;
- the literal POSIX `argv`, including caller-selected `argv[0]`;
- an explicit environment overlay applied to the runner process environment;
- an absolute, existing scratch working directory;
- already-materialized input artifact identities and absolute paths;
- declared output roots expressed as relative paths under scratch;
- an optional non-negative timeout;
- an optional synchronous cancellation query;
- non-secret resource and license lease binding identities.

The cancellation query is observational authority supplied by the caller. It
must be cheap and must not throw. The runner polls it while the invocation is
active. It does not create a cancellation service or a second execution
context.

Environment overlay values are execution-only data. They are never copied into
`ToolRunOutcome`. The captured streams remain verbatim tool output; adapters
must not arrange for tools to print credentials or other secrets.

## Preflight

Preflight completes before any process is spawned. It rejects:

- an empty tool binding identity;
- a missing, non-regular, non-executable, or non-absolute executable path;
- an empty `argv` or strings that cannot be represented by POSIX `execve`;
- a missing, non-directory, non-writable, or non-absolute scratch path;
- missing input paths or non-absolute input paths;
- invalid or duplicate environment overlay names;
- negative timeouts;
- empty resource or license binding identities;
- empty or absolute output paths, lexical parent traversal, duplicate output
  paths, existing symlink prefixes that resolve outside scratch, and output
  roots that cannot be snapshotted before launch.

Preflight and runner-internal failures are returned as `llvm::Error`. A failure
after process launch begins is represented by `ToolRunOutcome` when the raw
attempt facts are available.

## Local Execution

The implementation invokes the executable directly with `execve`. It never
constructs or invokes a shell command, so shell metacharacters in `argv` remain
literal bytes.

Before fork, the calling thread blocks signals. The private supervisor resets
inherited signal dispositions and clears its mask before creating the tool, so
ignored `SIGCHLD`, ignored termination signals, blocked signals, and inherited
handlers cannot alter tool execution or supervisor waiting. The tool closes
all descriptors above the standard descriptor set except the close-on-exec
launch-status descriptor. Internal pipe descriptors are first moved above
stdin, stdout, and stderr so initially closed standard descriptors are safely
reused.

The tool leads a dedicated Linux session and process group. Because the
invoked executable is both session leader and process-group leader, it cannot
move itself with `setsid` or `setpgid`. A private reservation process is
created in that group before `execve`, and the tool is not released until the
reservation has published the PGID directly to both the supervisor and the
caller.

A private Linux subreaper supervisor remains outside the tool session. It
commits timeout or cancellation, sends the stop, continue, and graceful
termination signals, and reaps group members. The supervisor checks for
completed leader state, briefly stops the group, and commits an interrupt only
after observing the leader stopped rather than exited. It then sends `SIGTERM`
and `SIGCONT`. This prevents a leader that completed before the commit from
being reclassified.

Final-signal ownership is published with the reserved process group. The
supervisor is the sole normal owner. It transitions ownership before sending
the final negative-PGID `SIGKILL`, publishes completion, and only then reaps
the reservation. Every negative-PGID signal first verifies the published
reservation is still live. If the supervisor dies before publishing
completion, the caller transfers ownership only after observing that death
and performs the emergency final signal. A completed or released ownership
record prevents any later caller signal. This preserves the PGID even when
inherited `SIGCHLD` handling or another reaper would otherwise release it
early.

Background members of the launched group are cleaned up after a normal leader
exit.

Separate nonblocking pipes capture stdout and stderr concurrently. Capture is
unbounded in total and preserves stream separation, but each descriptor has a
fixed per-loop drain quota so one continuous stream cannot starve the other
stream, cancellation, timeout, or process observation. The runner does not
parse either stream.

The containment boundary is the launched process group. A descendant that
creates another session or process group is outside that boundary. After the
supervisor completes bounded cleanup and closes its result channel, the parent
performs one final bounded drain and closes capture descriptors instead of
waiting for escaped descendants that retained them. ToolRunner does not build
an independent process-tree tracker or claim ownership of arbitrary daemons.

## Outcome Contract

`ToolRunStatus` distinguishes:

- `LaunchFailure`: process setup or `execve` failed after preflight;
- `Exited`: the tool exited normally, including a nonzero exit code;
- `Signaled`: the tool terminated from a signal without timeout or
  cancellation;
- `TimedOut`: the invocation timeout initiated group termination;
- `Cancelled`: the caller's cancellation query initiated group termination;
- `InfrastructureFailure`: process launch began, but the runner could not
  produce a complete supervised result or complete process-group cleanup.

`ToolRunOutcome` contains only raw execution facts:

- the status and optional exit code, termination signal, or launch errno;
- a launch diagnostic when launch failed;
- an optional infrastructure diagnostic when status is
  `InfrastructureFailure`;
- captured stdout and stderr as separate byte strings;
- a deterministic lexicographically sorted inventory of regular files created
  or changed under declared output roots by this run;
- an optional raw inventory diagnostic;
- non-semantic wall-clock start and end timestamps;
- the non-secret tool, resource lease, and license lease binding identities.

Declared output roots are snapshotted before launch and after process cleanup.
Inventory paths are normalized relative paths under scratch. Missing declared
outputs, unchanged preexisting files, and files outside declared roots are
absent. Directory roots are enumerated recursively without following symlinks.
An escaping post-run symlink or post-run traversal failure leaves the process
status, streams, timestamps, and identities intact, returns no partial
inventory, and records the failure in `inventoryDiagnostic`. The inventory is
not an artifact manifest and carries no digest, schema, metric, finding, or
report meaning.

Start and end timestamps are wall-clock provenance only. They do not define
timeout behavior, duration ordering, or any semantic result. Timeout decisions
use a monotonic clock.

## Exclusions

ToolRunner does not:

- parse tool reports or assign Evaluation semantics;
- construct metrics, findings, Evidence, or artifact identities;
- finalize or persist artifacts;
- select tools or resolve model configuration;
- acquire, renew, release, schedule, or queue resource and license leases;
- schedule Evaluation DAGs, workflows, or design-space exploration;
- retry invocations;
- provide plugin, remote, cluster, container, or shell backends.

Tool-specific adapters consume `ToolRunOutcome`, parse their own reports, and
construct the typed Evaluation result owned by their model contract.
