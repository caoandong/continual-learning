# Coding Principles

Detailed coding philosophy based on Casey Muratori's "Semantic Compression" approach. Reference this when planning significant implementations.

## Code Quality Standards

### Type Safety
- Strict TypeScript: enable `strict` mode
- **No `any`**: Avoid `any` at all cost; use `unknown` and narrow with type guards
- **No `as` casting**: Avoid type assertions; use explicit type-checks and narrowing instead
  - Bad: `const user = data as User`
  - Good: `if (isUser(data)) { /* data is now User */ }`
- Explicit types: annotate function signatures, avoid inference for public APIs
- Runtime validation: use Zod schemas at system boundaries (API inputs, external data)
- Prefer discriminated unions over optional fields for variant types

### Immutability
- Default to `readonly` for object properties and array types
- Use `as const` for literal values and tuples
- Never mutate function arguments; return new values
- Prefer `ReadonlyArray<T>` over `T[]` for function parameters

### Statelessness
- Prefer pure functions: same inputs always produce same outputs
- Isolate side effects to the edges (entry points, I/O boundaries)
- Be critical when introducing state; document why it's necessary
- When state is needed, make it explicit and minimal

### Function Design
- **Size limit**: Functions should fit on one screen (~40 lines including comments)
- **Argument limit**: Maximum 2 arguments; use an options object for more
- **Single responsibility**: One function does one thing
- **No classes**: Prefer pure functions and plain data objects
- **Early returns**: Fail fast, reduce nesting

### Flat Code — Maximum 2 Levels of Nesting

Deeply nested code is hard to read, hard to test, and hides bugs. **Every indentation level inside a function body costs readability exponentially.** Enforce a hard limit: no more than 2 levels of nesting inside any function. When you hit the limit, extract the inner block into its own named function.

**The rule**: If a `for`/`while` loop or `if` branch contains another loop or branch, the inner body must be a call to a separate function.

#### Why this matters

- Each nesting level multiplies the mental state a reader must track
- Extracted functions are independently testable and reusable
- Flat code reads like a sequence of steps, not a maze of conditions
- Enforced by linters: Python `max-nested-blocks = 3` (Ruff), TypeScript `cognitive-complexity: 15` (ESLint)

#### TypeScript

```typescript
// BAD: 4 levels deep — logic is buried and untestable
async function processOrders(orders: readonly Order[]): Promise<void> {
  for (const order of orders) {
    if (order.status === "pending") {
      for (const item of order.items) {
        if (item.stock > 0) {
          await reserveStock(item.id, item.quantity);
          await notifyWarehouse(order.id, item.id);
        } else {
          await backorderItem(order.id, item.id);
        }
      }
    }
  }
}

// GOOD: max 2 levels — each concern is a named, testable function
async function fulfillItem(orderId: string, item: OrderItem): Promise<void> {
  if (item.stock > 0) {
    await reserveStock(item.id, item.quantity);
    await notifyWarehouse(orderId, item.id);
    return;
  }
  await backorderItem(orderId, item.id);
}

async function fulfillOrder(order: Order): Promise<void> {
  if (order.status !== "pending") return;
  await Promise.all(order.items.map((item) => fulfillItem(order.id, item)));
}

async function processOrders(orders: readonly Order[]): Promise<void> {
  await Promise.all(orders.map(fulfillOrder));
}
```

#### Python

```python
# BAD: 4 levels deep — nested loops with conditionals
def export_user_reports(users: list[User]) -> list[Report]:
    reports = []
    for user in users:
        if user.is_active:
            for account in user.accounts:
                if account.balance > 0:
                    report = generate_report(user, account)
                    reports.append(report)
    return reports

# GOOD: max 2 levels — each inner concern is its own function
def exportable_accounts(user: User) -> Iterator[Account]:
    """Yield accounts that qualify for reporting."""
    return (a for a in user.accounts if a.balance > 0)

def build_user_reports(user: User) -> list[Report]:
    """Generate reports for one active user's qualifying accounts."""
    if not user.is_active:
        return []
    return [generate_report(user, account) for account in exportable_accounts(user)]

def export_user_reports(users: list[User]) -> list[Report]:
    """Generate reports for all active users with positive-balance accounts."""
    return [
        report
        for user in users
        for report in build_user_reports(user)
    ]
```

#### Flattening techniques (quick reference)

| Technique | When to use |
|---|---|
| **Early return / continue** | Guard clause eliminates an outer `if` wrapping the entire body |
| **Extract inner function** | A loop body or branch body is ≥3 lines — give it a name |
| **Comprehension / map** | Transform-and-filter over a collection (replaces loop + conditional + append) |
| **Lookup table** | Multiple `if/elif` branches selecting a value — use a dict/Map instead |
| **Promise.all / gather** | Independent async operations inside a loop — parallelize and flatten |

### Naming
- **No abbreviations**: `getUserById` not `getUsrById`, `repository` not `repo`
- **Descriptive**: Names should explain intent without reading implementation
- **Consistent vocabulary**: Use the same term for the same concept everywhere
- **Boolean prefixes**: `is`, `has`, `should`, `can` for boolean variables/functions
- **No `_` prefix**: Don't use leading underscores to signal "private" or "internal". In Python, use `__all__` to control public exports. In TypeScript, don't export what shouldn't be public. The name itself should be clear enough without a visibility sigil.

### Constants and Generality
- **No hard-coded constants scattered in logic**: Magic numbers and string literals buried in functions are invisible assumptions that break silently when context changes.
- **Prefer general, robust solutions over micro-fixes**: A fix that works because you tuned a threshold to pass one test case is not a fix — it's overfitting. Solve the underlying problem so the solution holds across all cases, not just the ones you've seen.
- **When constants are unavoidable, centralize them**: Define a single constants file (or config object) per domain as the source of truth. Every consumer imports from that one place. Changing a value should require exactly one edit in one file.
  - Bad: `if (score > 0.73)` deep in a function body
  - Bad: The same threshold defined in three files with different values
  - Good: `import { RELEVANCE_THRESHOLD } from "./constants"` used everywhere
- **Name constants by intent, not value**: `MAX_RETRY_ATTEMPTS` not `THREE`. The name should explain *why* the value exists.

### Async and Parallelization
- Parallelize independent operations with `Promise.all`
- Never `await` in a loop when operations are independent
- Handle errors explicitly; don't let promises silently fail
- Use `AbortSignal` for cancellable operations

## Core Loop: Implement → Verify → Compress → Expand
### 1. Start Concrete
Write the direct solution for your specific case. No "reusable architecture" until you have at least two real instances.

### 2. Verify Immediately
Run tests/checks after each change. Don't refactor unverified code.

### 3. Compress on Repetition
Extract shared semantics when:
- Logic is duplicated a second time
- Same patterns appear (call sequences, control flow, data wrangling)
- Next feature requires many manual edits

### 4. Expand with Vocabulary
Once compressed, new features become expressions using the established primitives.

## Compression Techniques

**Context Objects**: Bundle repeated parameter clusters into a context that travels together.

**Finalize Derived Values**: Compute totals/sizes at the end rather than maintaining parallel counters.

**Semantic Helpers**: Wrap repetitive low-level calls with meaningful names (`emit_field`, `append_row`).

**Layer, Don't Replace**: Add convenience layers on top of primitives. Don't break the general form when adding convenience.

**Table-Driven**: When you have many similar behaviors, encode variation as data. Parameterize handlers instead of duplicating functions.

**Match Reality in Types**: If a concept is pass/fail/N-A, don't force it into boolean.

## External APIs

### FAL AI Models

Before writing interfaces or types for a FAL AI model, **always fetch the model's llm.txt first**. This file contains the canonical input/output schemas.

**URL pattern**: `https://fal.ai/models/{model-id}/llms.txt`

Example: For model `fal-ai/kling-video/v2.6/pro/image-to-video`, fetch:
```
https://fal.ai/models/fal-ai/kling-video/v2.6/pro/image-to-video/llms.txt
```

Use the schemas from llm.txt to define your TypeScript types and Zod validators. Do not guess or assume parameter names/types.

## Logging

This repo uses a custom structured logger from `@video-agent/core`. **Never use `console.log`, `console.warn`, `console.error`, or `console.debug` directly.** All logging goes through `createLogger`.

### Creating a Logger

One module-level logger per file, named after the component:

```typescript
import { createLogger } from "@video-agent/core";

const log = createLogger("component-name");
```

Component names should be short, lowercase, hyphen-separated identifiers that describe the module's role: `"runner"`, `"fs"`, `"http"`, `"convex-client"`, `"computer-loop"`, `"videogenJobs"`, `"sandboxPool"`.

### Log Levels

- `log.debug()` — Verbose diagnostics (hidden by default; `minLevel: "info"` is the default)
- `log.info()` — Normal operational events (startup, completion, state transitions)
- `log.warn()` — Unexpected but recoverable situations
- `log.error()` — Failures that need attention

### Structured Context, Not String Interpolation

Pass data as a context object (key-value pairs), not interpolated into the message string. This keeps messages greppable and context machine-parseable.

```typescript
// Good: structured context
log.info("File uploaded", { path: "/tmp/video.mp4", size: 1024, duration: 45 });
log.error("Request failed", { status: 500, endpoint: "/api/run" });

// Bad: string interpolation
log.info(`File uploaded: /tmp/video.mp4 (1024 bytes, 45s)`);
log.error(`Request failed with status 500 at /api/run`);
```

Context values must be primitives (`string | number | boolean | null | undefined`). For errors, convert to string: `{ error: String(error) }`.

### Timing Helpers

Use built-in timing methods instead of manual `Date.now()` arithmetic:

**`log.time(label)`** — Single operation duration:
```typescript
const endTimer = log.time("database query");
const result = await db.query(sql);
endTimer(); // → "database query 45ms"
```

**`log.trace(name)`** — Multi-phase operations:
```typescript
const trace = log.trace("request");
await authenticate();
trace.phase("auth");     // → request:auth +45ms (45ms)
await fetchData();
trace.phase("fetch");    // → request:fetch +120ms (165ms)
trace.end();             // → request:done 165ms
```

**`log.stepper()`** — Sequential steps:
```typescript
const step = log.stepper();
step("Loading config");   // → [1] Loading config
step("Connecting to DB");  // → [2] Connecting to DB
step("Starting server");   // → [3] Starting server
```

### Child Loggers

Use `log.child(context)` to create a logger with persistent context that appears on every log entry:

```typescript
const requestLog = log.child({ requestId: "abc-123", userId: "user-456" });
requestLog.info("Processing request"); // includes requestId and userId on every line
```

### Configuration Options

```typescript
// Suppress logs below a certain level
const log = createLogger("fixtures", { minLevel: "warn" });

// Send all output to stderr (useful for CLIs where stdout is reserved)
const log = createLogger("cli", { stderrOnly: true });

// Disable console output (logs still go to sinks)
const log = createLogger("runner", { writeToConsole: false });
```

### What NOT to Do (TypeScript)

- **No `console.log`** — It bypasses structured logging, sinks, and level filtering
- **No string interpolation in messages** — Put variable data in the context object
- **No manual timing** — Use `log.time()` or `log.trace()` instead of `Date.now()` arithmetic
- **No creating loggers inside functions** — One module-level instance per file
- **No logging secrets** — Never log API keys, tokens, or credentials

### Python Logging (videogen)

The `videogen` package uses Python's standard `logging` module for diagnostic logs and a custom structured event system for operational telemetry.

#### Module-Level Logger

One logger per module using `__name__`:

```python
import logging

logger = logging.getLogger(__name__)
```

#### Log Levels

Same semantics as TypeScript:
- `logger.debug()` — Verbose diagnostics (hidden at default INFO level)
- `logger.info()` — Normal operational events
- `logger.warn()` — Unexpected but recoverable situations
- `logger.error()` — Failures that need attention

#### `%s` Formatting, Not f-strings

Use `%s` placeholders in log calls. Python's logging module evaluates these lazily — the string is only formatted if the message passes the level filter. f-strings are always evaluated, wasting cycles at suppressed levels.

```python
# Good: lazy formatting
logger.info("[extract_segments] Extracted %d segments to %s", len(segments), temp_dir)
logger.warning("[unify] Attempt %d/%d failed: %s", attempt, max_attempts, type(exc).__name__)

# Bad: eager f-string (always evaluated even if level is suppressed)
logger.info(f"[extract_segments] Extracted {len(segments)} segments to {temp_dir}")
```

#### Section Tags in Messages

Prefix log messages with a bracketed tag identifying the logical section. This makes logs greppable across large pipeline runs:

```python
logger.info("[pipeline_v2] Starting video-to-timeline pipeline")
logger.debug("[extract_segments] Starting: video=%s segments=%d", video_path, len(segments))
logger.info("[unify] Calling unification model: %s", settings.model)
```

#### Structured CLI Events

For operational telemetry (not diagnostics), use `emit_cli_event` from `videogen.shared.cli.logging`. These produce structured JSON consumed by the trace system:

```python
from videogen.shared.cli.logging import emit_cli_event

emit_cli_event(
    event="deep_research.search_complete",
    payload={"query": query, "result_count": len(results)},
    level="info",
    component="videogen.research",
)
```

For domain-specific events, wrap `emit_cli_event` in a thin helper:

```python
def emit_research_event(
    *, event: str, payload: JsonObject | None = None, level: EventLevel = "info",
) -> None:
    emit_cli_event(
        event=f"deep_research.{event}",
        payload=payload,
        level=level,
        component="videogen.research",
    )
```

#### Run Context via Environment Variables

Run identity is threaded through environment variables, not function parameters. The structured event system reads these automatically:

- `TRACE_RUN_ID` — Primary run identifier
- `TRACE_PARENT_SPAN_ID` — Span hierarchy for nested operations
- `RUN_TOKEN` — Token-as-key credential for Convex proxy

When these are set, `emit_structured_event` switches from legacy JSON format to the `__trace` protocol and routes events to the Convex trace sink (buffered batch POST). When absent, CLI event wrappers choose their local sink behavior (human-readable by default, JSON when explicitly configured). Code should **never** read these directly — the structured event layer handles detection and format switching.

#### Logging Configuration

For scripts and entry points, configure the root logger explicitly:

```python
def configure_logging(log_path: Path, *, verbose: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # Suppress noisy third-party libraries
    for noisy in ("httpx", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
```

Libraries and pipeline modules should **never** call `logging.basicConfig()` or configure the root logger — that's the entry point's job.

#### What NOT to Do (Python)

- **No `print()`** — Use `logger` for diagnostics, `emit_cli_event` for structured telemetry
- **No f-strings in log calls** — Use `%s` placeholders for lazy evaluation
- **No `logging.basicConfig()` in library code** — Only entry points configure the root logger
- **No creating loggers inside functions** — One module-level `logger` per file
- **No logging secrets** — Never log API keys, tokens, or credentials
- **No reading `TRACE_RUN_ID` / `RUN_TOKEN` directly** — Let the structured event layer handle run context

## Anti-Patterns

- Designing reusable architecture from a single example
- Optimizing for "clean" or "elegant" when it raises total cost
- Complex machinery for small gains
- Scattering related logic across files such that patterns become invisible
- **Overfitting on test cases**: Tweaking a constant or adding a special-case branch to make a failing test pass without understanding *why* it fails. If you need a magic number to fix something, you don't understand the problem yet.
- **Scattered constants**: The same conceptual value (threshold, limit, timeout) defined in multiple places. When someone changes one copy and not the others, you get silent inconsistencies that are brutal to debug.

## Decision Framework

Evaluate tradeoffs by **total lifetime cost** (write, debug, modify, integrate, perform), not by style commandments.
