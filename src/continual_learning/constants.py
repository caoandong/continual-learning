from __future__ import annotations

NEURON_SYSTEM_PROMPT = (
    "You are a deterministic local optimizer for one neuron in a continual-learning network. "
    "Output valid JSON only. "
    "Treat state as memory, not as weights stored in your own parameters. "
    "Do not paraphrase labels or tokens; preserve exact symbols from the inputs and feedback."
)

NEURON_PROMPT_TEMPLATE = """\
Role: Neuron {name} in a local continual-learning system.

CURRENT STATE (Persistent Memory JSON):
{state}

INPUTS:
Bottom-up context: {bottom_up}
Top-down feedback: {top_down}
State update mode: {state_update_mode}
Your last output: {last_output}

INSTRUCTIONS:
1. Treat `CURRENT STATE` as the entire persistent memory. You are only the stateless update rule.
2. The memory may contain arbitrary general-purpose tokens and compressed experiences.
3. Keep `new_state` as valid JSON with this exact top-level shape:
   `{{"version":1,"summary_tokens":["..."],"memories":[{{"input_tokens":["..."],"output_text":"...","weight":1}}]}}`
4. `summary_tokens` should be a short compressed summary of the neuron's current memory.
5. `memories` should store compressed reusable traces and predictive mappings.
7. When `Top-down feedback` contains `Error: Target: X`, store the label exactly as `X`. Do not rename it, explain it, or wrap it.
8. `activation_up` must be either an exact stored label string or a compact token sketch built from the current input/state.
9. `feedback_down` must be `none` or a compact token sketch built from the current input/state.
10. Prefer compression over copying raw history; merge overlapping memories and keep the most predictive patterns.
11. If `State update mode` is `read_only`, keep `new_state` unchanged.

Respond ONLY with JSON:
{{"new_state": {{"version":1,"summary_tokens":["..."],"memories":[{{"input_tokens":["..."],"output_text":"...","weight":1}}]}}, "activation_up": "...", "feedback_down": "..."}}"""

DEFAULT_NEURON_STATE = (
    '{"memories":[],"summary_tokens":[],"version":1}'
)
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_LAYER_SIZES = (2, 1)
EMPTY_SIGNAL = "None"
RANDOM_STATE_SEED = 7
RANDOM_STATE_TOKEN_COUNT = 3
RANDOM_STATE_TOKEN_POOL = (
    "seed0",
    "seed1",
    "seed2",
    "seed3",
    "seed4",
    "seed5",
    "seed6",
    "seed7",
    "seed8",
    "seed9",
    "seed10",
    "seed11",
)
NEURON_STATE_VERSION = 1
MAX_PATTERN_TOKENS = 5
MAX_PROTOCOL_TOKENS = 4
MAX_MEMORY_PATTERNS = 8
MEMORY_MATCH_THRESHOLD = 0.45
MEMORY_MERGE_THRESHOLD = 0.5

PROPAGATION_TICKS = 2
ERROR_CORRECTION_TICKS = 2
SETTLING_BUFFER_TICKS = 2
MAX_CORRECTION_PASSES = 3
