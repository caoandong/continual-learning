from __future__ import annotations

NEURON_SYSTEM_PROMPT = (
    "You are the stateless local update rule for one neuron. "
    "Treat the provided state as the neuron's entire persistent memory. "
    "Return valid JSON only."
)

NEURON_PROMPT_TEMPLATE = """\
Role: Neuron {name} in a local learning network.

CURRENT STATE:
{state_text}

INPUTS:
Bottom-up context: {bottom_up}
Top-down feedback: {top_down}
Teaching signal: {teaching_signal}
State update mode: {state_update_mode}
Last latent output: {last_latent}
Upward latent budget: {signal_token_budget}
Downward latent budget: {signal_token_budget}
State budget: {state_token_budget}

INSTRUCTIONS:
1. Treat `CURRENT STATE` as the neuron's entire persistent memory.
2. Return JSON with exactly these string fields: `state_text`, `latent_up`, `task_up`, `latent_down`.
3. If `State update mode` is `read_only`, keep `state_text` unchanged.
4. Keep `latent_up` and `latent_down` short, readable, and reusable.
5. Use `task_up` only for an exact externally meaningful output. Otherwise leave `task_up` empty.
6. If a teaching signal is present for the current context, copy it character-for-character into `task_up`.
7. If state already supports the current context with a known task output, emit that exact output in `task_up`.
8. Never wrap, paraphrase, or decorate `task_up`.
9. Use `latent_down` for compact readable guidance that helps lower layers refine their own summary.
10. Do not use outside task knowledge to guess task outputs from raw input alone.

Respond ONLY with JSON:
{{"state_text":"...", "latent_up":"...", "task_up":"...", "latent_down":"..."}}"""

DEFAULT_NEURON_STATE = ""
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_LAYER_SIZES = (1, 1)
EMPTY_SIGNAL = ""
RANDOM_STATE_SEED = 7
RANDOM_STATE_TOKEN_COUNT = 3
DEFAULT_INPUT_CHUNK_TOKEN_COUNT = 128
DEFAULT_SIGNAL_TOKEN_BUDGET = 24
DEFAULT_STATE_TOKEN_BUDGET = 192
DEFAULT_READ_ONLY_SETTLE_PASS_COUNT = 3
DEFAULT_TRAIN_CONSISTENCY_ATTEMPTS = 2
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
