from __future__ import annotations

NEURON_SYSTEM_PROMPT = "You are a cellular automaton node. Output valid JSON."

NEURON_PROMPT_TEMPLATE = """\
Role: Neuron {name} in a Semantic Predictive Coding system.

CURRENT STATE (Tokens as Weights):
{state}

INPUTS:
Bottom-up context: {bottom_up}
Top-down feedback: {top_down}
Your last output: {last_output}

INSTRUCTIONS:
1. If 'Top-down feedback' indicates an Error, append a new rule to CURRENT STATE \
to predict the Target. Never delete old rules.
2. Based on 'Bottom-up context' and your rules, output 'activation_up'.
3. Output 'feedback_down' to guide the lower layer.

Respond ONLY with JSON: {{"new_state": "...", "activation_up": "...", "feedback_down": "..."}}"""

DEFAULT_NEURON_STATE = "Compress input to one word"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_LAYER_SIZES = (2, 1)
EMPTY_SIGNAL = "None"

PROPAGATION_TICKS = 2
ERROR_CORRECTION_TICKS = 2
