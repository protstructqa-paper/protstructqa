"""ProtStructQA baselines: model adapters + prompts + scoring + runners.

L0: zero-shot single-shot.
L1: grammar-constrained execution-feedback (xgrammar over Lark grammar +
    rejection sampling on program-execution errors).
L2: ReAct-style multi-turn with tool calls.

All three use the same scoring module against the same gold answers.
"""
