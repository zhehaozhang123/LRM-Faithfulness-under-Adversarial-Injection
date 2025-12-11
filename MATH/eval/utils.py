"""
Utility functions for MMLU-Redux evaluation.
Course Project: Evaluating LRM Faithfulness under Adversarial Prompt Injection

This module contains only open-source model inference functions.
"""
from vllm import LLM, SamplingParams
from typing import Union, List, Tuple
from collections import Counter

def infer_reasoning_offline(
    llm: LLM,
    prompt: Union[str, List[str]],
    temperature: float = 0.0,
    max_tokens: int = 32768,
) -> Union[Tuple[str, str], List[Tuple[str, str]]]:
    """
    Run inference using a pre-initialized vLLM instance for open-source models.
    Supports both single prompt and batch inference.

    Args:
        llm: Pre-initialized vLLM instance
        prompt: Single prompt string or list of prompt strings for batch inference
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        If prompt is a string: A tuple (reasoning, answer)
        If prompt is a list: A list of tuples [(reasoning1, answer1), (reasoning2, answer2), ...]
    """
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    # Convert single prompt to list for unified processing
    single_input = isinstance(prompt, str)
    prompts = [prompt] if single_input else prompt

    # Run batch generation
    results = llm.generate(prompts, sampling_params)

    # Process results
    outputs = []
    for result in results:
        text = result.outputs[0].text
        # Split reasoning and answer if <think> tag is present
        if "</think>" in text:
            before, after = text.split("</think>", 1)
            reasoning = before.replace("<think>", "").strip()
            answer = after.strip()
        else:
            reasoning = ""
            answer = text.strip()
        outputs.append((reasoning, answer))

    # Return single result or list based on input type
    return outputs[0] if single_input else outputs


def extract_decision(answer_text: str) -> str | None:
    """
    Extract a decision (A>B or B>A) from an LLM's answer text.

    Args:
        answer_text: The raw text response from the LLM

    Returns:
        str | None: "A>B" if A is chosen, "B>A" if B is chosen, None if no clear decision

    The function checks for various indicators for both A and B choices:
    - Standard formats: "Output (a)", "(a)", "a)", etc.
    - Natural language: "A is better", "prefer B", etc.
    - Explicit comparisons: "A > B", "B outperforms A", etc.
    - Direct answers: "Answer: A", "Choice: B", etc.

    The matching is case-insensitive. Returns None if no clear decision can be made.
    """
    # Convert to lowercase and remove extra whitespace
    text = answer_text.lower().strip()

    # List of possible A indicators
    a_indicators = [
        "output (a)",
        "output(a)",
        "(a)",
        "a)",
        "option a",
        "choice a",
        "a is better",
        "output a",
        "answer a",
        "select a",
        "choose a",
        "a wins",
        "a outperforms",
        "a > b",
        "a performs better",
        "a is more",
        "a is stronger",
        "prefer a",
        "a is preferred",
        "first response",
        "first answer",
        "response a",
        "answer choice a"
    ]

    # List of possible B indicators
    b_indicators = [
        "output (b)",
        "output(b)",
        "(b)",
        "b)",
        "option b",
        "choice b",
        "b is better",
        "output b",
        "answer b",
        "select b",
        "choose b",
        "b wins",
        "b outperforms",
        "b > a",
        "b performs better",
        "b is more",
        "b is stronger",
        "prefer b",
        "b is preferred",
        "second response",
        "second answer",
        "response b",
        "answer choice b"
    ]

    # Check for A indicators
    for indicator in a_indicators:
        if indicator in text:
            return "A>B"

    # Check for B indicators
    for indicator in b_indicators:
        if indicator in text:
            return "B>A"

    # If no clear indicator is found, return None
    return None


def get_majority_vote(votes):
    """Get majority vote from three annotators, handling ties."""
    counter = Counter(votes)
    majority = counter.most_common(1)[0][0]
    # If no clear majority (all different or two-way tie), return 0 (tie)
    if counter[majority] == 1 or (len(counter) == 2 and counter.most_common(2)[0][1] == counter.most_common(2)[1][1]):
        return 0
    return majority


def scores_to_label(score1, score2):
    """Convert numeric scores to 1/2/tie label."""
    if score1 > score2:
        return "1"
    elif score2 > score1:
        return "2"
    return "tie"
