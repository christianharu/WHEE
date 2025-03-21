"""Classify the dataset."""

import json
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from haru_llm.agents import ConversationalAgent, StructuredAssistantAgent
from pydantic import BaseModel, Field
from tqdm import tqdm

DEFAULT_INSTRUCTION = "".strip()
EMPATHY_INSTRUCTION = """
Consider the following **empathy-related features**:
- **Emotional Reactions** (range: [0, 2]) – Measures emotional expressiveness in the person who could provide empathy response:
    - 0 = Does not allude to any emotion.
    - 1 = Weak (no explicit emotional label).
    - 2 = Strong (explicit emotional response, e.g., "I feel sad for you").
- **Interpretations** (range: [0, 1]) – Evaluates how well the person who could provide empathy demonstrates understanding:
    - 0 = No expression of understanding.
    - 1 = Weak (generic acknowledgment, e.g., "I understand how you feel").
    - 2 = Strong (specific inference, e.g., "This must be terrifying" or descriptions of similar experiences).
- **Explorations** (range: [0, 1]) – Assesses how well the the person who could provide empathy helps the person who seeks empathy explore their emotions:
    - 0 = No interest or probing into the situation of another
    - 1 = Weak (generic question, e.g., "What happened?").
    - 2 = Strong (specific question, e.g., "Are you feeling alone right now?").

Respond with:
- An Emotion Reaction of level 2 (Strong)
- An Interpretation of level 2 (Strong)
- An Exploration of level 2 (Strong)
""".strip()

def generate(
    dataset_file: str,
    agent_file: str,
    inputs_cols: str | list[str],
    *,
    n: int | None = None,
) -> None:
    """Classify the dataset."""
    dataset_file = Path(dataset_file)
    inputs_cols = [inputs_cols] if isinstance(inputs_cols, str) else inputs_cols

    # Read dataset
    ds = pd.read_csv(dataset_file)
    ds = ds.sample(n) if n is not None else ds

    # Define LLM agent
    agent = ConversationalAgent.from_yaml(agent_file)
    model_id = agent.model_id.split("/")[-1]

    # Generate function
    def generate_fn(row: pd.Series) -> dict:
        inputs = {k: v for k, v in row.to_dict().items() if k in inputs_cols}
        outputs = agent.generate(inputs, instruction=row["instruction"], add_to_history=False)
        response = list(outputs)[-1].response if outputs is not None else ""
        return response

    # Generation
    ds['instruction'] = ds.apply(lambda x: EMPATHY_INSTRUCTION if x["classification_label"] == 1 else DEFAULT_INSTRUCTION, axis=1)
    tqdm.pandas(desc=f"Generate with {model_id}")
    ds["llm_utterance"] = ds.progress_apply(generate_fn, axis=1)

    # Save results
    ds.to_csv(dataset_file.parent / f'{dataset_file.stem}_generate.csv', index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(generate)
