"""Classify the dataset."""

import json
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from haru_llm.agents import ConversationalAgent, StructuredAssistantAgent
from pydantic import BaseModel, Field
from tqdm import tqdm


class ClassificationResult(BaseModel):
    """Classification result."""

    reason: str = Field(
        description="Step-by-step reasoning process.",
    )
    who: Literal[0, 1, 2] = Field(
        description="Predicted who field.",
    )
    classification_label: Literal[0, 1, 2] = Field(
        description="Predicted classification_label field.",
    )


def classify(
    dataset_file: str,
    agent_file: str,
    inputs_cols: str | list[str],
    save_file: str,
    *,
    use_structured_outputs: bool = False,
    n: int | None = None,
) -> None:
    """Classify the dataset."""
    dataset_file = Path(dataset_file)
    inputs_cols = [inputs_cols] if isinstance(inputs_cols, str) else inputs_cols

    # Read dataset
    ds = pd.read_csv(dataset_file)
    ds = ds.sample(n) if n is not None else ds

    # Define LLM agent
    if use_structured_outputs:
        agent = StructuredAssistantAgent.from_yaml(agent_file)
    else:
        agent = ConversationalAgent.from_yaml(agent_file)
    model_id = agent.model_id.split("/")[-1]

    # Generate function
    def generate_fn(row: pd.Series) -> dict:
        inputs = {k: v for k, v in row.to_dict().items() if k in inputs_cols}
        if use_structured_outputs:
            outputs = agent.generate_from_object(
                inputs, class_object=ClassificationResult
            )
            response = (
                json.dumps(outputs.structured_dict, indent=4)
                if outputs is not None
                else ""
            )
        else:
            outputs = agent.generate(inputs, add_to_history=False)
            response = list(outputs)[-1].response if outputs is not None else ""

        return response

    # Generation
    tqdm.pandas(desc=f"Generate with {model_id}")
    ds["response"] = ds.progress_apply(generate_fn, axis=1)

    # Process response
    if use_structured_outputs:
        ds["reason"] = ds["response"].apply(lambda x: json.loads(x).get("reason"))
        ds["who"] = ds["response"].apply(lambda x: json.loads(x).get("who"))
        ds["classification_label"] = ds["response"].apply(
            lambda x: json.loads(x).get("classification_label")
        )
    else:
        ds["reason"] = ds["response"].apply(
            lambda x: re.search(r"(?<=reason:\s)([^\n]*)", x).group(1)
            if re.search(r"(?<=reason:\s)([^\n]*)", x) is not None
            else None
        )
        ds["who"] = ds["response"].apply(
            lambda x: re.search(r"(?<=who:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=who:\s)([\d.]*)", x) is not None
            else None
        )
        ds["classification_label"] = ds["response"].apply(
            lambda x: re.search(r"(?<=classification_label:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=classification_label:\s)([\d.]*)", x) is not None
            else None
        )

    # Save results
    save_file = (
        Path(f"results/{dataset_file.stem}/{save_file.format(model=model_id)}")
        if use_structured_outputs is False
        else Path(
            f"results/{dataset_file.stem}/struct/{save_file.format(model=model_id)}"
        )
    )
    save_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(save_file, index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(classify)
