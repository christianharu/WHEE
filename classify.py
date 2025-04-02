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
    arousal: float = Field(
        description="Predicted arousal field.",
        ge=-1.0,
        le=1.0,
    )
    valence: float = Field(
        description="Predicted valence field.",
        ge=-1.0,
        le=1.0,
    )
    who: Literal[0, 1, 2] = Field(
        description="Predicted who field.",
    )
    sentiment: Literal["negative", "neutral", "positive"] = Field(
        description="Predicted sentiment field.",
    )
    emotional_reaction: Literal[0, 1, 2] = Field(
        description="Predicted emotional reaction field.",
    )
    interpretations: Literal[0, 1, 2] = Field(
        description="Predicted interpretations field.",
    )
    explorations: Literal[0, 1, 2] = Field(
        description="Predicted explorations field.",
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
        ds["arousal"] = ds["response"].apply(lambda x: json.loads(x).get("arousal"))
        ds["valence"] = ds["response"].apply(lambda x: json.loads(x).get("valence"))
        ds["who"] = ds["response"].apply(lambda x: json.loads(x).get("who"))
        ds["sentiment"] = ds["response"].apply(lambda x: json.loads(x).get("sentiment"))
        ds["emotional_reaction"] = ds["response"].apply(
            lambda x: json.loads(x).get("emotional_reaction")
        )
        ds["interpretations"] = ds["response"].apply(
            lambda x: json.loads(x).get("interpretations")
        )
        ds["explorations"] = ds["response"].apply(
            lambda x: json.loads(x).get("explorations")
        )
        ds["classification_label"] = ds["response"].apply(
            lambda x: json.loads(x).get("classification_label")
        )
    else:
        ds["reason"] = ds["response"].apply(
            lambda x: re.search(r"(?<=reason:\s)([^\n]*)", x).group(1)
            if re.search(r"(?<=reason:\s)([^\n]*)", x) is not None
            else None
        )
        ds["arousal"] = ds["response"].apply(
            lambda x: re.search(r"(?<=arousal:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=arousal:\s)([\d.]*)", x) is not None
            else None
        )
        ds["valence"] = ds["response"].apply(
            lambda x: re.search(r"(?<=valence:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=valence:\s)([\d.]*)", x) is not None
            else None
        )
        ds["who"] = ds["response"].apply(
            lambda x: re.search(r"(?<=who:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=who:\s)([\d.]*)", x) is not None
            else None
        )
        ds["sentiment"] = ds["response"].apply(
            lambda x: re.search(
                r"(?<=sentiment:\s)(negative|neutral|positive)", x
            ).group(1)
            if re.search(r"(?<=sentiment:\s)(negative|neutral|positive)", x) is not None
            else None
        )
        ds["emotional_reaction"] = ds["response"].apply(
            lambda x: re.search(r"(?<=emotional_reaction:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=emotional_reaction:\s)([\d.]*)", x) is not None
            else None
        )
        ds["interpretations"] = ds["response"].apply(
            lambda x: re.search(r"(?<=interpretations:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=interpretations:\s)([\d.]*)", x) is not None
            else None
        )
        ds["explorations"] = ds["response"].apply(
            lambda x: re.search(r"(?<=explorations:\s)([\d.]*)", x).group(1)
            if re.search(r"(?<=explorations:\s)([\d.]*)", x) is not None
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
