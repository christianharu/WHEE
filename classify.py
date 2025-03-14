"""Classify the dataset."""

import json
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
    save_dir: str,
    *,
    use_structured_outputs: bool = False,
    n: int | None = None,
) -> None:
    """Classify the dataset."""
    dataset_file = Path(dataset_file)
    inputs_cols = [inputs_cols] if isinstance(inputs_cols, str) else inputs_cols
    save_dir = Path(save_dir)

    # Read dataset
    ds = pd.read_csv(dataset_file)
    ds = ds.sample(n) if n is not None else ds

    # Define LLM agent
    if use_structured_outputs:
        agent = StructuredAssistantAgent.from_yaml(agent_file)
    else:
        agent = ConversationalAgent.from_yaml(agent_file)

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
    tqdm.pandas(desc=f"Generate with {agent.model_id}")
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
            lambda x: x.split("reason: ")[1].split("\n")[0].strip()
        )
        ds["arousal"] = ds["response"].apply(
            lambda x: x.split("arousal: ")[1].split("\n")[0].strip()
        )
        ds["valence"] = ds["response"].apply(
            lambda x: x.split("valence: ")[1].split("\n")[0].strip()
        )
        ds["who"] = ds["response"].apply(
            lambda x: x.split("who: ")[1].split("\n")[0].strip()
        )
        ds["sentiment"] = ds["response"].apply(
            lambda x: x.split("sentiment: ")[1].split("\n")[0].strip()
        )
        ds["emotional_reaction"] = ds["response"].apply(
            lambda x: x.split("emotional_reaction: ")[1].split("\n")[0].strip()
        )
        ds["interpretations"] = ds["response"].apply(
            lambda x: x.split("interpretations: ")[1].split("\n")[0].strip()
        )
        ds["explorations"] = ds["response"].apply(
            lambda x: x.split("explorations: ")[1].split("\n")[0].strip()
        )
        ds["classification_label"] = ds["response"].apply(
            lambda x: x.split("classification_label: ")[1].split("\n")[0].strip()
        )

    # Save results
    save_file = save_dir / dataset_file.stem / f"{agent.model_id}.csv"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(save_file, index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(classify)
