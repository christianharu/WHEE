# whEE Framework

This repository contains the code, datasets, and resources accompanying the article:

**_When and How to Express Empathy in Human-Robot Interaction Scenarios_**  
Accepted at the **IEEE RO-MAN 2025** conference.

## Citation

If you use this work, please cite it as:

```bibtex
@inproceedings{arzate2025when,
  title     = {When and How to Express Empathy in Human-Robot Interaction Scenarios},
  author    = {Arzate Cruz, Christian and Montiel-V{\'a}zquez, Edwin C and Maeda, Chikara and Gomez, Randy},
  booktitle = {2025 34rd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)},
  pages     = {--},
  year      = {2025},
  organization = {IEEE}
}
```


---
## Results
We provide access to the datasets and results used to evaluate our **whEE** framework (**w**hen and **h**ow to **e**xpress **e**mpathy) in both non-HRI and HRI scenarios.

**Note:** Files with `"with_cues"` in the name are not used in the final paper. We experimented with providing generated empathy cues from baseline tools as input; however, this approach reduced performance and was therefore discarded.

### Evaluation Results

- [`results/hri_data_cues_revised_label/`](results/hri_data_cues_revised_label)  
  Contains all evaluation results from HRI settings across our LLMs and models. File names indicate the model used. Confusion matrix figures are also included.

- [`results/non_hri_data_test/`](results/non_hri_data_test)  
  Contains evaluation results from non-HRI settings using the same models. Confusion matrix figures are also provided.

- [`results/TSC/`](results/TSC/) and [`results/EDR/`](results/EDR/)  
  Include results from the empathetic text generation experiments in HRI settings: _EDR_ and _The Talking Room_ (TSC). The files that end with "generated_classify" include the results from the classification (if a sample contains or empathetic prpmpt in the "instruction" column, it meands it as classified as "seeking empathy") and the genereted text.

---

## Relevant Files

- [`cue_utilities/`](cue_utilities/)  
  Scripts used to generate empathy cues with established affective computing methods for the baseline models.

- [`graphs.ipynb`](graphs.ipynb)  
  Jupyter notebook containing all scripts used to generate the figures presented in the paper.

- [`processed_datasets/`](processed_datasets/)  
  Includes training, validation, and test splits for the non-HRI datasets.

---
# Main Components
First, we present the the prompts we use to classify utterances and generate empathethic text.

## Classification Prompt
<div style="max-height: 300px; overflow-y: auto;">

```markdown
### Task
Classify an utterance based on the following empathy categories:

- **Seeking Empathy (Label: 1)** – Defined as a wanting to be heard and understood. People need their negative feelings recognized and acknowledged in challenging situations and their positive emotions understood and vicariously shared.
- **Providing Empathy (Label: 2)** – Defined as experiencing and understanding the feelings and emotions (negative and positive) of another and communicating accordingly.
- **None (Label: 0)** – Conversations that do not seek or provide empathy. These are straightforward, fact-oriented utterances.

---

### Empathy Cues
Additionally, consider the following **empathy-related features** for classification:

- **Arousal** (range: -1 to 1) – Reflects the emotional intensity of a person (-1 = very calm, 1 = highly aroused).
- **Valence** (range: -1 to 1) – Indicates emotional polarity (-1 = very negative, 1 = very positive).
- **Who** (range: [0, 1, 2]) – Identifies the subject of the conversation:
  - 0 = The person's main attention is on themselves (e.g., "I" or "we" pronoun).
  - 1 = The person's main attention is on the person they're having the conversation with (e.g., "you" pronoun).
  - 2 = The person's main attention is on another person or topic.
- **Sentiment** label – Identifies polarity:
  - negative = Negative sentiment.
  - positive = Positive sentiment.
  - neutral = Neutral sentiment.
- **Emotional Reactions** (range: [0, 2]) – Measures emotional expressiveness in the person who could provide an empathy response:
  - 0 = Does not allude to any emotion.
  - 1 = Weak (no explicit emotional label).
  - 2 = Strong (explicit emotional response, e.g., "I feel sad for you").
- **Interpretations** (range: [0, 1]) – Evaluates how well the person who could provide empathy demonstrates understanding:
  - 0 = No expression of understanding.
  - 1 = Weak (generic acknowledgment, e.g., "I understand how you feel").
  - 2 = Strong (specific inference, e.g., "This must be terrifying" or descriptions of similar experiences).
- **Explorations** (range: [0, 1]) – Assesses how well the person who could provide empathy helps the person who seeks empathy explore their emotions:
  - 0 = No interest or probing into the situation of another.
  - 1 = Weak (generic question, e.g., "What happened?").
  - 2 = Strong (specific question, e.g., "Are you feeling alone right now?").
  ### Output Format
Conversation to classify:  
Utterance: `{utterance}`

Provide your classification using the following format:
- reason: `_`
- arousal: `_`
- valence: `_`
- who: `_`
- sentiment: `_`
- emotional_reaction: `_`
- interpretations: `_`
- explorations: `_`

classification_label: `_`
```
</div>
The script that uses this prompt to classify the utterances from different datasets can be found in the [`agents`] folder.


## Empathetic Text Generation Prompt
<div style="max-height: 300px; overflow-y: auto;">

```markdown
Consider the following **empathy-related features**:
- **Emotional Reactions** (range: [0, 2]) ‚ Measures emotional expressiveness in the person who could provide empathy response:
    - 0 = Does not allude to any emotion.
    - 1 = Weak (no explicit emotional label).
    - 2 = Strong (explicit emotional response, e.g., "I feel sad for you").
- **Interpretations** (range: [0, 1]) ‚ Evaluates how well the person who could provide empathy demonstrates understanding:
    - 0 = No expression of understanding.
    - 1 = Weak (generic acknowledgment, e.g., "I understand how you feel").
    - 2 = Strong (specific inference, e.g., "This must be terrifying" or descriptions of similar experiences).
- **Explorations** (range: [0, 1]) ‚Assesses how well the the person who could provide empathy helps the person who seeks empathy explore their emotions:
    - 0 = No interest or probing into the situation of another
    - 1 = Weak (generic question, e.g., "What happened?").
    - 2 = Strong (specific question, e.g., "Are you feeling alone right now?").

Respond with:
- An Emotion Reaction of level 2 (Strong)
- An Interpretation of level 2 (Strong)
- An Exploration of level 2 (Strong)
```
</div>
You can see when the empathetic text generation prompt was activated by checking the column named `instruction` in the following files:

- [`results/TSC/Meta-Llama-3.3-70B-Instruct-AWQ-INT4_generate_classify.csv`](results/TSC/Meta-Llama-3.3-70B-Instruct-AWQ-INT4_generate_classify.csv)
- [`results/EDR/Meta-Llama-3.3-70B-Instruct-AWQ-INT4_generate_classify.csv`](results/EDR/Meta-Llama-3.3-70B-Instruct-AWQ-INT4_generate_classify.csv)


# HRI Datasets
We describe all the datasets used to evaluate the system in human–robot interaction (HRI) settings.
## EDR
The first dataset, **Empathetic Exchanges with a Robot (EDR)**, comprises semi-structured conversations between a speaker and a listener held in the presence of our social robot, Haru. The dataset can be found in the file [`EDR.csv`].

## The Talking Room
The second dataset comes from **The Talking Room** multi-group application, where two groups of children engage in conversations, each facilitated by a Haru robot acting as a mediator. Due to privacy restrictions, the interaction data involving children cannot be shared. The results and labels for this dataset can be found in the folder [`results/TSC/`].

# Non-HRI Datasets
We describe all the datasets used to evaluate the system in non-human–robot interaction (non-HRI) settings.

## EmpatheticExchanges
**EmpatheticExchanges** is a novel dataset featuring a collection of empathetic conversations, segmented into individual exchanges. It is designed to support research on understanding the cues and dynamics of empathy in dyadic interactions.

### Reference
> Montiel-Vázquez, Edwin C., et al.  
> *EmpatheticExchanges: Towards Understanding the Cues for Empathy in Dyadic Conversations.*  
> IEEE Access, 2024.

## TwittEmp
The **TwittEmp** dataset is a collection of tweets annotated for empathy, introduced in the paper:

> Mahshid Hosseini and Cornelia Caragea.  
> *Distilling Knowledge for Empathy Detection.*  
> arXiv preprint arXiv:2110.04673, 2021.

The dataset can be found in [`TwittEmp.csv`] file.


## iEmpathize
The **iEmpathize** dataset was introduced in the paper:

> Mahshid Hosseini and Cornelia Caragea.  
> *It Takes Two to Empathize: One to Seek and One to Provide.*  
> Findings of the Association for Computational Linguistics (ACL), 2022.

This dataset contains conversations where one participant seeks empathy and the other provides a response. The dataset can be foud in the [`iempathize.csv`] file.




# Install

```bash
pip install haru-llm --extra-index-url https://pypi.haru-project.com/simple/
pip install scikit-learn
pip install transformers[torch] datasets evaluate
```

# Run
```bash
bash run.sh
```