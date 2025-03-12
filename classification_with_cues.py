import openai
import pandas as pd
import argparse
import os

# Set your OpenAI API key
client = openai.OpenAI(api_key= "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")

# Define the classification prompt template
prompt_template = """
Classify an utterance based on the following empathy categories:

- **Seeking Empathy (Label: 1)** – Defined as a need to be heard and understood. When people experience challenging situations, they need their feelings to be recognized and acknowledged.
- **Providing Empathy (Label: 2)** – Defined as experiencing and understanding the feelings of another and acting accordingly.
- **None (Label: 0)** – Conversations that do not seek or provide empathy. These are straightforward, fact-oriented exchanges.

Additionally, consider the following **empathy-related features** for classification:

- **Arousal** (range: -1 to 1) – Reflects the emotional intensity of a person (-1 = very calm, 1 = highly aroused).
- **Valence** (range: -1 to 1) – Indicates emotional polarity (-1 = very negative, 1 = very positive).
- **Who** (range: [0, 1, 2]) – Identifies the subject of the conversation:
  - 0 = The person's main attention is on themselves (e.g., "I" or "we" pronoun).
  - 1 = The person's main attention is on the person their having the conversation with (e.g., "you" pronoun).
  - 2 = The person's main attention is on another person or topic.
- **Sentiment** (range: [0, 1]) – Identifies polarity:
  - negative = Negative sentiment.
  - positive = Positive sentiment.
  - neutral = Neutral sentiment.
- **Emotional Reactions** (range: [0, 1]) – Measures emotional expressiveness in the person who could provide empathy response:
  - 0 = Weak (no explicit emotional label).
  - 1 = Strong (explicit emotional response, e.g., "I feel sad for you").
- **Interpretations** (range: [0, 1]) – Evaluates how well the person who could provide empathy demonstrates understanding:
  - 0 = Weak (generic acknowledgment, e.g., "I understand how you feel").
  - 1 = Strong (specific inference, e.g., "This must be terrifying" or descriptions of similar experiences).
- **Explorations** (range: [0, 1]) – Assesses how well the the person who could provide empathy helps the person who seeks empathy explore their emotions:
  - 0 = Weak (generic question, e.g., "What happened?").
  - 1 = Strong (specific question, e.g., "Are you feeling alone right now?").


  
Conversation to classify:  
Utterance: {utterance_classify}

Also consider for the classification the observed empathy feature values in the utterance:
arousal: {arousal_level}
valence: {valence_level}
sentiment: {sentiment_label}
emotional_reaction: {emotional_reaction_label}
interpretations: {interpretations_label}
explorations: {explorations_label}


Provide your classification using the following format:
reason: _
who: _ 
classification_label: _  
"""
def classify_conversation(model, utterance_classify, arousal_level, valence_level, sentiment_label, emotional_reaction_label, interpretations_label, explorations_label):
    """
    Sends a conversation to OpenAI API for classification based on empathy-related factors.
    """
    # Format the prompt with user input
    prompt = prompt_template.format(utterance_classify = utterance_classify, arousal_level = arousal_level, valence_level = valence_level, sentiment_label = sentiment_label, emotional_reaction_label = emotional_reaction_label, interpretations_label = interpretations_label, explorations_label = explorations_label)

    try:
        # Call OpenAI's GPT model using the updated API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in affective computing and empathy classification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Keep responses deterministic for consistency
        )

        # Extract and return the classification result
        return response.choices[0].message.content
    
    except openai.OpenAIError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Classify an utterance based on empathy-related factors.")
    parser.add_argument("--database", 
                        type=str, help="The name of the database to load",
                        default="hri_data_cues_revised_label.csv",)
    
    parser.add_argument("--model", 
                        type=str, help="The name LLM model to use",
                         default="gpt-4o-mini",) 
    
    args = parser.parse_args()


    current_folder = os.getcwd()
    database_path = os.path.join(current_folder, 'processed_datasets',args.database)
    model = args.model


    # Load the database
    database = pd.read_csv(database_path)


    # add new column to store the reason
    database["classification_label"] = ""
    database["reason"] = ""     




    # Iterate over the database
    for index, row in database.iterrows():
        # Get the text and label
        utterance = row["text"]
        arousal_level = row["arousal"]
        valence_level = row["valence"]
        sentiment_label = row["sentiment"]
        emotional_reaction_label = row["predictions_ER"]
        interpretations_label = row["predictions_IP"]
        explorations_label = row["predictions_EX"]


        print("\nIndex: ", index)

        # Get classification
        result = classify_conversation(model, utterance, arousal_level, valence_level, sentiment_label, emotional_reaction_label, interpretations_label, explorations_label)


        # get only the classification_label from result
        classification_label = result.split("classification_label: ")[1].split("\n")[0].strip()
        reason = result.split("reason: ")[1].split("\n")[0].strip()
        who = result.split("who: ")[1].split("\n")[0].strip()


        # Save the classification results
        database.at[index, "classification_label"] = classification_label
        database.at[index, "reason"] = reason
 

        # Save the updated database
        database.to_csv(f"{database_path.split('.csv')[0]}_{model}_classified.csv", index=False)            

    # The end
    print("\n\n Finished! Classification results saved to: ", f"{database_path.split('.csv')[0]}_{model}_classified.csv")