import openai

# Set your OpenAI API key
client = openai.OpenAI(api_key= "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")

# Define the classification prompt template
prompt_template = """
Classify a conversation based on the following empathy categories:

- **Seeking Empathy (Label: 1)** – Defined as a need to be heard and understood. When people experience challenging situations, they need their feelings to be recognized and acknowledged.
- **Providing Empathy (Label: 2)** – Defined as experiencing and understanding the feelings of another and acting accordingly.
- **None (Label: 0)** – Conversations that do not seek or provide empathy. These are straightforward, fact-oriented exchanges.

Additionally, consider the following **empathy-related features** for classification:

- **Arousal** (range: -1 to 1) – Reflects the emotional intensity of the speaker and listener (-1 = very calm, 1 = highly aroused).
- **Valence** (range: -1 to 1) – Indicates emotional polarity (-1 = very negative, 1 = very positive).
- **Who** (range: [0, 1, 2]) – Identifies the subject of the conversation of the speaker and listener:
  - 0 = The speaker or listener talks about themselves (e.g., "I" pronoun).
  - 1 = The speaker or listener talks about the listener (e.g., "you" pronoun).
  - 2 = The speaker or listener talks about another person or topic.
- **Sentiment** (range: [0, 1]) – Identifies polarity:
  - 0 = Negative sentiment.
  - 1 = Positive sentiment.
- **Emotional Reactions** (range: [0, 1]) – Measures emotional expressiveness in the listener’s response:
  - 0 = Weak (no explicit emotional label).
  - 1 = Strong (explicit emotional response, e.g., "I feel sad for you").
- **Interpretations** (range: [0, 1]) – Evaluates how well the listener demonstrates understanding:
  - 0 = Weak (generic acknowledgment, e.g., "I understand how you feel").
  - 1 = Strong (specific inference, e.g., "This must be terrifying" or descriptions of similar experiences).
- **Explorations** (range: [0, 1]) – Assesses how well the listener helps the speaker explore their emotions:
  - 0 = Weak (generic question, e.g., "What happened?").
  - 1 = Strong (specific question, e.g., "Are you feeling alone right now?").

Provide your classification using the following format:

Classification label (Speaker): _  
Classification label (Listener): _  
Reason: _  
Arousal (Speaker): _  
Valence (Speaker): _  
Arousal (Listener): _  
Valence (Listener): _  
Who (Speaker): _  
Who (Listener): _  
Sentiment (Speaker): _  
Sentiment (Listener): _  
Emotional Reaction (Listener): _  
Interpretations (Listener): _  
Explorations (Listener): _  

Conversation to classify:  
Speaker: {speaker_text}  
Listener: {listener_text}
"""
def classify_conversation(speaker_text, listener_text):
    """
    Sends a conversation to OpenAI API for classification based on empathy-related factors.
    """
    # Format the prompt with user input
    prompt = prompt_template.format(speaker_text=speaker_text, listener_text=listener_text)

    try:
        # Call OpenAI's GPT model using the updated API
        response = client.chat.completions.create(
            model="gpt-4",
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
    # User input for conversation
    speaker_text = input("Enter the speaker's message: ")
    listener_text = input("Enter the listener's response: ")

    # Get classification
    result = classify_conversation(speaker_text, listener_text)

    # Print the classification result
    print("\nClassification Result:\n", result)