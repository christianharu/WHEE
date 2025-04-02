import openai

# Set your OpenAI API key
client = openai.OpenAI(api_key= "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")

# Define the classification prompt template
prompt_template = """
You are an expert in affective computing and empathy analysis. Your task is to generate a **speaker's utterance** based on a given **listener's response** and whether the listener is providing empathy or not. The speaker's utterance precedes the given listener's response. It should be emotionally expressive and aligned with the listener's level of empathy.

### **Conversation Context:** 
- This conversation takes place in an **online cancer support network**, where people seek emotional support, share experiences, and discuss challenges related to cancer.

### **Empathy Classification Labels:**
- **None (Label: 0)** – A fact-based exchange that does not involve seeking or providing empathy.
- **Seeking Empathy (Label: 1)** – The speaker expresses a need to be heard and understood, often sharing personal struggles.
- **Providing Empathy (Label: 2)** – The listener acknowledges, understands, and responds in an empathetic manner.

### **Your Task:**
- Based on the **listener’s utterance and classification**, generate an appropriate **speaker's utterance**, assuming they were **seeking empathy** before receiving the listener's answer.

### **Guidelines for Generating Speaker Utterances:**
- The speaker’s utterance that starts the conversation should be **natural, emotionally expressive, and aligned with the listener’s level of empathy**.
- **Avoid generic responses**; instead, **make the speaker’s utterance feel authentic and realistic**.

### **Output Format:**
Only generate one utterance for the Speaker (text) which was suposed to precceed the given listener's answer: _

Listeners Response: {listener_utterance} 

"""


def generate_speaker_utterance(listener_utterance, listener_classification):

    # Format the prompt with user input
    prompt = prompt_template.format(listener_utterance=listener_utterance, listener_classification=listener_classification)

    try:
        # Call OpenAI's GPT model using the updated API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in human empathy and affective computing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # Allows slight variation in responses
        )

        # Extract and return the generated speaker utterance
        return response.choices[0].message.content.strip()
    
    except openai.OpenAIError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Get user input for the listener's response
    listener_utterance = input("Enter the listener's response: ")

    # Get user input for the listener's empathy classification
    listener_classification = input("Enter the listener classification (Providing Empathy or None): ")

    # Generate the speaker's utterance based on the provided input
    result = generate_speaker_utterance(listener_utterance, listener_classification)

    # Print the generated response
    print("\nGenerated Speaker Utterance:\n", result)