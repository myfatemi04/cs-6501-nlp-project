import openai

sys_prompt_1 = "You are a helpful assistant, generating a dataset of sentences. You simply generate several example sentences according to the user's prompt. You write the sentences as bulleted lists, without any commentary before or after. Each sentence should be independent from the others (i.e. not referring to the earlier ones whatsoever.) Write as many sentences as possible."
# language_classification_prmopt = "Write several sentences that could be classified as 'english', 'mandarin', 'spanish', or 'german'. Format each sentence as \"<language name>: <sentence>\""
# sentiment_classification_prompt = "Write a neutral sentence, and then a positive and a negative version of the sentence. Format each sentence as \"<sentiment type>: <sentence>\""
# sentiment_classification_prompt = "Write a 'positive', 'negative', or 'neural'. Format each sentence as \"<sentiment type>: <sentence>\""
# chem_prompt = "Write several facts about chemistry."
# phys_prompt = "Write several facts about physics."
# math_prompt = "Write several facts about mathematics."

sys_prompt_2 = ""
contextualization_prompt = """
Generate several test cases of this nature.

1. He dropped it because it was too (heavy, hot).
Context for 'heavy': He was given a 50kg weight.
Context for 'hot': The waiter was given a dish, but he wasn't wearing gloves.
""".strip()

history = [
    # {"role": "system", "content": sys_prompt_2},
    {"role": "user", "content": contextualization_prompt}
]
for _ in range(30):
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        temperature=1.0
    )
    text = result.choices[0].message.content
    print(text, flush=True)
    history.append({"role": "assistant", "content": text})

