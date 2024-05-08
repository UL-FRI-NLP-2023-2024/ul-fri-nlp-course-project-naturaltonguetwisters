max_seq_length = 2048
dtype = None
load_in_4bit = True

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llama3_8b_alpaca_clean",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = "You are {character_name} from {novel_title}. Stay true to the character from the novel; embody the character as much as possible. Have their personality come across in your words. Be conversational and brief, converse with me in the manner this character would converse. Be friendly and engaging, keep the conversation going; be curious about me."

def ask_question(character_name, novel_title, prompt):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction.format(character_name = character_name, novel_title = novel_title),
                prompt,
                "",
            )
        ], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    response_text = tokenizer.batch_decode(outputs)
    return response_text[0].split('### Response:\n')[1].replace('<|eot_id|>', '')


print("\n")

import json
with open("tests.json", "r") as f:
    tests = json.load(f)

for test in tests:
    character_name = test["character_name"]
    novel_title = test["novel_title"]
    prompt = test["prompt"]
    print(f"Asking {character_name} from {novel_title} the following question:")
    print(prompt)
    response = ask_question(character_name, novel_title, prompt)
    print("Response:")
    print(response)
    print('---------------')
    print()
