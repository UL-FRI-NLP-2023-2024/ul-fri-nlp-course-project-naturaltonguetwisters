max_seq_length = 2048
dtype = None
load_in_4bit = True

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llama_alpaca_clean",
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

instruction = "You are Harry Potter. You answer in an engaging way and embody the personality of Harry. Answer in a human way of expressing yourself. Sometimes a short answer is the best, but never write a whole wall of text, because that is not very human-like. So prioritize shorter answers."

def ask_harry(input):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,
                input,
                "",
            )
        ], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    response_text = tokenizer.batch_decode(outputs)
    return response_text[0].split('### Response:\n')[1].replace('<|eot_id|>', '')


input = "What are your biggest achivements in life and your biggest blunders?"
print(f"Question: {input}")
print(ask_harry(input))

input = "What is your favorite spell?"
print(f"Question: {input}")
print(ask_harry(input))

input = "Who is your best friend, if you can choose only one?"
print(f"Question: {input}")
print(ask_harry(input))


print("End of inference test.")
