import os
import tempfile
import requests
import torch
import transformers
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS

from huggingface_hub import login
from constants import huggingface_token

login(token = huggingface_token, add_to_git_credential = True)

#LLM_MODEL = "llama3_8b_alpaca_clean"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
device = f"cuda:{torch.cuda.current_device()}"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_config = transformers.AutoConfig.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
)
retrieval_model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
)
retrieval_model.eval()

retrieval_tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LLM_MODEL,
)
retrieval_tokenizer.pad_token = retrieval_tokenizer.eos_token
retrieval_tokenizer.padding_side = "right"

generate_text = transformers.pipeline(
    task="text-generation",
    model=retrieval_model,
    tokenizer=retrieval_tokenizer,
    return_full_text=True,
    max_new_tokens=8192,
    repetition_penalty=1.1,
)

llm = HuggingFacePipeline(pipeline=generate_text)

template = """
You are a helpful AI QA assistant. When answering questions, use the context enclosed by triple backquotes if it is relevant.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Reply your answer in markdown format.
Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)
chain = prompt | llm

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda"},
)

filename = 'MakeItBlack.txt'

loader = TextLoader(filename)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    k=10,
)


PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with the further context about the story and an input that provides a question. Write a response that appropriately completes the request.

### Instruction:
You are {character_name} from {novel_title}. Stay true to the character from the novel; embody the character as much as possible. Have their personality come across in your words. Be conversational and brief, converse with me in the manner this character would converse. Be friendly and engaging, keep the conversation going; be curious about me.

### Context:
```
{context}
```

### Input:
{question}

### Response:
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE.strip(),
)

llm_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    verbose=False,
)

def get_docs_makeitblack(character_name: str, question: str) -> list[str]:
    QUESTION = f"Question for {character_name}:\n{question}"
    docs = vectorstore.similarity_search(QUESTION.format(character_name=character_name, question=question))
    print(docs)
    return docs


def answer_question(character_name: str, novel_title: str, question: str, history: dict[str] = None) -> str:
    if history is None:
        history = []
    response = llm_chain.invoke({"character_name": character_name, "novel_title": novel_title, "question": question, "chat_history": history})
    answer = response["answer"].split("### Answer:")[-1].strip()
    return answer
