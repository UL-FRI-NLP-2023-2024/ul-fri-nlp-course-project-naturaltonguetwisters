import os
import tempfile
import requests
import torch
import transformers
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.faiss import FAISS

LLM_MODEL = "llama3_8b_alpaca_clean"
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

filename = 'Remarkably_Bright_Creatures.txt'
texts = []
with open(filename, 'r', encoding='utf-8') as f:
    texts.extend(f.read())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

vectorstore = FAISS.from_texts(texts, embedding)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    k=10,
)

PROMPT_TEMPLATE = """
You are a helpful AI QA assistant. When answering questions, use the context enclosed by triple backquotes if it is relevant.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Reply your answer in markdown format.

```
{context}
```

### Question:
{question}

### Answer:
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

def answer_question(question: str, history: dict[str] = None) -> str:
    if history is None:
        history = []
    response = llm_chain.invoke({"question": question, "chat_history": history})
    answer = response["answer"].split("### Answer:")[-1].strip()
    return answer
