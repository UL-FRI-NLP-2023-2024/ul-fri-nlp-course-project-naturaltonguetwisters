from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS


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


def get_docs_makeitblack(character_name: str, question: str) -> list[str]:
    QUESTION = f"Question for {character_name}:\n{question}"
    docs = vectorstore.similarity_search(QUESTION.format(character_name=character_name, question=question))
    return docs
