import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
# Load docs
from langchain_community.document_loaders import WebBaseLoader

os.environ["OPENAI_API_KEY"] = "your key"

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Store splits
vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# LLM

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# # # Create the question-answering chain using the retrieval index
# qa_chain = load_qa_chain(llm)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",  # Specify 'stuff' for single-document QA
    retriever=vectorstore.as_retriever()
)

from langchain import hub
# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
# prompt = hub.pull("rlm/rag-prompt")
# qa_chain = RetrievalQA.from_llm(
#     llm, retriever=vectorstore.as_retriever(), prompt=prompt
# )


# # Function to ask questions

# # Example question
question = "What are autonomous agents?"
result = qa_chain.invoke(question)

print(" \nAnswer:", result)
