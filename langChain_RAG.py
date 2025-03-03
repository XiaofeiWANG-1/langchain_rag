import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model

# os.environ["OPENAI_API_KEY"] = "your key"
# export OPENAI_API_KEY="your key" # system setting

def load_file(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

pdf_text = load_file("resume.pdf")

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_text(pdf_text)

documents = [Document(page_content=chunk) for chunk in chunks]
         


# # Initialize the OpenAI model for embeddings
embedding_model = OpenAIEmbeddings()

# # Create the FAISS index for document retrieval
vectorstore = FAISS.from_documents(documents, embedding_model)
# Store splits
# # # Initialize the OpenAI language model
# llm = OpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# # # Create the question-answering chain using the retrieval index
# qa_chain = load_qa_chain(llm)
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff",  # Specify 'stuff' for single-document QA
#     retriever=vectorstore.as_retriever()
# )

from langchain import hub
# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
prompt = hub.pull("rlm/rag-prompt")
# qa_chain = RetrievalQA.from_llm(
#     llm, retriever=vectorstore.as_retriever(), prompt=prompt
# )

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# qa_chain = (
#     {
#         "context": vectorstore.as_retriever() | format_docs,
#         "question": RunnablePassthrough(),
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # # Example question
# question = "What do lions eat?"
# result = qa_chain.invoke(question)

# print(" \nAnswer:", result)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

result= rag_chain.invoke({"input": "Introduce Xiaofei"})
print(" \nAnswer:", result['answer'])
