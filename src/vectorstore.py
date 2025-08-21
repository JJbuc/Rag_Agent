from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from config import load_config
from langchain_community.document_loaders import DirectoryLoader, TextLoader

cfg = load_config()

def get_retriever():
    loader = DirectoryLoader(
        cfg["paths"]["raw_dir"],
        glob="*.txt",  # or "*.md", "*.pdf" etc. depending on your files
        loader_cls=TextLoader
    )
    docs_list = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    )

    # Create retriever
    retriever = vectorstore.as_retriever(k=3)

    return retriever

def retrieve_doc(question):
    print("#### Entered retrieve_doc ####")
    """
    Retreieve the relevant documents from the documents that we have
    """
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return documents