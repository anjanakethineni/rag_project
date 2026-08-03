from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY
 
 
def build_retriever(docs_path: str = "data/docs", persist_dir: str = "data/docs"):
    loader = DirectoryLoader(docs_path,glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
 
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # #vectorstore = Chroma.from_documents(
    #     chunks, embeddings, persist_directory=persist_dir
    # )
 
    # return vectorstore.as_retriever(search_kwargs={"k": 4})