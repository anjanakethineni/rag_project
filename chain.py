from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from config import OPENAI_API_KEY
from rag import build_retriever

 
rate_limiter = InMemoryRateLimiter(
   requests_per_second=0.5,
   check_every_n_seconds=0.1,
   max_bucket_size=5,
)
 
llm = ChatOpenAI(
  model="gpt-4o-mini",
   api_key=OPENAI_API_KEY,
   rate_limiter=rate_limiter,
)
 
prompt = ChatPromptTemplate.from_messages([
   ("system", "Answer using only the provided context. If unsure, say so."),
   ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
 
retriever = build_retriever()
 
loader=PyPDFLoader("../rag_project/data/docs/Neonatology.pdf")
docs=loader.load()
def format_docs(docs):
   return "\n\n".join(d.page_content for d in docs)
 
 
rag_chain = (
   {"context": retriever | format_docs, "question": RunnablePassthrough()}
   | prompt
   | llm
   | StrOutputParser()
)