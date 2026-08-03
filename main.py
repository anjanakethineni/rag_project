from chain import rag_chain
from retry_utils import with_retry
from logger import logger
 
 
@with_retry
def call_chain(question: str) -> str:
   logger.info(f"Calling RAG chain with: {question!r}")
   answer = rag_chain.invoke(question)
   logger.info("Chain call succeeded")
   return answer
 
 
def ask(question: str) -> str:
   if not question or not question.strip():
       return "Error: question cannot be empty."
 
   try:
       return call_chain(question)
   except (TimeoutError, ConnectionError) as e:
    logger.error(f"Transient failure after retries: {e}")
    return "Service temporarily unavailable. Try again."
   except Exception as e:
    logger.error(f"Unhandled error: {e}", exc_info=True)
    return "Something went wrong. The issue has been logged."
 
 
if __name__ == "__main__":
   print("Day 16 RAG app. Type 'exit' to quit.\n")
   while True:
       q = input("Ask something: ")
       if q.strip().lower() == "exit":
           break
       print(f"\n> {ask(q)}\n")