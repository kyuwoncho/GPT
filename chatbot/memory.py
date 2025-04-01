from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

class MemoryManager:
    def __init__(self, openai_api_key):
        self.history = []
        self.cache = {}
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = None 

    def _hash(self, text):
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def check_cache(self, question):
        return self.cache.get(self._hash(question))

    def add_to_cache(self, question, result):
        self.cache[self._hash(question)] = result

    def store_history(self, question, answer):
        self.history.append((question, answer))
        doc = Document(page_content=question)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vectorstore.add_documents([doc])

    def find_similar_question(self, question, k=1):
        if self.vectorstore is None:
            return None
        result = self.vectorstore.similarity_search(question, k=k)
        if result:
            text = result[0].page_content
            for q, a in self.history:
                if q == text:
                    return a
        return None
