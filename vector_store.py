from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def build_vector_store(chunks, embeddings, settings):
    if not chunks:
        raise ValueError("No transcript chunks were available for indexing.")

    documents = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k},
    )

    return vectorstore, retriever
