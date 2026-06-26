from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def build_vector_store(

    chunks,

    embeddings,

    settings,

):

    documents = [

        Document(

            page_content=chunk["text"],

            metadata=chunk["metadata"],

        )

        for chunk in chunks

    ]

    vectorstore = FAISS.from_documents(

        documents,

        embeddings,

    )

    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={
            "k": settings.dense_top_k,
            "fetch_k": max(settings.dense_top_k * 3, 20),
        }

    )

    return vectorstore, retriever