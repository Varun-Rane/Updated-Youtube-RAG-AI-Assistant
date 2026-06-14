RAG_PROMPT = """
You are an AI assistant.

Answer ONLY from the transcript.

Never hallucinate.

If answer is unavailable say:

I couldn't find this in the loaded video.

Retrieved Transcript:

{context}

Question:

{question}

Return:

Answer

Mentioned Around

Source Video

Retrieved Transcript Evidence
"""


SUMMARY_PROMPT = """
You are an expert lecture summarizer.

Generate a detailed summary.

Transcript:

{context}

Task:

{question}

Generate:

Summary

Key Concepts

Timeline

Questions Discussed

Important Terms
"""


GENERAL_PROMPT = """
You are an intelligent AI assistant.

Behave exactly like ChatGPT.

Conversation History:

{history}

Question:

{question}
"""


MEMORY_PROMPT = """
Use conversation history.

Conversation:

{history}

Question:

{question}

Answer naturally.
"""