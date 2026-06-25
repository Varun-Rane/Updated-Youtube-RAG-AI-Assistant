import re

from utils import trim_text


_STOP_WORDS = {
    "what", "is", "the", "a", "an",
    "explain", "tell", "me", "about",
    "in", "this", "video", "from",
    "how", "why", "when", "where",
    "does", "do", "did", "can",
    "could", "would", "should",
    "please", "give", "show",
    "describe", "define", "and",
    "or", "of", "to", "for",
    "with", "are", "was", "were",
}


TIMESTAMP_PATTERN = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?"
)


def extract_keywords(question):
    words = re.findall(
        r"\w+",
        question.lower(),
    )

    return [
        word
        for word in words
        if (
            len(word) > 2
            and word not in _STOP_WORDS
        )
    ]


def get_all_documents(retriever):
    try:
        vectorstore = getattr(
            retriever,
            "vectorstore",
            None,
        )

        if vectorstore is None:
            vectorstore = getattr(
                retriever,
                "_vectorstore",
                None,
            )

        if vectorstore is None:
            return []

        return list(
            vectorstore.docstore._dict.values()
        )

    except Exception as exc:
        print(
            "DOCSTORE ACCESS FAILED:",
            exc,
        )
        return []


def timestamp_search(
    retriever,
    timestamp_query,
):
    all_docs = get_all_documents(
        retriever
    )

    matches = []

    for doc in all_docs:

        ts_range = doc.metadata.get(
            "timestamp_range",
            "",
        )

        ts_single = doc.metadata.get(
            "timestamp",
            "",
        )

        if (
            timestamp_query in ts_range
            or timestamp_query == ts_single
        ):
            matches.append(doc)

    if matches:

        print("=" * 80)
        print("TIMESTAMP MATCH FOUND")
        print("QUERY:", timestamp_query)
        print("MATCHES:", len(matches))
        print("=" * 80)

    return matches[:5]


def vector_search(retriever, question,):
    docs = retriever.invoke(
        question
    )

    print("=" * 80)
    print("VECTOR SEARCH")
    print("QUESTION:", question)
    print("DOCS:", len(docs))

    for i, doc in enumerate(
        docs[:5],
        start=1,
    ):
        ts = doc.metadata.get(
            "timestamp_range",
            doc.metadata.get(
                "timestamp",
                "N/A",
            ),
        )
        print(
            f"{i}. {ts}"
        )

    print("=" * 80)

    return docs

def keyword_boost(
    retriever,
    question,
    vector_docs,
):
    print("=" * 80)
    print("KEYWORD BOOST ENTERED")
    print("QUESTION:", question)
    print("=" * 80)
    keywords = extract_keywords(question)
    print("EXTRACTED KEYWORDS:", keywords)
    if not keywords:
        return vector_docs

    all_docs = get_all_documents(
        retriever
    )

    keyword_hits = []

    for doc in all_docs:

        text = doc.page_content.lower()

        matched = []

        for keyword in keywords:

            if keyword in text:
                matched.append(
                    keyword
                )

        if matched:

            keyword_hits.append(
                (
                    len(matched),
                    doc,
                )
            )

    keyword_hits.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    keyword_docs = [
        item[1]
        for item in keyword_hits[:10]
    ]

    if keyword_docs:
        print("=" * 80)
        print("KEYWORD BOOST")
        print("KEYWORDS:", keywords)
        print("MATCHES:", len(keyword_docs))

        for hit in keyword_docs[:10]:
            print(hit.metadata.get("timestamp_range", "N/A"))

        print("=" * 80)

    merged = []
    seen = set()

    for doc in (keyword_docs + vector_docs):
        key = (
            doc.metadata.get("video_id", ""),
            doc.metadata.get("chunk_index", ""),
        )

        if key in seen:
            continue

        seen.add(key)

        merged.append(doc)

    return merged


COMPARISON_WORDS = {
    "compare",
    "comparison",
    "difference",
    "different",
    "vs",
    "versus",
}

def is_comparison_query(question):
    q = question.lower()
    return any(word in q for word in COMPARISON_WORDS)

def retrieve_documents(
    retriever,
    question,
):
    if retriever is None:
        return []

    question = question.strip()

    timestamp_match = (
        TIMESTAMP_PATTERN.search(
            question
        )
    )

    if timestamp_match:

        docs = timestamp_search(
            retriever,
            timestamp_match.group(),
        )

        if docs:
            return docs

    # Determine retrieval strategy
    if is_comparison_query(question):
        # Comparison query: search separately for each extracted keyword, skipping comparison words
        keywords = [
            kw for kw in extract_keywords(question)
            if kw not in COMPARISON_WORDS
        ]
        docs = []
        seen = set()
        for kw in keywords:
            # Perform a vector search for each keyword and collect results
            kw_docs = vector_search(retriever, kw)
            for doc in kw_docs:
                key = (
                    doc.metadata.get("video_id", ""),
                    doc.metadata.get("chunk_index", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
                # Stop early if we already have enough chunks (consistent with later slicing)
                if len(docs) >= 8:
                    break
            if len(docs) >= 8:
                break
        # No additional keyword boost – we already used keyword‑based retrieval
    else:
        # Regular query flow
        docs = vector_search(
            retriever,
            question,
        )

        docs = keyword_boost(
            retriever,
            question,
            docs,
        )
    print("=" * 80)
    print("IVF DEBUG")

    all_docs = get_all_documents(
        retriever
    )

    ivf_count = 0

    for doc in all_docs:

        text = doc.page_content.lower()

        if (
            "ivf" in text
            or "inverted" in text
            or "file index" in text
            or "आईबीएफ" in text
        ):
            ivf_count += 1

            print(
                doc.metadata.get(
                    "timestamp_range",
                    "N/A",
                )
            )

    print("TOTAL IVF DOCS:",ivf_count,)

    print("=" * 80)
    print("=" * 80)
    print("FINAL DOCS")
    
    for doc in docs[:8]:
        print(
            doc.metadata.get(
                "timestamp_range",
                "N/A"
            )
        )
    
    print("=" * 80)
    return docs[:8]


def format_retrieved_context(
    documents,
    max_chars,
):
    sections = []

    retrieved_chunks = []

    current_chars = 0

    for document in documents:

        metadata = document.metadata

        timestamp = metadata.get(
            "timestamp_range",
            metadata.get(
                "timestamp",
                "00:00",
            ),
        )

        video_title = metadata.get(
            "video_title",
            "Video",
        )

        video_url = metadata.get(
            "video_url",
            "",
        )

        source_label = metadata.get(
            "source_label",
            "Video",
        )

        text = trim_text(
            document.page_content.strip(),
            700,
        )

        block = f"""
            Source Video: {source_label} - {video_title}
            Video URL: {video_url}
            Timestamp: {timestamp}
            Transcript:
            {text}
        """

        if (current_chars + len(block) > max_chars):
            break

        current_chars += len(block)

        sections.append(block)

        retrieved_chunks.append(
            {
                "video_id": metadata.get(
                    "video_id",
                    "",
                ),
                "video_title": video_title,
                "video_url": video_url,
                "source_label": source_label,
                "timestamp": timestamp,
                "start": metadata.get(
                    "start",
                    "",
                ),
                "duration": metadata.get(
                    "duration",
                    "",
                ),
                "chunk_index": metadata.get(
                    "chunk_index",
                    "",
                ),
                "text": text,
            }
        )

    context = (
        "\n\n--------------------------\n\n".join(
            sections
        )
    )

    context = trim_text(
        context,
        max_chars,
    )

    return (
        context,
        retrieved_chunks,
    )


def unique_timestamps(chunks,):
    seen = set()

    result = []

    for chunk in chunks:

        ts = chunk.get(
            "timestamp"
        )

        if (
            ts
            and ts not in seen
        ):
            seen.add(ts)
            result.append(ts)

    return result


def unique_sources(
    chunks,
    videos=None,
):
    seen = set()

    result = []

    for chunk in chunks:

        url = chunk.get(
            "video_url",
            "",
        )

        if url in seen:
            continue

        seen.add(url)

        result.append(
            {
                "label": chunk.get(
                    "source_label",
                    "Video",
                ),
                "title": chunk.get(
                    "video_title",
                    "Video",
                ),
                "url": url,
            }
        )

    if result:
        return result

    if videos:
        for video in videos:
            result.append(
                {
                    "label": video.get(
                        "label",
                        "Video",
                    ),
                    "title": video.get(
                        "video_title",
                        "Video",
                    ),
                    "url": video.get(
                        "video_url",
                        "",
                    ),
                }
            )

    return result