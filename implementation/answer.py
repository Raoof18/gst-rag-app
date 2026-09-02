import os
import requests
import psycopg2
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

CONN_STRING = os.getenv("SUPABASE_CONNECTION_STRING")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

llm = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))


def embed_query(text: str):
    # has to be the same model that embedded the stored chunks, or the vectors
    # aren't comparable and search just returns garbage with no error
    response = requests.post(
        HF_EMBED_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()
    # sometimes this comes back as per-token embeddings instead of one pooled vector
    if isinstance(result[0], list) and isinstance(result[0][0], list):
        tokens = result[0]
        dim = len(tokens[0])
        pooled = [sum(t[i] for t in tokens) / len(tokens) for i in range(dim)]
        return pooled
    return result


def vector_search(query_vector, top_k=10):
    conn = psycopg2.connect(CONN_STRING)
    cur = conn.cursor()
    cur.execute(
        """
        select text, source, page, page_label
        from gst_chunks
        order by embedding <-> %s::vector
        limit %s
        """,
        (query_vector, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        Document(page_content=r[0], metadata={"source": r[1], "page": r[2], "page_label": r[3]})
        for r in rows
    ]


def keyword_search(query: str, top_k=10):
    conn = psycopg2.connect(CONN_STRING)
    cur = conn.cursor()
    cur.execute(
        """
        select text, source, page, page_label,
               ts_rank(text_search, plainto_tsquery('english', %s)) as rank
        from gst_chunks
        where text_search @@ plainto_tsquery('english', %s)
        order by rank desc
        limit %s
        """,
        (query, query, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        Document(page_content=r[0], metadata={"source": r[1], "page": r[2], "page_label": r[3]})
        for r in rows
    ]


def reciprocal_rank_fusion(result_lists, k=60):
    # can't just average the raw scores, vector distance and text rank aren't
    # on the same scale -- so this only looks at rank position, not the score itself
    scores, doc_lookup = {}, {}
    for docs in result_lists:
        for rank, doc in enumerate(docs):
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:150])
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            doc_lookup[key] = doc
    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_lookup[key] for key in ranked_keys]


def hybrid_retrieve(query: str, top_k=10):
    query_vector = embed_query(query)
    vector_docs = vector_search(query_vector, top_k=top_k)
    keyword_docs = keyword_search(query, top_k=top_k)
    return reciprocal_rank_fusion([vector_docs, keyword_docs])


REWRITE_PROMPT = """Rewrite the following question as a clear, standalone English question suitable for searching legal/tax documents. If it's in Hindi or Hinglish, translate it. Return ONLY the rewritten question, nothing else.

Question: {question}"""

REWRITE_PROMPT_WITH_HISTORY = """Rewrite the LATEST question as a clear, standalone English question suitable for searching legal/tax documents. Use the previous exchange ONLY to resolve references, pronouns, or implicit context in the latest question (e.g. "what about for goods?" following a question about services). If the latest question is already standalone, ignore the previous exchange. If it's in Hindi or Hinglish, translate it. Return ONLY the rewritten standalone question, nothing else.

Previous question: {prev_question}
Previous answer: {prev_answer}

Latest question: {question}"""

RERANK_PROMPT = """You are ranking document chunks by relevance to a question. Given the question and a list of numbered chunks, return ONLY a comma-separated list of chunk numbers, ordered from MOST relevant to LEAST relevant. Include all chunk numbers exactly once. No other text.

Question: {question}

Chunks:
{chunks_text}"""

PROMPT_VARIANTS = {
    "precise": """You are a GST compliance assistant. Answer only using the provided context below. Be precise and cite sources using [1], [2] etc. If the context defines a term, use that definition directly rather than reasoning about it independently. If you cannot answer confidently from the context, say so clearly.

IMPORTANT: Do not state a definitive tax liability, exact amount owed, or "you owe X" as a confirmed fact for any specific transaction. Explain the RULE and the CONDITIONS that must be met rather than declaring an outcome. If the question asks for a specific number or a yes/no liability determination, explain what the outcome depends on and recommend the person confirm their specific situation with a tax professional or CA.

Do not add details that are not present in the provided context, even if you believe them to be generally true. If the context doesn't mention something, do not include it.

Context:
{context}""",
    "simple": """You are a GST compliance assistant helping someone with no tax background. Answer using only the provided context below, in plain, simple language. Cite sources using [1], [2] etc.

IMPORTANT: Do not state a definitive tax liability or exact amount owed as confirmed fact. Explain the conditions instead, and recommend consulting a tax professional for specific numbers.

Do not add details that are not present in the provided context. If the context doesn't cover the question, say so clearly.

Context:
{context}""",
}


def rewrite_query(question: str, prev_exchange: dict | None = None) -> str:
    # prev_exchange only matters for follow-ups like "what about for goods instead" --
    # a plain standalone question just ignores it
    if prev_exchange and prev_exchange.get("question"):
        prompt = REWRITE_PROMPT_WITH_HISTORY.format(
            prev_question=prev_exchange["question"],
            prev_answer=prev_exchange.get("answer", "")[:2000],  # learned this the hard way -- 500 chars was
            question=question,                                    # cutting off before point 5 of a 5-point answer
        )
    else:
        prompt = REWRITE_PROMPT.format(question=question)

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def rerank_chunks(question: str, docs, top_n=5):
    chunks_text = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs))
    response = llm.invoke([HumanMessage(content=RERANK_PROMPT.format(question=question, chunks_text=chunks_text))])
    try:
        order = [int(x.strip()) for x in response.content.strip().split(",")]
        ranked_docs = [docs[i] for i in order if i < len(docs)]
    except (ValueError, IndexError):
        # model didn't return clean numbers, just fall back rather than blow up
        ranked_docs = docs
    return ranked_docs[:top_n]


def fetch_context(question: str, top_n=5, prev_exchange: dict | None = None):
    rewritten = rewrite_query(question, prev_exchange=prev_exchange)
    # search both the raw question and the rewritten one -- if the rewrite goes
    # sideways for some reason, we're not fully betting on it being right
    original_fused = hybrid_retrieve(question)
    rewritten_fused = hybrid_retrieve(rewritten)

    seen = set()
    merged_docs = []
    for doc in original_fused + rewritten_fused:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:150])
        if key not in seen:
            seen.add(key)
            merged_docs.append(doc)

    return rerank_chunks(question, merged_docs, top_n=top_n)


def format_context_with_citations(docs):
    context_parts, sources = [], []
    for i, doc in enumerate(docs, 1):
        raw_source = doc.metadata.get('source', 'unknown')
        clean_name = os.path.splitext(os.path.basename(raw_source))[0]  # drop the folder + extension, nobody needs that

        page_label = doc.metadata.get('page_label')
        if page_label:
            label = f"[{i}] {clean_name} (page {page_label})"
        else:
            label = f"[{i}] {clean_name}"  # HTML sources don't have pages, so just skip that part

        context_parts.append(f"{label}\n{doc.page_content}")
        sources.append(label)
    return "\n\n".join(context_parts), sources


def answer_question(question: str, style: str = "precise", top_n: int = 5,
                     prev_exchange: dict | None = None, conversation_history: list | None = None):
    # prev_exchange only feeds the rewrite step (retrieval). conversation_history
    # is different -- it goes straight to the model so it actually remembers the
    # conversation, otherwise stuff like "what did I ask first" just doesn't work
    docs = fetch_context(question, top_n=top_n, prev_exchange=prev_exchange)
    context, sources = format_context_with_citations(docs)

    system_prompt = PROMPT_VARIANTS[style].format(context=context) + """

Note: you may also see earlier turns of this conversation above. If the person asks about the
conversation itself (e.g. what they asked earlier), you can answer from that directly -- that is
not subject to the "only use the provided context" restriction, which applies to GST/legal
substance only."""

    messages = [SystemMessage(content=system_prompt)]

    if conversation_history:
        for turn in conversation_history:
            if turn.get("role") == "user":
                messages.append(HumanMessage(content=turn.get("text", "")))
            elif turn.get("role") == "bot":
                messages.append(AIMessage(content=turn.get("text", "")))

    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, sources
