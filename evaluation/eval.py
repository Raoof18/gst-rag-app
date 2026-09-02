import sys
import math
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context

load_dotenv(override=True)

# kept as its own model instance, separate from the pipeline's llm in answer.py,
# in case judging needs different settings later
judge_llm = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(description="Concise feedback comparing the answer to the reference answer")
    accuracy: float = Field(description="1 (wrong -- any wrong answer must score 1) to 5 (perfectly accurate). 3 is acceptable.")
    completeness: float = Field(description="1 (missing key info) to 5 (all reference info included). Only 5 if ALL reference info is present.")
    relevance: float = Field(description="1 (off-topic) to 5 (directly addresses the question, no extra info). Only 5 if fully on-topic.")


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 5) -> RetrievalEval:
    """Evaluate retrieval performance for a test question, using the SAME
    fetch_context pipeline (rewrite + hybrid search + rerank) the live app uses."""
    retrieved_docs = fetch_context(test.question, top_n=k)

    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for s in mrr_scores if s > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """Evaluate answer quality using LLM-as-a-judge."""
    generated_answer, sources = answer_question(test.question)

    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers about Indian GST law. "
                       "Evaluate the generated answer by comparing it to the reference answer. "
                       "Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Evaluate on three dimensions:
1. Accuracy: factual correctness vs the reference answer. If the answer is wrong, accuracy must be 1.
2. Completeness: does it cover everything in the reference answer?
3. Relevance: does it directly answer the question without unnecessary extra info?

Give detailed feedback and scores 1-5 for each.""",
        },
    ]

    structured_judge = judge_llm.with_structured_output(AnswerEval)
    answer_eval = structured_judge.invoke(judge_messages)

    return answer_eval, generated_answer, sources


def evaluate_all_retrieval(k: int = 5):
    tests = load_tests()
    total = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test, k=k)
        yield test, result, (index + 1) / total


def evaluate_all_answers():
    tests = load_tests()
    total = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        yield test, result, (index + 1) / total


def run_cli_evaluation(test_number: int):
    tests = load_tests()
    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}\nTest #{test_number}\n{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    print(f"\n{'=' * 80}\nRetrieval Evaluation\n{'=' * 80}")
    retrieval_result = evaluate_retrieval(test)
    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    print(f"\n{'=' * 80}\nAnswer Evaluation\n{'=' * 80}")
    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)
    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def run_full_evaluation():
    """Run retrieval + answer eval across ALL tests and print averaged summary stats."""
    tests = load_tests()

    print(f"Running full evaluation on {len(tests)} test questions...\n")

    retrieval_results, answer_results = [], []
    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] {test.category}: {test.question[:60]}...")
        retrieval_results.append(evaluate_retrieval(test))
        answer_results.append(evaluate_answer(test)[0])

    n = len(tests)
    print(f"\n{'=' * 80}\nSUMMARY ({n} questions)\n{'=' * 80}")
    print(f"Avg MRR: {sum(r.mrr for r in retrieval_results) / n:.4f}")
    print(f"Avg nDCG: {sum(r.ndcg for r in retrieval_results) / n:.4f}")
    print(f"Avg Keyword Coverage: {sum(r.keyword_coverage for r in retrieval_results) / n:.1f}%")
    print(f"Avg Accuracy: {sum(a.accuracy for a in answer_results) / n:.2f}/5")
    print(f"Avg Completeness: {sum(a.completeness for a in answer_results) / n:.2f}/5")
    print(f"Avg Relevance: {sum(a.relevance for a in answer_results) / n:.2f}/5")

    # Per-category breakdown -- useful for spotting weak spots (e.g. out_of_scope, hinglish)
    categories = sorted(set(t.category for t in tests))
    print(f"\n{'-' * 80}\nBy category:\n{'-' * 80}")
    for cat in categories:
        idxs = [i for i, t in enumerate(tests) if t.category == cat]
        cat_n = len(idxs)
        cat_acc = sum(answer_results[i].accuracy for i in idxs) / cat_n
        cat_mrr = sum(retrieval_results[i].mrr for i in idxs) / cat_n
        print(f"{cat:15s} (n={cat_n:2d})  accuracy={cat_acc:.2f}/5  mrr={cat_mrr:.3f}")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "all":
        run_full_evaluation()
        return

    if len(sys.argv) != 2:
        print("Usage:\n  uv run eval.py <test_row_number>\n  uv run eval.py all")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer, or the word 'all'")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
