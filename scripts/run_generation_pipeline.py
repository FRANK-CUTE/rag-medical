from __future__ import annotations

import argparse
import json
import os
from pprint import pprint
from typing import Any, Dict, List

import torch

from answer_evaluator import AnswerEvaluator
from batch_generation_runner import BatchGenerationRunner
from context_assembler import ContextAssembler
from generation_cache import GenerationCache
from llm_generator import LLMGenerator
from medical_generation_pipeline import MedicalGenerationPipeline
from multi_path_retriever import MultiPathRetriever, load_bm25_index
from prompt_templates import MedicalPromptTemplates
from query_processor import process_query
from reranker import load_reranker, rerank_results
from retrieve_with_chroma import load_model


DEFAULT_TEST_QUERIES = [
    "Effect of metformin on CVD after 2020",
    "Recent evidence on aspirin for primary prevention after 2020",
    "GLP-1 receptor agonists and cardiovascular outcomes after 2020",
]

DEFAULT_REFERENCE_ANSWERS = {
    "Effect of metformin on CVD after 2020": (
        "Recent evidence suggests metformin remains important for glycemic control, "
        "but cardiovascular benefit is often discussed in comparison with newer agents. "
        "The answer should mention outcome evidence after 2020, uncertainty across populations, "
        "and safety or treatment considerations."
    ),
    "Recent evidence on aspirin for primary prevention after 2020": (
        "Recent evidence generally emphasizes balancing modest cardiovascular benefit against bleeding risk "
        "in primary prevention. The answer should mention risk, safety, and the need for individualized "
        "treatment recommendations."
    ),
    "GLP-1 receptor agonists and cardiovascular outcomes after 2020": (
        "Recent evidence generally supports cardiovascular benefit for some GLP-1 receptor agonists in "
        "appropriate patients, while noting differences across trials, safety issues, and treatment recommendations."
    ),
}


def format_top_results(results: List[Dict[str, Any]], show_top_n: int) -> None:
    print("\nTop reranked results:")
    for item in results[:show_top_n]:
        print(f"===== Rank {item.get('rank', '?')} =====")
        print("ID:", item.get("id"))
        print("Source:", item.get("source"))
        print("Scores:")
        pprint(item.get("rerank_scores", {}), sort_dicts=False)
        print("Metadata:")
        pprint(item.get("metadata", {}), sort_dicts=False)
        print("Document:")
        print((item.get("document", "") or "")[:500])
        print()


def save_log(log_path: str, payload: Any) -> None:
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved log to: {log_path}")


def load_reference_answers(reference_answers_path: str = "") -> Dict[str, str]:
    if reference_answers_path and os.path.exists(reference_answers_path):
        with open(reference_answers_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    return dict(DEFAULT_REFERENCE_ANSWERS)


class RetrievalResources:
    def __init__(self, args):
        if not os.path.exists(args.bm25_cache):
            raise FileNotFoundError(
                f"BM25 cache not found: {args.bm25_cache}\n"
                f"Please run build_bm25_index.py first."
            )

        print("\nLoading BM25 index...")
        self.bm25_index = load_bm25_index(args.bm25_cache)
        print("BM25 index loaded")

        print("\nLoading vector model...")
        self.vector_tokenizer, self.vector_model = load_model(args.vector_model, args.device)
        print("Vector model loaded")

        print("\nLoading reranker...")
        self.reranker_tokenizer, self.reranker_model = load_reranker(
            args.reranker_model,
            args.device,
        )
        print("Reranker loaded")

        self.retriever = MultiPathRetriever(
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            tokenizer=self.vector_tokenizer,
            model=self.vector_model,
            device=args.device,
            bm25_index=self.bm25_index,
        )


def build_generation_pipeline(args) -> MedicalGenerationPipeline:
    context_assembler = ContextAssembler(
        tokenizer_path=args.vector_model,
        max_context_tokens=args.max_context_tokens,
        dedup_threshold=args.dedup_threshold,
        same_source_soft_limit=args.same_source_soft_limit,
        generation_chunk_token_limit=args.generation_chunk_token_limit,
        generation_prompt_safety_margin=args.generation_prompt_safety_margin,
        min_context_tokens=args.min_context_tokens,
    )

    prompt_templates = MedicalPromptTemplates()

    cache = None
    if args.enable_cache:
        cache = GenerationCache(
            max_entries=args.cache_max_entries,
            ttl_seconds=args.cache_ttl_seconds,
            cache_temperature_threshold=args.cache_temperature_threshold,
        )

    llm_generator = LLMGenerator(
        model_name=args.ollama_model,
        base_url=args.ollama_base_url,
        timeout=args.ollama_timeout,
        default_temperature=args.default_temperature,
        default_max_tokens=args.default_max_output_tokens,
        cache=cache,
        debug=args.debug_llm,
    )

    return MedicalGenerationPipeline(
        context_assembler=context_assembler,
        prompt_templates=prompt_templates,
        llm_generator=llm_generator,
        enable_evidence_evaluation=not args.disable_evidence_evaluation,
        enable_critical_review=not args.disable_critical_review,
        evidence_context_budget=args.evidence_context_budget,
        filtered_context_budget=args.filtered_context_budget,
        review_context_budget=args.review_context_budget,
    )


def evaluate_answer_if_needed(
    result: Dict[str, Any],
    evaluator: AnswerEvaluator,
    reference_answers: Dict[str, str],
    args,
) -> None:
    if not args.enable_answer_evaluation:
        return

    query = result.get("query", "")
    reference_answer = reference_answers.get(query, "")
    filtered_context = result.get("intermediate_results", {}).get("filtered_context", "")

    evaluation = evaluator.evaluate(
        generated_answer=result.get("answer", ""),
        reference_answer=reference_answer,
        context_text=filtered_context,
    )
    result["answer_evaluation"] = evaluation


def run_single_query(
    query: str,
    args,
    generation_pipeline: MedicalGenerationPipeline,
    resources: RetrievalResources,
    evaluator: AnswerEvaluator,
    reference_answers: Dict[str, str],
) -> Dict[str, Any]:
    print("=" * 80)
    print("Query:", query)
    print("Device:", args.device)

    query_info = process_query(query)

    print("\nQuery info:")
    pprint(query_info, sort_dicts=False)

    retrieval_outputs = resources.retriever.retrieve(
        query_info=query_info,
        top_k_vector=args.top_k_vector,
        top_k_keyword=args.top_k_keyword,
        fusion_strategy=args.fusion_strategy,
        final_top_k=args.final_top_k,
    )

    reranked_results = rerank_results(
        query_info=query_info,
        results=retrieval_outputs["fused_results"],
        tokenizer=resources.reranker_tokenizer,
        model=resources.reranker_model,
        device=args.device,
    )

    format_top_results(reranked_results, args.show_top_n)

    result = generation_pipeline.run(query=query, reranked_results=reranked_results)
    evaluate_answer_if_needed(result, evaluator, reference_answers, args)

    print("\nFinal answer:\n")
    print(result["answer"])

    print("\nContext metadata:")
    pprint(result["context_metadata"], sort_dicts=False)

    print("\nGeneration metrics:")
    pprint(result["generation_metrics"], sort_dicts=False)

    if "answer_evaluation" in result:
        print("\nAnswer evaluation:")
        pprint(result["answer_evaluation"], sort_dicts=False)

    cache_obj = getattr(generation_pipeline.llm_generator, "cache", None)
    if cache_obj is not None:
        print("\nCache stats:")
        pprint(cache_obj.stats(), sort_dicts=False)

    print("\nSources:")
    pprint(result["sources"], sort_dicts=False)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, default=DEFAULT_TEST_QUERIES[0])

    parser.add_argument("--persist_dir", type=str, default="../output/chroma_db")
    parser.add_argument("--collection", type=str, default="oa_comm_full_bge_small")

    parser.add_argument(
        "--vector_model",
        type=str,
        default="../models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="../models/models--BAAI--bge-reranker-base/refs",
    )
    parser.add_argument("--bm25_cache", type=str, default="../output/bm25_index.pkl")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--top_k_vector", type=int, default=10)
    parser.add_argument("--top_k_keyword", type=int, default=10)
    parser.add_argument(
        "--fusion_strategy",
        type=str,
        default="rrf",
        choices=["rrf", "weighted", "simple"],
    )
    parser.add_argument("--final_top_k", type=int, default=8)
    parser.add_argument("--show_top_n", type=int, default=8)

    parser.add_argument("--ollama_model", type=str, required=True)
    parser.add_argument("--ollama_base_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_timeout", type=int, default=300)

    parser.add_argument("--max_context_tokens", type=int, default=1200)
    parser.add_argument("--generation_chunk_token_limit", type=int, default=450)
    parser.add_argument("--generation_prompt_safety_margin", type=int, default=200)
    parser.add_argument("--min_context_tokens", type=int, default=120)
    parser.add_argument("--evidence_context_budget", type=int, default=500)
    parser.add_argument("--filtered_context_budget", type=int, default=900)
    parser.add_argument("--review_context_budget", type=int, default=400)
    parser.add_argument("--dedup_threshold", type=float, default=0.8)
    parser.add_argument("--same_source_soft_limit", type=int, default=3)

    parser.add_argument("--default_temperature", type=float, default=0.1)
    parser.add_argument("--default_max_output_tokens", type=int, default=180)
    parser.add_argument("--debug_llm", action="store_true")

    parser.add_argument("--disable_evidence_evaluation", action="store_true")
    parser.add_argument("--disable_critical_review", action="store_true")

    parser.add_argument("--run_test_queries", action="store_true")
    parser.add_argument("--run_batch_test_queries", action="store_true")
    parser.add_argument("--batch_max_workers", type=int, default=None)

    parser.add_argument("--enable_answer_evaluation", action="store_true")
    parser.add_argument("--reference_answers_path", type=str, default="")

    parser.add_argument("--enable_cache", action="store_true")
    parser.add_argument("--cache_max_entries", type=int, default=128)
    parser.add_argument("--cache_ttl_seconds", type=int, default=3600)
    parser.add_argument("--cache_temperature_threshold", type=float, default=0.2)

    parser.add_argument("--log_path", type=str, default="../output/week8_generation_log.json")
    parser.add_argument("--llm_num_ctx", type=int, default=4096)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    generation_pipeline = build_generation_pipeline(args)
    resources = RetrievalResources(args)
    evaluator = AnswerEvaluator()
    reference_answers = load_reference_answers(args.reference_answers_path)

    if args.run_batch_test_queries:
        runner = BatchGenerationRunner(max_workers=args.batch_max_workers)
        results = runner.run_batch(
            DEFAULT_TEST_QUERIES,
            lambda q: run_single_query(
                q,
                args,
                generation_pipeline,
                resources,
                evaluator,
                reference_answers,
            ),
        )
        save_log(args.log_path, results)
        return

    if args.run_test_queries:
        all_results: List[Dict[str, Any]] = []
        for query in DEFAULT_TEST_QUERIES:
            all_results.append(
                run_single_query(
                    query,
                    args,
                    generation_pipeline,
                    resources,
                    evaluator,
                    reference_answers,
                )
            )
        save_log(args.log_path, all_results)
        return

    result = run_single_query(
        args.query,
        args,
        generation_pipeline,
        resources,
        evaluator,
        reference_answers,
    )
    save_log(args.log_path, result)


if __name__ == "__main__":
    main()