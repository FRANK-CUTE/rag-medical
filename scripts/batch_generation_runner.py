from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence


class BatchGenerationRunner:
    def __init__(self, max_workers: Optional[int] = None):
        cpu_count = os.cpu_count() or 2
        self.max_workers = max_workers or max(1, min(4, cpu_count))

    def run_batch(
        self,
        items: Sequence[Any],
        worker_fn: Callable[[Any], Any],
    ) -> List[Dict[str, Any]]:
        ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(worker_fn, item): idx
                for idx, item in enumerate(items)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                item = items[idx]
                try:
                    result = future.result()
                    ordered_results[idx] = {
                        "index": idx,
                        "input": item,
                        "success": True,
                        "result": result,
                        "error": None,
                    }
                except Exception as exc:
                    ordered_results[idx] = {
                        "index": idx,
                        "input": item,
                        "success": False,
                        "result": None,
                        "error": str(exc),
                    }
        return [item for item in ordered_results if item is not None]
