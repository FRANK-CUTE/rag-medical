from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional


class GenerationCache:
    def __init__(
        self,
        max_entries: int = 128,
        ttl_seconds: int = 3600,
        cache_temperature_threshold: float = 0.2,
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.cache_temperature_threshold = cache_temperature_threshold
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "evictions": 0,
            "expired": 0,
            "skipped_high_temperature": 0,
        }

    def build_key(
        self,
        query: str,
        context: str,
        stage_name: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "query": query or "",
            "context": context or "",
            "stage_name": stage_name or "",
            "model_name": model_name or "",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        self._evict_expired(now)
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        item = self._cache.pop(key)
        item["last_accessed_at"] = now
        self._cache[key] = item
        self._stats["hits"] += 1
        return item["value"]

    def set(self, key: str, value: Dict[str, Any], temperature: float) -> bool:
        if temperature > self.cache_temperature_threshold:
            self._stats["skipped_high_temperature"] += 1
            return False

        now = time.time()
        self._evict_expired(now)
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = {
            "value": value,
            "created_at": now,
            "last_accessed_at": now,
            "expires_at": now + self.ttl_seconds,
        }
        self._stats["writes"] += 1
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1
        return True

    def stats(self) -> Dict[str, Any]:
        now = time.time()
        self._evict_expired(now)
        return {
            **self._stats,
            "current_entries": len(self._cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "cache_temperature_threshold": self.cache_temperature_threshold,
        }

    def _evict_expired(self, now: float) -> None:
        expired_keys = [key for key, item in self._cache.items() if item.get("expires_at", 0) <= now]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._stats["expired"] += 1
