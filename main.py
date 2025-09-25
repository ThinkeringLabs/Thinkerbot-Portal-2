diff --git a//dev/null b/main.py
index 0000000000000000000000000000000000000000..cace5d79b7fa56fb3b2227d198fcdd47b189f9b0 100644
--- a//dev/null
+++ b/main.py
@@ -0,0 +1,160 @@
+"""Thinkering Assistant API.
+
+This module provides a small FastAPI application that allows fellows to log
+reflections and retrieve lightweight summaries of their recent activity.  The
+original implementation depended on the OpenAI Assistants API and therefore
+required external network access and an API key.  In this environment we keep
+state in memory and build a simple rule-based summariser instead so that the
+service runs out of the box.
+"""
+
+from __future__ import annotations
+
+from collections import deque
+from datetime import datetime
+import re
+from typing import Deque, Dict, Iterable, List
+
+from fastapi import FastAPI, HTTPException
+from pydantic import BaseModel, Field, validator
+
+# ---------------------------------------------------------------------------
+# Models
+# ---------------------------------------------------------------------------
+
+
+class LogEntry(BaseModel):
+    """A single log entry submitted by a fellow."""
+
+    fellow_name: str = Field(..., min_length=1, description="Name of the fellow")
+    entry: str = Field(..., min_length=1, description="Reflection or note to log")
+    timestamp: datetime | None = Field(
+        default=None, description="Optional timestamp supplied by the client"
+    )
+
+    @validator("entry")
+    def _strip_entry(cls, value: str) -> str:
+        cleaned = value.strip()
+        if not cleaned:
+            raise ValueError("entry must not be empty")
+        return cleaned
+
+    @validator("fellow_name")
+    def _strip_name(cls, value: str) -> str:
+        cleaned = value.strip()
+        if not cleaned:
+            raise ValueError("fellow_name must not be empty")
+        return cleaned
+
+    @validator("timestamp", pre=True, always=True)
+    def _ensure_timestamp(cls, value: datetime | None) -> datetime:
+        return value or datetime.utcnow()
+
+
+class StoredLog(BaseModel):
+    """Internal representation of a log entry."""
+
+    entry: str
+    timestamp: datetime
+
+
+# ---------------------------------------------------------------------------
+# Application state and helpers
+# ---------------------------------------------------------------------------
+
+app = FastAPI(title="Thinkering Assistant")
+
+# Maintain up to this many logs per fellow in memory.
+MAX_LOGS_PER_FELLOW = 50
+
+# Maps the canonical fellow name (lower-case) to their stored logs.
+fellow_logs: Dict[str, Deque[StoredLog]] = {}
+
+
+def _canonical_name(name: str) -> str:
+    return name.casefold()
+
+
+def _get_logs_bucket(name: str) -> Deque[StoredLog]:
+    key = _canonical_name(name)
+    if key not in fellow_logs:
+        fellow_logs[key] = deque(maxlen=MAX_LOGS_PER_FELLOW)
+    return fellow_logs[key]
+
+
+def _split_sentences(text: str) -> Iterable[str]:
+    # Fall back to a simple split if the regex fails (e.g. there are no sentence
+    # terminators).  This keeps the summary logic robust for short fragments.
+    parts = re.split(r"(?<=[.!?])\s+", text)
+    if not parts:
+        return []
+    return [segment.strip() for segment in parts if segment.strip()]
+
+
+def _summarise(logs: Iterable[StoredLog]) -> str:
+    """Produce a short summary from the fellow's stored logs.
+
+    The goal is to surface the most recent distinct thoughts.  We iterate over
+    the logs from newest to oldest, collect sentences, deduplicate them in a
+    case-insensitive manner, and return up to three bullet points joined by
+    newlines.  This keeps the behaviour predictable without relying on an
+    external LLM.
+    """
+
+    sentences: List[str] = []
+    for log in reversed(list(logs)):
+        sentences.extend(_split_sentences(log.entry))
+
+    seen: set[str] = set()
+    unique_sentences: List[str] = []
+    for sentence in sentences:
+        key = re.sub(r"\W+", " ", sentence).strip().casefold()
+        if not key or key in seen:
+            continue
+        seen.add(key)
+        unique_sentences.append(sentence)
+        if len(unique_sentences) == 3:
+            break
+
+    if not unique_sentences:
+        raise HTTPException(status_code=404, detail="No content available to summarise")
+
+    bullets = [f"• {sentence}" for sentence in unique_sentences]
+    return "\n".join(bullets)
+
+
+# ---------------------------------------------------------------------------
+# Routes
+# ---------------------------------------------------------------------------
+
+
+@app.post("/log", status_code=201)
+def create_log(log: LogEntry):
+    """Store a log entry for the given fellow."""
+
+    bucket = _get_logs_bucket(log.fellow_name)
+    bucket.append(StoredLog(entry=log.entry, timestamp=log.timestamp))
+    return {
+        "status": "success",
+        "fellow_name": log.fellow_name,
+        "log_count": len(bucket),
+    }
+
+
+@app.get("/summary/{fellow_name}")
+def get_summary(fellow_name: str):
+    """Return a brief summary of a fellow's recent logs."""
+
+    bucket = fellow_logs.get(_canonical_name(fellow_name))
+    if not bucket:
+        raise HTTPException(status_code=404, detail="No logs found for this fellow")
+
+    summary = _summarise(bucket)
+    return {"fellow_name": fellow_name, "summary": summary}
+
+
+@app.get("/")
+def healthcheck():
+    """Simple health-check endpoint."""
+
+    return {"status": "Thinkering Assistant is live!"}
