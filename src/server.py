from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.config import WATCH_DIRECTORIES, DATA_DIR
from src.rag.ingest import ingest_paths, ingest_data_dir, ingest_file
from src.rag.retriever import answer_query

app = FastAPI(title="Finance RAG Service")


class IngestRequest(BaseModel):
    paths: Optional[List[str]] = None


class QueryRequest(BaseModel):
    query: str
    chart: Optional[bool] = None


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    if req.paths:
        return {"results": ingest_paths(req.paths)}
    return {"results": ingest_data_dir()}


@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / file.filename
    with open(out_path, "wb") as f:
        f.write(await file.read())
    res = ingest_paths([str(out_path)])
    return {"saved_to": str(out_path), "ingest": res}


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    result = answer_query(req.query)
    return result


# Watcher
class WatcherThread(threading.Thread):
    def __init__(self, paths: List[str]):
        super().__init__(daemon=True)
        self.paths = paths
        self._stop = threading.Event()

    def run(self):
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except Exception:
            return

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return
                if Path(event.src_path).suffix.lower() in {".pdf", ".docx", ".csv", ".xlsx"}:
                    try:
                        # small debounce
                        time.sleep(0.5)
                        ingest_file(event.src_path)
                    except Exception:
                        pass

        observer = Observer()
        for p in self.paths:
            observer.schedule(Handler(), p, recursive=True)
        observer.start()
        try:
            while not self._stop.is_set():
                time.sleep(1.0)
        finally:
            observer.stop()
            observer.join()

    def stop(self):
        self._stop.set()


watcher = WatcherThread(WATCH_DIRECTORIES)


@app.on_event("startup")
async def on_startup():
    watcher.start()


@app.on_event("shutdown")
async def on_shutdown():
    watcher.stop()