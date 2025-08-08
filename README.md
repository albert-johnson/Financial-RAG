# Finance RAG Service

A Retrieval-Augmented Generation (RAG) service tailored for finance-related documents. It supports multi-document ingestion, automatic updates on file changes, table extraction, snapshots of relevant pages, and chart generation upon request.

## Features
- Ingest PDFs, DOCX, CSV, XLSX from the `data/` directory or via API
- Auto-update index when documents are added/removed/modified (file watcher)
- Extract tabular data (PDF, DOCX, CSV, XLSX)
- Extract page snapshots/images for context
- Answer queries using RAG
- Generate charts from tabular data when requested

## Quickstart
1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create directories:
   ```bash
   mkdir -p data storage outputs/charts
   ```
4. Set your LLM API key (OpenAI is supported):
   ```bash
   export OPENAI_API_KEY=YOUR_KEY
   ```
5. Start the server:
   ```bash
   uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
   ```

## API
- POST `/ingest` {"paths": ["/abs/path/file1.pdf", "/abs/path/file2.xlsx"]} — optional; empty to scan `data/`
- POST `/query` {"query": "...", "chart": true} — returns answer, sources, snapshots, and optional chart path

## Notes
- Page snapshots are stored under `storage/page_images/`
- Extracted tables are stored under `storage/tables/` and indexed for semantic retrieval
- Vector store persists in `storage/vectorstore/`
- Charts saved to `outputs/charts/`

## Updating
The file watcher monitors `data/` for changes and updates the index automatically.