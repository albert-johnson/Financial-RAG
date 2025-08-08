from __future__ import annotations

import argparse
import json

from src.rag.ingest import ingest_paths, ingest_data_dir
from src.rag.retriever import answer_query


def main():
    parser = argparse.ArgumentParser(description="Finance RAG CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("paths", nargs="*", help="File paths to ingest; empty to scan data/")

    p_query = sub.add_parser("query")
    p_query.add_argument("query", type=str)

    args = parser.parse_args()

    if args.cmd == "ingest":
        if args.paths:
            res = ingest_paths(args.paths)
        else:
            res = ingest_data_dir()
        print(json.dumps(res, indent=2))
    elif args.cmd == "query":
        res = answer_query(args.query)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()