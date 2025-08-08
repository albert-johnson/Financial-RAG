from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt

from src.config import CHARTS_DIR
from src.rag.tabular_extractor import load_table_df


@dataclass
class ChartSpec:
    chart_type: str  # line, bar, scatter, pie
    x: Optional[str]
    y: List[str]
    title: Optional[str]
    filters: Dict[str, Any]


CHART_KEYWORDS = [
    "chart", "plot", "graph", "visualize", "visualise", "figure",
    "line chart", "bar chart", "scatter", "pie"
]


def is_chart_request(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in CHART_KEYWORDS)


def _guess_chart_type(query: str, columns: List[str]) -> str:
    q = query.lower()
    if "scatter" in q:
        return "scatter"
    if "bar" in q or "hist" in q:
        return "bar"
    if "pie" in q:
        return "pie"
    return "line"


def _extract_column_mentions(query: str, columns: List[str]) -> List[str]:
    found = []
    for c in columns:
        pattern = re.compile(re.escape(str(c)), re.IGNORECASE)
        if pattern.search(query):
            found.append(c)
    return found


def _coerce_numeric(values: List[str]) -> List[float]:
    out = []
    for v in values:
        try:
            out.append(float(str(v).replace(",", "").strip()))
        except Exception:
            out.append(float('nan'))
    return out


def build_chart(
    query: str,
    table_path: str,
    override_spec: Optional[ChartSpec] = None,
) -> Optional[str]:
    tbl = load_table_df(table_path)
    columns = [str(c) for c in tbl.get("columns", [])]
    rows = tbl.get("rows", [])
    if not columns or not rows:
        return None

    spec = override_spec
    if spec is None:
        y_candidates = _extract_column_mentions(query, columns)
        if not y_candidates and len(columns) >= 2:
            y_candidates = [columns[1]]
        elif not y_candidates and len(columns) == 1:
            y_candidates = [columns[0]]
        x_candidates = [c for c in columns if c not in y_candidates]
        x_col = x_candidates[0] if x_candidates else None
        spec = ChartSpec(
            chart_type=_guess_chart_type(query, columns),
            x=x_col,
            y=y_candidates[:2] if y_candidates else [],
            title=query.strip().capitalize(),
            filters={},
        )

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / f"chart_{uuid.uuid4().hex[:8]}.png"

    # Build series
    data_by_col: Dict[str, List[str]] = {c: [r[i] if i < len(r) else "" for r in rows] for i, c in enumerate(columns)}
    x_values = data_by_col.get(spec.x) if spec.x else list(range(len(rows)))

    plt.figure(figsize=(10, 6))
    ct = spec.chart_type
    if ct == "line":
        for y in spec.y:
            y_vals = data_by_col.get(y, [])
            if spec.x:
                try:
                    xv = _coerce_numeric(x_values)  # try numeric x
                except Exception:
                    xv = list(range(len(y_vals)))
            else:
                xv = list(range(len(y_vals)))
            yv = _coerce_numeric(y_vals)
            plt.plot(xv, yv, label=y)
    elif ct == "bar":
        indices = list(range(len(rows)))
        width = 0.8 / max(1, len(spec.y))
        for idx, y in enumerate(spec.y or columns[:1]):
            y_vals = _coerce_numeric(data_by_col.get(y, []))
            offset = [(i + idx * width) for i in indices]
            plt.bar(offset, y_vals, width=width, label=y)
        plt.xticks([i + width for i in indices], x_values if spec.x else indices, rotation=45, ha='right')
    elif ct == "scatter" and spec.x and spec.y:
        xv = _coerce_numeric(x_values)
        yv = _coerce_numeric(data_by_col.get(spec.y[0], []))
        plt.scatter(xv, yv)
    elif ct == "pie" and spec.y:
        series = data_by_col.get(spec.y[0], [])
        # Count occurrences
        from collections import Counter
        counts = Counter(series)
        labels, values = zip(*counts.items()) if counts else ([], [])
        if values:
            plt.pie(values, labels=labels, autopct='%1.1f%%')
    else:
        # Fallback to line
        y = spec.y[0] if spec.y else columns[0]
        yv = _coerce_numeric(data_by_col.get(y, []))
        plt.plot(range(len(yv)), yv, label=y)

    plt.title(spec.title or "Chart")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)