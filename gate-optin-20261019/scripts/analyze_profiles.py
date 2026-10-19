"""Summarize clipped GPU activity unions within synchronized training-step ranges."""

import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def length(intervals):
    return sum(end - start for start, end in union(intervals))


def intersect(left, right):
    left, right = union(left), union(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path):
    c = sqlite3.connect(path)
    kinds = dict(c.execute("select id,label from ENUM_CUDA_MEMCPY_OPER"))
    results = []
    for label, start, end in c.execute(
        "select text,start,end from NVTX_EVENTS where text like 'training_step:%' order by start"
    ):
        kernels = c.execute(
            "select max(k.start,?),min(k.end,?),s.value,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start<? and k.end>?",
            (start, end, end, start),
        ).fetchall()
        copies = c.execute(
            "select max(start,?),min(end,?),bytes,copyKind,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where start<? and end>?",
            (start, end, end, start),
        ).fetchall()
        k = [(a, b) for a, b, *_ in kernels]
        m = [(a, b) for a, b, *_ in copies]
        overlap = intersect(k, m)
        record = dict(
            label=label,
            wall_ms=(end - start) / 1e6,
            kernel_busy_union_ms=length(k) / 1e6,
            memcpy_busy_union_ms=length(m) / 1e6,
            copy_without_kernel_ms=(length(m) - overlap) / 1e6,
            kernel_copy_overlap_ms=overlap / 1e6,
            neither_kernel_nor_copy_ms=((end - start) - length(k + m)) / 1e6,
        )
        record["copies"] = {}
        for kind in sorted({x[3] for x in copies}):
            rows = [x for x in copies if x[3] == kind]
            record["copies"][kinds[kind]] = dict(
                count=len(rows),
                bytes=sum(x[2] for x in rows),
                sum_ms=sum(x[1] - x[0] for x in rows) / 1e6,
                union_ms=length([(x[0], x[1]) for x in rows]) / 1e6,
                streams=sorted({x[4] for x in rows}),
            )
        totals = {}
        for a, b, name, stream in kernels:
            total = totals.setdefault(name, dict(count=0, sum_ms=0))
            total["count"] += 1
            total["sum_ms"] += (b - a) / 1e6
        record["top_kernels"] = dict(
            sorted(totals.items(), key=lambda x: x[1]["sum_ms"], reverse=True)[:15]
        )
        record["apis"] = [
            dict(name=name, count=count, sum_ms=duration / 1e6)
            for name, count, duration in c.execute(
                "select s.value,count(*),sum(min(r.end,?)-max(r.start,?)) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start<? and r.end>? group by s.value order by 3 desc limit 12",
                (end, start, end, start),
            )
        ]
        results.append(record)
    c.close()
    assert len(results) == 3, (path, len(results))
    return results


if __name__ == "__main__":
    output = {}
    for arm in ("baseline", "candidate"):
        path = ROOT / f"profile-{arm}" / "trace.sqlite"
        rows = analyze(path)
        connection = sqlite3.connect(path)
        for row in rows:
            start, end = connection.execute(
                "select start,end from NVTX_EVENTS where text=?", (row["label"],)
            ).fetchone()
            row["all_kernels"] = {
                name: dict(count=count, sum_ms=duration / 1e6)
                for name, count, duration in connection.execute(
                    "select s.value,count(*),sum(min(k.end,?)-max(k.start,?)) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start<? and k.end>? group by s.value",
                    (end, start, end, start),
                )
            }
        connection.close()
        output[arm] = rows
    (ROOT / "profile-analysis.json").write_text(json.dumps(output, indent=2) + "\n")
