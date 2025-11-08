"""
Interactive Rich-based CLI console for searching a SPLADE index.

Usage:
    python -m splade_easy.console ./my_index
or (with a console script):
    splade-search ./my_index
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from sentence_transformers import SentenceTransformer, SimilarityFunction

from splade_easy import SpladeIndex

THEME = Theme(
    {
        "banner": "bold cyan",
        "prompt": "bold green",
        "hint": "dim",
        "error": "bold red",
        "cmd": "bold magenta",
        "meta": "cyan",
        "score": "bold yellow",
    }
)


console = Console(theme=THEME)


def build_header(
    index_dir: Path, model_name: str, mode: str, top_k: int, num_workers: int
) -> Panel:
    title = Text("SPLADE-Easy Search Console", style="banner")
    subtitle = Text.assemble(
        ("index=", "hint"),
        (str(index_dir), "meta"),
        ("  model=", "hint"),
        (model_name, "meta"),
        ("  mode=", "hint"),
        (mode, "meta"),
        ("  top_k=", "hint"),
        (str(top_k), "meta"),
        ("  workers=", "hint"),
        (str(num_workers), "meta"),
    )
    body = Text.from_markup(
        "[hint]Type a query and press Enter to search.\n"
        "Commands: [cmd]:help[/cmd], [cmd]:stats[/cmd], "
        "[cmd]:topk N[/cmd], [cmd]:mode disk|memory[/cmd], [cmd]:quit[/cmd][/hint]"
    )

    content = Group(
        title,
        subtitle,
        Text(),  # blank line
        body,
    )

    return Panel(
        Align.left(content),
        border_style="banner",
    )


def render_results(
    query: str,
    results,
    similarity_scores: list[float] | None,
    show_text: bool = True,
) -> None:
    if not results:
        console.print(
            Panel(
                Text.assemble(
                    ("No results for ", "error"),
                    (repr(query), "error"),
                ),
                border_style="error",
            )
        )
        return

    table = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE_HEAVY,
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", style="score", width=8)
    table.add_column("Sim", justify="right", style="score", width=8)
    table.add_column("Doc ID", style="bold")
    if show_text:
        table.add_column("Text", overflow="fold", ratio=3)
    table.add_column("Metadata", overflow="fold", style="meta", ratio=2)

    for i, r in enumerate(results, 1):
        meta = r.metadata or {}
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""

        # Retrieval score
        score_str = f"{r.score:.3f}"

        # Similarity score (if provided)
        if similarity_scores is not None and i - 1 < len(similarity_scores):
            sim_val = similarity_scores[i - 1]
            sim_str = f"{sim_val:.3f}"
        else:
            sim_str = ""

        row = [
            str(i),
            score_str,
            sim_str,
            str(r.doc_id),
        ]
        if show_text:
            row.append(r.text or "")
        row.append(meta_str)
        table.add_row(*row)

    console.print(
        Panel(
            Align.left(table),
            title=f" Results for {query!r} ",
            border_style="score",
        )
    )


def print_stats(index: SpladeIndex) -> None:
    stats = index.stats()
    table = Table(
        show_header=False,
        box=box.SIMPLE,
        expand=False,
    )
    for key in ("num_docs", "num_shards", "deleted_docs", "total_size_mb"):
        if key in stats:
            table.add_row(str(key), str(stats[key]))
    console.print(
        Panel(
            table,
            title=" Index Stats ",
            border_style="banner",
        )
    )


def parse_command(line: str) -> tuple[str | None, list[str]]:
    """Return (command, args) if line starts with ':', otherwise (None, [])."""
    if not line.startswith(":"):
        return None, []
    parts = line[1:].strip().split()
    if not parts:
        return None, []
    return parts[0].lower(), parts[1:]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive SPLADE-Easy search console.",
    )
    parser.add_argument(
        "index_dir",
        type=Path,
        help="Path to an existing SPLADE index directory.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="naver/splade-v3",
        help="Sentence-Transformers SPLADE model name (default: naver/splade-v3).",
    )
    parser.add_argument(
        "--mode",
        choices=("disk", "memory"),
        default="disk",
        help="Retriever mode: 'disk' or 'memory' (default: disk).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return per query (default: 5).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for search (default: 4).",
    )
    args = parser.parse_args(argv)

    index_dir: Path = args.index_dir
    model_name: str = args.model
    mode: str = args.mode
    top_k: int = max(1, args.top_k)
    num_workers: int = max(1, args.num_workers)

    if not index_dir.exists():
        console.print(f"[error]Index directory does not exist: {index_dir}[/error]")
        raise SystemExit(1)

    console.print(build_header(index_dir, model_name, mode, top_k, num_workers))

    console.print("Loading SPLADE model...", style="hint")
    model = SentenceTransformer(model_name)

    console.print("Opening index and retriever...", style="hint")
    index = SpladeIndex(str(index_dir))
    retriever = SpladeIndex.retriever(str(index_dir), mode=mode)

    console.print(Panel("Ready. Type a query or :help for commands.", border_style="banner"))

    try:
        while True:
            try:
                line = Prompt.ask(
                    "[prompt]search>[/prompt]",
                    default="",
                    show_default=False,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[hint]Exiting.[/hint]")
                break

            if not line:
                continue

            cmd, cmd_args = parse_command(line)
            if cmd is not None:
                # Command handling
                if cmd in ("quit", "exit", "q"):
                    console.print("[hint]Goodbye.[/hint]")
                    break
                elif cmd == "help":
                    console.print(
                        Panel(
                            Align.left(
                                Text.from_markup(
                                    "Commands:\n"
                                    "  [cmd]:help[/cmd]              Show this help.\n"
                                    "  [cmd]:stats[/cmd]             Show index statistics.\n"
                                    "  [cmd]:topk N[/cmd]            Set number of results per query.\n"
                                    "  [cmd]:mode disk|memory[/cmd]  Switch retriever mode.\n"
                                    "  [cmd]:quit[/cmd]              Exit the console.\n"
                                )
                            ),
                            title=" Help ",
                            border_style="banner",
                        )
                    )
                elif cmd == "stats":
                    print_stats(index)
                elif cmd == "topk":
                    if not cmd_args:
                        console.print("[error]Usage: :topk N[/error]")
                    else:
                        try:
                            new_top_k = int(cmd_args[0])
                            if new_top_k < 1:
                                raise ValueError
                            top_k = new_top_k
                            console.print(f"[hint]top_k set to {top_k}[/hint]")
                        except ValueError:
                            console.print("[error]Invalid value for top_k.[/error]")
                elif cmd == "mode":
                    if not cmd_args or cmd_args[0] not in ("disk", "memory"):
                        console.print("[error]Usage: :mode disk|memory[/error]")
                    else:
                        new_mode = cmd_args[0]
                        if new_mode != mode:
                            mode = new_mode
                            retriever = SpladeIndex.retriever(str(index_dir), mode=mode)
                            console.print(f"[hint]Switched mode to {mode}[/hint]")
                else:
                    console.print(f"[error]Unknown command: :{cmd}[/error]")
                continue

            # Normal search query
            query = line
            console.print(f"[hint]Searching for {query!r}...[/hint]")
            try:
                results = retriever.search_text(
                    query=query,
                    model=model,
                    top_k=top_k,
                    return_text=True,
                    num_workers=num_workers,
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                console.print(f"[error]Search failed: {exc}[/error]")
                continue

            # Compute cosine similarity scores using sentence-transformers
            similarity_scores: list[float] | None = None
            try:
                texts = [r.text or "" for r in results]
                if texts:
                    # Encode query and documents
                    query_emb = model.encode([query])  # shape (1, d)
                    doc_embs = model.encode(texts)  # shape (k, d)

                    sim_fn = SimilarityFunction.to_similarity_fn("cosine")
                    sim_matrix = sim_fn(query_emb, doc_embs)  # shape (1, k)

                    # Convert to a simple Python list of floats
                    try:
                        similarity_scores = sim_matrix[0].tolist()
                    except Exception:
                        similarity_scores = sim_matrix.tolist()
            except Exception as exc:
                console.print(f"[error]Failed to compute similarity scores: {exc}[/error]")

            render_results(query, results, similarity_scores=similarity_scores, show_text=True)

    finally:
        # Retrievers typically do not need explicit cleanup, but hook left here.
        pass


if __name__ == "__main__":
    main()
