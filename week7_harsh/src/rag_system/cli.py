"""
CLI Interface
=============
Command-line interface for the RAG system, built with Typer + Rich.

Commands
--------
rag ingest <path>       Ingest a document or directory.
rag query  <question>   Ask a question against ingested documents.
rag status              Show pipeline and vector store status.
rag clear               Clear all stored documents.
rag eval  <qa_file>     Run RAGAS evaluation on a Q&A JSONL file.

Examples
--------
    # Ingest a single PDF
    rag ingest report.pdf

    # Ingest an entire folder
    rag ingest ./docs

    # Ask a question
    rag query "What is the main conclusion of the paper?"

    # Ask with verbose context display
    rag query "What datasets were used?" --show-context

    # Run evaluation
    rag eval eval_set.jsonl

    # Use local embedding model (offline)
    rag --embedding-backend local ingest report.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="rag",
    help="RAG System — Retrieval-Augmented Generation for document Q&A",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# Global state — pipeline is built lazily
_pipeline = None


def _get_pipeline(embedding_backend: Optional[str] = None, reranker: str = "flashrank"):
    """Return the shared pipeline singleton (created once per process)."""
    global _pipeline
    if _pipeline is None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Initialising pipeline…"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            progress.add_task("init", total=None)
            from rag_system.pipeline import RAGPipeline
            _pipeline = RAGPipeline.from_config(
                embedding_backend=embedding_backend,
                reranker_backend=reranker,
            )
    return _pipeline


# ---------------------------------------------------------------------------
# Global options callback
# ---------------------------------------------------------------------------

@app.callback()
def main(
    ctx: typer.Context,
    embedding_backend: str = typer.Option(
        None,
        "--embedding-backend",
        "-e",
        help="Embedding backend: 'openai' or 'local'",
        envvar="EMBEDDING_BACKEND",
    ),
    reranker: str = typer.Option(
        "flashrank",
        "--reranker",
        "-r",
        help="Reranker backend: 'flashrank', 'cross-encoder', or 'none'",
    ),
) -> None:
    """[bold]RAG System[/bold] — document ingestion and Q&A"""
    ctx.ensure_object(dict)
    ctx.obj["embedding_backend"] = embedding_backend
    ctx.obj["reranker"] = reranker


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Path to a PDF/TXT/MD file or a directory"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse into subdirectories"),
    chunking: str = typer.Option("recursive", "--chunking", "-c", help="Chunking strategy: 'recursive' or 'semantic'"),
) -> None:
    """[bold green]Ingest[/bold green] a document or directory into the vector store."""
    pipeline = _get_pipeline(
        embedding_backend=ctx.obj.get("embedding_backend"),
        reranker=ctx.obj.get("reranker", "flashrank"),
    )

    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Path not found: {path}")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task(f"Ingesting [cyan]{path}[/cyan]…", total=None)
        try:
            # Apply chunking strategy if overridden
            pipeline.chunking = chunking
            results = pipeline.ingest(path, recursive=recursive)
            progress.update(task, completed=True)
        except Exception as exc:
            progress.stop()
            console.print(f"[bold red]Ingestion failed:[/bold red] {exc}")
            raise typer.Exit(code=1)

    # Build a results table
    table = Table(title="Ingestion Results", show_header=True, header_style="bold cyan")
    table.add_column("Source", style="dim", no_wrap=False, max_width=50)
    table.add_column("Docs", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Stored", justify="right")
    table.add_column("Status")

    for r in results:
        status = "[red]ERROR[/red]" if r.error else "[green]OK[/green]"
        table.add_row(
            Path(r.source).name,
            str(r.num_documents),
            str(r.num_chunks),
            str(r.num_stored),
            status,
        )

    console.print(table)
    total_chunks = sum(r.num_chunks for r in results if not r.error)
    console.print(
        f"\n[bold]Total:[/bold] {len(results)} file(s), {total_chunks} chunk(s) ingested. "
        f"Vector store now has [bold cyan]{pipeline.vector_store.count()}[/bold cyan] chunk(s)."
    )


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------

@app.command()
def query(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="The question to answer"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Candidates to retrieve"),
    top_n: int = typer.Option(5, "--top-n", "-n", help="Contexts to pass to LLM after reranking"),
    show_context: bool = typer.Option(False, "--show-context", help="Print the retrieved context chunks"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    source_filter: Optional[str] = typer.Option(None, "--source", help="Filter by filename (e.g. report.pdf)"),
) -> None:
    """[bold green]Query[/bold green] the document store with a natural-language question."""
    pipeline = _get_pipeline(
        embedding_backend=ctx.obj.get("embedding_backend"),
        reranker=ctx.obj.get("reranker", "flashrank"),
    )

    metadata_filter = {"filename": source_filter} if source_filter else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task(f"Searching for: [italic]{question[:60]}[/italic]…", total=None)
        try:
            result = pipeline.query(
                question,
                top_k=top_k,
                top_n=top_n,
                metadata_filter=metadata_filter,
            )
        except RuntimeError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    if output_json:
        output = {
            "question": result.question,
            "answer": result.answer,
            "sources": result.sources,
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens,
                "total": result.total_tokens,
            },
        }
        rprint(output)
        return

    # Pretty output
    console.print()
    console.print(Panel(
        Text(result.answer, style="white"),
        title=f"[bold cyan]Answer[/bold cyan]",
        subtitle=f"model={result.model}  tokens={result.total_tokens}",
        border_style="cyan",
        expand=False,
    ))

    if result.sources:
        console.print()
        src_table = Table(title="Sources", show_header=True, header_style="bold magenta")
        src_table.add_column("#", justify="right", width=3)
        src_table.add_column("File")
        src_table.add_column("Page", justify="right")
        src_table.add_column("Score", justify="right")
        for s in result.sources:
            src_table.add_row(
                str(s["source_num"]),
                str(s.get("filename", "?")),
                str(s.get("page", "?")),
                f"{s.get('score', 0):.4f}",
            )
        console.print(src_table)

    if show_context:
        console.print()
        for i, ctx_text in enumerate(result.contexts_used, start=1):
            console.print(Panel(
                ctx_text[:800] + ("…" if len(ctx_text) > 800 else ""),
                title=f"[dim]Context [{i}][/dim]",
                border_style="dim",
                expand=False,
            ))


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

@app.command()
def status(ctx: typer.Context) -> None:
    """Show the current [bold]pipeline status[/bold] and vector store statistics."""
    pipeline = _get_pipeline(
        embedding_backend=ctx.obj.get("embedding_backend"),
        reranker=ctx.obj.get("reranker", "flashrank"),
    )
    info = pipeline.status()

    table = Table(title="RAG Pipeline Status", show_header=False, box=None)
    table.add_column("Key", style="bold cyan", width=30)
    table.add_column("Value")

    for k, v in info.items():
        table.add_row(k.replace("_", " ").title(), str(v))

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# clear command
# ---------------------------------------------------------------------------

@app.command()
def clear(
    ctx: typer.Context,
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """[bold red]Clear[/bold red] all documents from the vector store."""
    if not confirm:
        confirmed = typer.confirm(
            "This will delete ALL ingested documents. Are you sure?", default=False
        )
        if not confirmed:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    pipeline = _get_pipeline(
        embedding_backend=ctx.obj.get("embedding_backend"),
        reranker=ctx.obj.get("reranker", "flashrank"),
    )
    pipeline.clear()
    console.print("[bold red]Vector store cleared.[/bold red] All documents removed.")


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------

@app.command()
def eval(
    ctx: typer.Context,
    qa_file: Path = typer.Argument(
        ...,
        help="Path to a JSONL file with fields: question, ground_truth (optional: answer)",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to a JSON file"),
    top_k: int = typer.Option(20, "--top-k", "-k"),
    top_n: int = typer.Option(5, "--top-n", "-n"),
    sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Evaluate only N questions"),
) -> None:
    """
    [bold green]Evaluate[/bold green] retrieval and generation quality using RAGAS metrics.

    The QA file must be a JSONL file where each line contains:
        {"question": "...", "ground_truth": "..."}
    """
    if not qa_file.exists():
        console.print(f"[bold red]File not found:[/bold red] {qa_file}")
        raise typer.Exit(code=1)

    pipeline = _get_pipeline(
        embedding_backend=ctx.obj.get("embedding_backend"),
        reranker=ctx.obj.get("reranker", "flashrank"),
    )

    # Load QA pairs
    qa_pairs = []
    with qa_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_pairs.append(json.loads(line))

    if sample:
        qa_pairs = qa_pairs[:sample]

    console.print(f"Loaded [bold]{len(qa_pairs)}[/bold] Q&A pairs from {qa_file.name}")

    # Run evaluation via the evaluation module
    from rag_system.evaluation import Evaluator

    evaluator = Evaluator(pipeline)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Running evaluation…", total=len(qa_pairs))
        report = evaluator.evaluate(
            qa_pairs,
            top_k=top_k,
            top_n=top_n,
            progress_callback=lambda: progress.advance(task),
        )

    # Print summary
    summary_table = Table(title="Evaluation Summary", header_style="bold cyan")
    summary_table.add_column("Metric")
    summary_table.add_column("Score", justify="right")

    for metric, value in report["summary"].items():
        color = "green" if value >= 0.7 else "yellow" if value >= 0.5 else "red"
        summary_table.add_row(metric, f"[{color}]{value:.4f}[/{color}]")

    console.print()
    console.print(summary_table)

    if output:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        console.print(f"\nFull report saved to [cyan]{output}[/cyan]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cli_main() -> None:
    app()


if __name__ == "__main__":
    cli_main()
