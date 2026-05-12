"""Textual TUI for splade-easy. Interactive HF-dataset indexing + search.

Run with `uv run splade-tui` (after `uv sync --extra tui`).

All indexes go into a single folder (default `~/.splade-easy/indexes/`, override
with `SPLADE_EASY_INDEX_DIR`). The sidebar lists every index there and lets you
flip between them without leaving the search context.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, VerticalScroll
    from textual.reactive import reactive
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        ContentSwitcher,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        LoadingIndicator,
        Select,
        SelectionList,
        Static,
    )
    from textual.widgets.selection_list import Selection
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "splade-easy[tui] is required to run the TUI. "
        "Install with: uv sync --extra tui  (or: uv add 'splade-easy[tui]')"
    ) from exc

from . import SpladeRetriever, encode_corpus
from .models import DEFAULT_MODEL, KNOWN_MODELS

# ---------- index folder + slug helpers ----------


def _default_index_dir() -> Path:
    raw = os.environ.get("SPLADE_EASY_INDEX_DIR", "~/.splade-easy/indexes")
    return Path(raw).expanduser()


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("_", name).strip("_")
    return slug or "index"


@dataclass
class IndexInfo:
    name: str  # subdir name
    path: Path
    n_docs: int
    model_id: str
    size_bytes: int


def _list_indexes(root: Path) -> list[IndexInfo]:
    if not root.exists():
        return []
    out: list[IndexInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        params = child / "params.json"
        if not params.exists():
            continue
        try:
            import json

            meta = json.loads(params.read_text())
        except Exception:
            continue
        size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
        out.append(
            IndexInfo(
                name=child.name,
                path=child,
                n_docs=int(meta.get("n_docs", 0)),
                model_id=str(meta.get("model_id", "?")),
                size_bytes=size,
            )
        )
    return out


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# ---------- HF dataset metadata ----------


@dataclass
class DatasetMeta:
    configs: list[str]
    default_config: str | None
    splits_by_config: dict[str, list[str]] = field(default_factory=dict)
    columns_by_config: dict[str, list[str]] = field(default_factory=dict)


def _fetch_dataset_metadata(repo: str) -> DatasetMeta:
    """Return available configs/splits/columns without downloading rows.

    `datasets` 4.x no longer supports a `trust_remote_code` kwarg (script-based
    datasets aren't executed anymore); the parquet-backed view is used directly.
    Any error from `datasets` is re-raised so the caller can show the real
    cause rather than falling back to bogus defaults.
    """
    from datasets import get_dataset_config_names, load_dataset_builder

    configs = get_dataset_config_names(repo)
    if not configs:
        configs = [None]  # dataset with no named configs

    splits_by_config: dict[str, list[str]] = {}
    cols_by_config: dict[str, list[str]] = {}
    first_err: Exception | None = None
    for cfg in configs:
        try:
            builder = load_dataset_builder(repo, cfg) if cfg else load_dataset_builder(repo)
        except Exception as e:
            if first_err is None:
                first_err = e
            continue
        info = builder.info
        splits_by_config[cfg or ""] = sorted((info.splits or {}).keys())
        cols_by_config[cfg or ""] = list((info.features or {}).keys())

    if not splits_by_config:
        raise RuntimeError(f"Could not load metadata for {repo}: {first_err}") from first_err

    return DatasetMeta(
        configs=[c or "" for c in configs if (c or "") in splits_by_config],
        default_config=(configs[0] or "") if configs else "",
        splits_by_config=splits_by_config,
        columns_by_config=cols_by_config,
    )


def _load_dataset_rows(
    repo: str,
    config: str | None,
    split: str,
    columns: list[str],
    max_docs: int | None,
) -> tuple[list[str], list[dict]]:
    """Load rows, join the chosen text columns. Returns (texts, raw_rows)."""
    from datasets import load_dataset

    ds = (
        load_dataset(repo, config or None, split=split)
        if config
        else load_dataset(repo, split=split)
    )
    n = len(ds)
    if max_docs is not None and max_docs > 0:
        n = min(n, max_docs)

    texts: list[str] = []
    raw: list[dict] = []
    for i in range(n):
        row = ds[i]
        text = "\n".join(str(row[c]) for c in columns if c in row and row[c] is not None)
        texts.append(text)
        raw.append({k: row[k] for k in row})
    return texts, raw


# ---------- modal screens ----------


class ResultDetail(ModalScreen[None]):
    """Full-doc viewer; Esc closes."""

    BINDINGS = [Binding("escape", "dismiss", "Close")]
    CSS = """
    ResultDetail { align: center middle; }
    ResultDetail > VerticalScroll {
        width: 80%;
        max-height: 80%;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(f"[b]{self._title}[/b]\n\n{self._body}")


class ConfirmDelete(ModalScreen[bool]):
    """Yes/no for deleting an index."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
    ]
    CSS = """
    ConfirmDelete { align: center middle; }
    ConfirmDelete > Container {
        width: 50; height: auto; padding: 1 2;
        border: round $error; background: $surface;
    }
    ConfirmDelete Button { margin: 1; }
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(f"Delete index [b]{self._name}[/b]?")
            with Horizontal():
                yield Button("Delete", id="yes", variant="error")
                yield Button("Cancel", id="no")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#yes")
    def _y(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _n(self) -> None:
        self.dismiss(False)


# ---------- main-area panels ----------


class WelcomePanel(Container):
    DEFAULT_CSS = "WelcomePanel { padding: 2 3; }"

    def compose(self) -> ComposeResult:
        yield Static(
            "[b]splade-easy[/b]\n\n"
            "Pick an index from the sidebar to start searching,\n"
            "or press [b]Ctrl+N[/b] (or click [b]+ new index[/b]) to index a HuggingFace dataset.\n\n"
            "Indexes live in [i]" + str(_default_index_dir()) + "[/i].\n"
            "Override the location with the [b]SPLADE_EASY_INDEX_DIR[/b] env var."
        )


class NewIndexPanel(Container):
    """Form for fetching + indexing a HF dataset."""

    DEFAULT_CSS = """
    NewIndexPanel { padding: 1 2; }
    NewIndexPanel .label { padding-top: 1; color: $accent; }
    NewIndexPanel Input { margin-bottom: 0; }
    NewIndexPanel #preview { color: $text-muted; padding: 1 0; max-height: 10; }
    NewIndexPanel #error { color: $error; padding: 1 0; min-height: 1; }
    NewIndexPanel #fetch { margin: 1 0; }
    NewIndexPanel #cols {
        min-height: 7;
        max-height: 12;
        border: round $primary;
        margin-bottom: 1;
    }
    NewIndexPanel #cols_label { color: $accent; padding-top: 1; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label("HuggingFace dataset id", classes="label")
            yield Input(
                placeholder="e.g. BeIR/scifact  (press Enter or click Fetch)",
                id="repo",
            )
            yield Button("Fetch metadata", id="fetch", variant="primary")
            yield Static("", id="error")

            yield Label("Config (subset)", classes="label")
            # Start blank; populated after `Fetch metadata` succeeds.
            yield Select[str](options=[], id="config_set", prompt="(fetch metadata first)")
            yield Label("Split", classes="label")
            yield Select[str](options=[], id="split_set", prompt="(fetch metadata first)")
            yield Static(
                "Text columns (fetch a dataset to populate)",
                id="cols_label",
            )
            yield SelectionList[str](id="cols")
            yield Static("", id="preview")

            yield Label("Model", classes="label")
            yield Select[str](
                options=[
                    (
                        f"{mid.split('/')[-1]}" + ("  (default)" if mid == DEFAULT_MODEL else ""),
                        mid,
                    )
                    for mid, spec in KNOWN_MODELS.items()
                    if spec.working
                ],
                value=DEFAULT_MODEL,
                id="model_set",
                allow_blank=False,
            )

            yield Label("Index name (subdir under the indexes folder)", classes="label")
            yield Input(id="name")

            yield Label("Max docs (blank = all)", classes="label")
            yield Input(placeholder="e.g. 10000", id="max_docs")

            yield Button("Index", id="index_btn", variant="primary")


class IndexingPanel(Container):
    DEFAULT_CSS = """
    IndexingPanel { padding: 2 3; }
    IndexingPanel #status { padding-bottom: 1; }
    IndexingPanel #detail { color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        yield Static("", id="detail")
        yield LoadingIndicator()


class SearchPanel(Container):
    """Query input + DataTable. Reactive `current_index` drives display."""

    DEFAULT_CSS = """
    SearchPanel { padding: 1 2; }
    SearchPanel #info { color: $accent; padding-bottom: 1; }
    SearchPanel #query_row { height: 3; }
    SearchPanel #k { width: 8; }
    SearchPanel #results { height: 1fr; }
    SearchPanel #footnote { color: $text-muted; padding-top: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="info")
        with Horizontal(id="query_row"):
            yield Input(placeholder="Type a query and press Enter…", id="query")
            yield Input(value="10", id="k")
        table: DataTable = DataTable(id="results", zebra_stripes=True, cursor_type="row")
        table.add_columns("#", "Score", "Doc")
        yield table
        yield Static("", id="footnote")


# ---------- sidebar ----------


class Sidebar(Container):
    """Persistent list of indexes + a button to create a new one."""

    DEFAULT_CSS = """
    Sidebar { width: 32; border-right: solid $primary; padding: 1; }
    Sidebar #title { padding-bottom: 1; color: $accent; }
    Sidebar ListView { height: 1fr; }
    Sidebar #new_btn { width: 100%; margin-top: 1; }
    Sidebar .empty { color: $text-muted; padding: 1 0; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Indexes (0)", id="title")
        yield ListView(id="index_list")
        yield Button("+ new index", id="new_btn", variant="primary")


# ---------- main app ----------


class SpladeTUI(App):
    CSS = """
    Screen { layout: horizontal; }
    Sidebar { dock: left; }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_index", "New", show=True),
        Binding("ctrl+r", "refresh_sidebar", "Refresh", show=True),
        Binding("delete", "delete_focused", "Delete", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    current_index: reactive[IndexInfo | None] = reactive(None)
    _retriever: SpladeRetriever | None = None

    def __init__(self, indexes_dir: Path | None = None, prefill_dataset: str | None = None) -> None:
        super().__init__()
        self.indexes_dir = indexes_dir or _default_index_dir()
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self._prefill_dataset = prefill_dataset
        self._dataset_meta: DatasetMeta | None = None

    # ---- composition ----

    def compose(self) -> ComposeResult:
        yield Header()
        yield Sidebar()
        with ContentSwitcher(initial="welcome", id="main"):
            yield WelcomePanel(id="welcome")
            yield NewIndexPanel(id="new")
            yield IndexingPanel(id="indexing")
            yield SearchPanel(id="search")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "splade-easy"
        self.sub_title = str(self.indexes_dir)
        self._refresh_sidebar()
        if self._prefill_dataset:
            self.action_new_index()
            self.query_one("#repo", Input).value = self._prefill_dataset

    # ---- sidebar ----

    def _refresh_sidebar(self) -> None:
        infos = _list_indexes(self.indexes_dir)
        title = self.query_one("#title", Static)
        title.update(f"Indexes ({len(infos)})")
        listview = self.query_one("#index_list", ListView)
        listview.clear()
        if not infos:
            listview.append(ListItem(Static("(no indexes yet)", classes="empty")))
        for info in infos:
            label = f"{info.name}\n  {info.n_docs:,} docs · {_human_size(info.size_bytes)}"
            item = ListItem(Static(label))
            item.data = info  # type: ignore[attr-defined]
            listview.append(item)

    # ---- actions ----

    def action_refresh_sidebar(self) -> None:
        self._refresh_sidebar()

    def action_new_index(self) -> None:
        self.query_one("#main", ContentSwitcher).current = "new"
        self._reset_new_form()
        self.query_one("#repo", Input).focus()

    def action_delete_focused(self) -> None:
        listview = self.query_one("#index_list", ListView)
        item = listview.highlighted_child
        if item is None or not hasattr(item, "data"):
            return
        info: IndexInfo = item.data  # type: ignore[attr-defined]

        def _do_delete(confirmed: bool | None) -> None:
            if confirmed:
                shutil.rmtree(info.path, ignore_errors=True)
                if self.current_index == info:
                    self.current_index = None
                    self._retriever = None
                    self.query_one("#main", ContentSwitcher).current = "welcome"
                self._refresh_sidebar()

        self.push_screen(ConfirmDelete(info.name), _do_delete)

    @on(Button.Pressed, "#new_btn")
    def _on_new_btn(self) -> None:
        self.action_new_index()

    # ---- sidebar selection -> load index -> search ----

    @on(ListView.Selected, "#index_list")
    def _on_sidebar_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if not hasattr(item, "data"):
            return
        info: IndexInfo = item.data  # type: ignore[attr-defined]
        self._load_and_switch_to_search(info)

    def _load_and_switch_to_search(self, info: IndexInfo) -> None:
        try:
            self._retriever = SpladeRetriever.load(info.path, mmap=True, load_corpus=True)
        except Exception as e:
            self.notify(f"Failed to load {info.name}: {e}", severity="error")
            return
        self.current_index = info
        switcher = self.query_one("#main", ContentSwitcher)
        switcher.current = "search"
        info_widget = self.query_one("#info", Static)
        info_widget.update(f"[b]{info.name}[/b] · {info.n_docs:,} docs · model: {info.model_id}")
        self.query_one("#results", DataTable).clear()
        footnote = self.query_one("#footnote", Static)
        footnote.update("Enter a query; press Enter on a row to view the full doc.")
        self.query_one("#query", Input).focus()

    # ---- new-index form ----

    def _reset_new_form(self) -> None:
        for w_id in ("repo", "name", "max_docs"):
            self.query_one(f"#{w_id}", Input).value = ""
        self.query_one("#preview", Static).update("")
        self.query_one("#error", Static).update("")
        # Clear config/split dropdowns until metadata is fetched.
        self.query_one("#config_set", Select).set_options([])
        self.query_one("#split_set", Select).set_options([])
        self.query_one("#cols", SelectionList).clear_options()
        # Reset model picker to the registered default.
        self.query_one("#model_set", Select).value = DEFAULT_MODEL
        self._dataset_meta = None

    @on(Button.Pressed, "#fetch")
    def _on_fetch_button(self) -> None:
        self._trigger_fetch()

    @on(Input.Submitted, "#repo")
    def _on_fetch_submit(self) -> None:
        self._trigger_fetch()

    def _trigger_fetch(self) -> None:
        repo = self.query_one("#repo", Input).value.strip()
        if not repo:
            self.query_one("#error", Static).update("Dataset id is required")
            return
        self.query_one("#error", Static).update("Fetching metadata…")
        self._fetch_worker(repo)

    @work(thread=True, exclusive=True, group="fetch")
    def _fetch_worker(self, repo: str) -> None:
        try:
            meta = _fetch_dataset_metadata(repo)
            self.call_from_thread(self._on_metadata_loaded, repo, meta)
        except Exception as e:
            self.call_from_thread(self.query_one("#error", Static).update, f"Fetch failed: {e}")

    def _on_metadata_loaded(self, repo: str, meta: DatasetMeta) -> None:
        self._dataset_meta = meta
        self.query_one("#error", Static).update("")
        # Populate config dropdown
        config_set = self.query_one("#config_set", Select)
        options = [(cfg if cfg else "(default)", cfg) for cfg in meta.configs]
        config_set.set_options(options)
        first_cfg = meta.configs[0] if meta.configs else ""
        config_set.value = first_cfg
        # Default name suggestion
        name_input = self.query_one("#name", Input)
        if not name_input.value:
            name_input.value = _slugify(repo)
        # Populate split + columns based on first config
        self._on_config_changed(first_cfg)

    @on(Select.Changed, "#config_set")
    def _config_changed(self, event: Select.Changed) -> None:
        meta = self._dataset_meta
        if meta is None or event.value is Select.BLANK:
            return
        self._on_config_changed(str(event.value))

    def _on_config_changed(self, cfg: str) -> None:
        meta = self._dataset_meta
        if meta is None:
            return
        splits = meta.splits_by_config.get(cfg, ["train"])
        split_set = self.query_one("#split_set", Select)
        split_set.set_options([(s, s) for s in splits])
        split_set.value = splits[0]

        cols = meta.columns_by_config.get(cfg, [])
        col_widget = self.query_one("#cols", SelectionList)
        col_widget.clear_options()
        for c in cols:
            preselect = c.lower() in {"text", "body", "content", "document"}
            col_widget.add_option(Selection(c, c, preselect))
        label = self.query_one("#cols_label", Static)
        if cols:
            label.update(
                f"Text columns — {len(cols)} available "
                "(check one or more; multiple are joined with newlines per row)"
            )
        else:
            label.update("Text columns — no columns reported by this dataset")
        self.query_one("#preview", Static).update("")

    @on(SelectionList.SelectedChanged, "#cols")
    def _cols_changed(self) -> None:
        # Trigger a small preview once at least one column is selected.
        meta = self._dataset_meta
        if meta is None:
            return
        chosen_cols = list(self.query_one("#cols", SelectionList).selected)
        if not chosen_cols:
            self.query_one("#preview", Static).update("")
            return
        # Defer the preview to avoid spamming on rapid toggles
        self._preview_worker(chosen_cols)

    @work(thread=True, exclusive=True, group="preview")
    def _preview_worker(self, cols: list[str]) -> None:
        repo = self.query_one("#repo", Input).value.strip()
        cfg = self._current_config()
        split = self._current_split()
        try:
            texts, _ = _load_dataset_rows(repo, cfg, split, cols, max_docs=3)
        except Exception as e:
            self.call_from_thread(
                self.query_one("#preview", Static).update,
                f"[i]preview failed: {e}[/i]",
            )
            return
        lines = []
        for i, t in enumerate(texts, 1):
            snip = t.replace("\n", " ⏎ ")[:200]
            if len(t) > 200:
                snip += "…"
            lines.append(f"[b]{i}.[/b] {snip}")
        self.call_from_thread(
            self.query_one("#preview", Static).update,
            "\n".join(lines) or "[i](empty)[/i]",
        )

    def _current_config(self) -> str | None:
        v = self.query_one("#config_set", Select).value
        if v is Select.BLANK or not v:
            return None
        return str(v)

    def _current_split(self) -> str:
        v = self.query_one("#split_set", Select).value
        return "train" if v is Select.BLANK else str(v)

    def _current_model(self) -> str:
        v = self.query_one("#model_set", Select).value
        return DEFAULT_MODEL if v is Select.BLANK else str(v)

    @on(Button.Pressed, "#index_btn")
    def _on_index_btn(self) -> None:
        if self._dataset_meta is None:
            self.query_one("#error", Static).update("Fetch metadata first")
            return
        repo = self.query_one("#repo", Input).value.strip()
        chosen_cols = list(self.query_one("#cols", SelectionList).selected)
        if not chosen_cols:
            self.query_one("#error", Static).update("Pick at least one text column")
            return
        name = (self.query_one("#name", Input).value or _slugify(repo)).strip()
        max_docs_str = self.query_one("#max_docs", Input).value.strip()
        max_docs = int(max_docs_str) if max_docs_str else None
        model_id = self._current_model()

        params = {
            "repo": repo,
            "config": self._current_config(),
            "split": self._current_split(),
            "cols": chosen_cols,
            "name": name,
            "max_docs": max_docs,
            "model": model_id,
        }
        self.query_one("#main", ContentSwitcher).current = "indexing"
        self.query_one("#status", Static).update(f"Indexing [b]{name}[/b]…")
        self.query_one("#detail", Static).update("Loading rows from HuggingFace…")
        self._index_worker(params)

    @work(thread=True, exclusive=True, group="index")
    def _index_worker(self, params: dict) -> None:
        detail = self.query_one("#detail", Static)
        try:
            t0 = time.time()
            texts, _raw = _load_dataset_rows(
                params["repo"],
                params["config"],
                params["split"],
                params["cols"],
                params["max_docs"],
            )
            n = len(texts)
            self.call_from_thread(
                detail.update,
                f"Encoding {n:,} docs with {params['model'].split('/')[-1]}…",
            )
            sparse_docs = encode_corpus(
                texts, model=params["model"], batch_size=32, show_progress=False
            )
            self.call_from_thread(detail.update, "Building inverted index…")
            retriever = SpladeRetriever(model=params["model"])
            retriever.index(sparse_docs)
            target = self.indexes_dir / params["name"]
            self.call_from_thread(detail.update, f"Saving to {target}…")
            retriever.save(target, corpus=texts)
            elapsed = time.time() - t0
            self.call_from_thread(self._on_index_done, params["name"], elapsed)
        except Exception as e:
            self.call_from_thread(self._on_index_failed, str(e))

    def _on_index_done(self, name: str, elapsed: float) -> None:
        self._refresh_sidebar()
        # find the new info and switch to search
        info = next((i for i in _list_indexes(self.indexes_dir) if i.name == name), None)
        if info is None:
            self.query_one("#main", ContentSwitcher).current = "welcome"
            return
        self.notify(f"Indexed {info.n_docs:,} docs in {elapsed:.1f}s", severity="information")
        self._load_and_switch_to_search(info)

    def _on_index_failed(self, msg: str) -> None:
        self.query_one("#main", ContentSwitcher).current = "new"
        self.query_one("#error", Static).update(f"Index failed: {msg}")

    # ---- search ----

    @on(Input.Submitted, "#query")
    def _on_query(self, event: Input.Submitted) -> None:
        if self._retriever is None or not event.value.strip():
            return
        k_input = self.query_one("#k", Input).value.strip()
        try:
            k = max(1, min(100, int(k_input))) if k_input else 10
        except ValueError:
            k = 10
        t0 = time.time()
        ids, scores = self._retriever.retrieve(event.value, k=k)
        dt_ms = (time.time() - t0) * 1000.0
        table = self.query_one("#results", DataTable)
        table.clear()
        corpus = self._retriever._corpus or []  # type: ignore[attr-defined]
        for rank, (did, score) in enumerate(zip(ids.tolist(), scores.tolist(), strict=True), 1):
            doc = corpus[int(did)] if int(did) < len(corpus) else {}
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            snip = text.replace("\n", " ⏎ ")[:200]
            if len(text) > 200:
                snip += "…"
            table.add_row(str(rank), f"{score:.3f}", snip, key=str(int(did)))
        self.query_one("#footnote", Static).update(f"{len(ids)} results · {dt_ms:.1f} ms")

    @on(DataTable.RowSelected, "#results")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if self._retriever is None:
            return
        try:
            did = int(event.row_key.value)
        except (TypeError, ValueError):
            return
        corpus = self._retriever._corpus or []  # type: ignore[attr-defined]
        if did >= len(corpus):
            return
        doc = corpus[did]
        if isinstance(doc, dict):
            body = doc.get("text", "")
            title = doc.get("title") or f"doc {did}"
        else:
            body = str(doc)
            title = f"doc {did}"
        self.push_screen(ResultDetail(title, body))


# ---------- entry point ----------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="splade-easy interactive TUI")
    p.add_argument("--dataset", help="Prefill the HF dataset id on the new-index form")
    p.add_argument(
        "--indexes-dir",
        help=(
            "Folder containing saved indexes "
            "(default: $SPLADE_EASY_INDEX_DIR or ~/.splade-easy/indexes)"
        ),
    )
    args = p.parse_args(argv)
    indexes_dir = Path(args.indexes_dir).expanduser() if args.indexes_dir else None
    SpladeTUI(indexes_dir=indexes_dir, prefill_dataset=args.dataset).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
