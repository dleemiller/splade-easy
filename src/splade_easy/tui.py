"""Textual TUI for splade-easy. Interactive HF-dataset indexing + search.

Run with `uv run splade-tui` (after `uv sync --extra tui`).

All indexes go into a single folder (default `~/.splade-easy/indexes/`, override
with `SPLADE_EASY_INDEX_DIR`). The sidebar lists every index there and lets you
flip between them without leaving the search context.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import time
from collections.abc import Callable
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
        ProgressBar,
        Select,
        SelectionList,
        Static,
        Switch,
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


def _config_path() -> Path:
    """Persisted user settings, in the parent of the index dir."""
    return _default_index_dir().parent / "config.json"


# ---------- persisted user settings ----------


@dataclass
class Settings:
    """Persisted user settings. Re-read on app start, written on Save in the modal."""

    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps
    batch_size: int = 32
    max_seq_length: int | None = None  # None = model default
    default_k: int = 10
    save_corpus: bool = True

    @classmethod
    def load(cls, path: Path) -> Settings:
        if not path.exists():
            return cls()
        try:
            import json as _json

            data = _json.loads(path.read_text())
        except Exception:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path: Path) -> None:
        import json as _json
        from dataclasses import asdict

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(asdict(self), indent=2))

    def device_arg(self) -> str | None:
        """Translate to the value passed to `encode_corpus(device=...)`."""
        return None if self.device == "auto" else self.device


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
    """Return only the list of config names; splits + columns are loaded lazily
    per-config via `_fetch_config_details`. For huge datasets like FineWeb-Edu
    with dozens of CommonCrawl-snapshot configs, fetching builder info for every
    config upfront would enumerate millions of parquet files.
    """
    from datasets import get_dataset_config_names

    try:
        configs = get_dataset_config_names(repo)
    except Exception as e:
        raise RuntimeError(f"Could not enumerate configs for {repo}: {e}") from e

    if not configs:
        configs = [""]  # dataset with no named configs — represent as empty string

    return DatasetMeta(
        configs=[c or "" for c in configs],
        default_config=(configs[0] or "") if configs else "",
        # splits/columns populated lazily per-config
    )


def _fetch_config_details(repo: str, cfg: str) -> tuple[list[str], list[str]]:
    """Get the splits and columns for a single config. Uses streaming to peek at
    the first row for column names — avoids downloading any data files for the
    schema lookup."""
    from datasets import get_dataset_split_names, load_dataset

    cfg_arg = cfg or None
    splits = sorted(get_dataset_split_names(repo, cfg_arg) or [])
    if not splits:
        return [], []
    # Streaming + take 1: peeks the first parquet block, no full download.
    ds = (
        load_dataset(repo, cfg_arg, split=splits[0], streaming=True)
        if cfg_arg
        else load_dataset(repo, split=splits[0], streaming=True)
    )
    try:
        first_row = next(iter(ds))
        cols = list(first_row.keys())
    except StopIteration:
        cols = []
    return splits, cols


def _load_dataset_rows(
    repo: str,
    config: str | None,
    split: str,
    columns: list[str],
    max_docs: int | None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> list[str]:
    """Stream rows up to `max_docs` (or all if None) and join the chosen columns.

    Uses `streaming=True` so a `max_docs=10000` request against a billion-doc
    corpus only pulls the first parquet shard(s) it actually needs. Only the
    joined text is kept per row — the full per-row dict the caller never used
    isn't materialized, which was several GB of pure waste on a 1M-row index.
    """
    from datasets import load_dataset

    ds = (
        load_dataset(repo, config or None, split=split, streaming=True)
        if config
        else load_dataset(repo, split=split, streaming=True)
    )

    texts: list[str] = []
    for i, row in enumerate(ds):
        if max_docs is not None and max_docs > 0 and i >= max_docs:
            break
        text = "\n".join(str(row[c]) for c in columns if c in row and row[c] is not None)
        texts.append(text)
        if progress_callback is not None and (i + 1) % 200 == 0:
            progress_callback(i + 1, max_docs)
    if progress_callback is not None:
        progress_callback(len(texts), max_docs)
    return texts


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


class SettingsModal(ModalScreen["Settings | None"]):
    """Adjust persisted settings. Returns the new Settings on Save, None on Cancel."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    SettingsModal { align: center middle; }
    SettingsModal > Container {
        width: 70; height: auto; max-height: 90%;
        padding: 1 2;
        border: round $primary; background: $surface;
    }
    SettingsModal .label { padding-top: 1; color: $accent; }
    SettingsModal Input { margin-bottom: 0; }
    SettingsModal .row { height: 3; }
    SettingsModal .row Switch { margin-left: 1; }
    SettingsModal Button { margin: 1 1 0 0; }
    SettingsModal #buttons { height: auto; padding-top: 1; }
    """

    def __init__(self, current: Settings) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        s = self._current
        with Container():
            yield Static("[b]Settings[/b]")

            yield Label("Device — where the encoder runs", classes="label")
            yield Select[str](
                options=[
                    ("auto (detect)", "auto"),
                    ("cpu", "cpu"),
                    ("cuda", "cuda"),
                    ("cuda:0", "cuda:0"),
                    ("mps (Apple Silicon)", "mps"),
                ],
                value=s.device,
                allow_blank=False,
                id="s_device",
            )

            yield Label("Encoder batch size", classes="label")
            yield Input(value=str(s.batch_size), id="s_batch_size")

            yield Label(
                "Max sequence length (blank = 512; higher costs O(N²) attention memory)",
                classes="label",
            )
            yield Input(value="" if s.max_seq_length is None else str(s.max_seq_length), id="s_msl")

            yield Label("Default top-k in search", classes="label")
            yield Input(value=str(s.default_k), id="s_k")

            with Horizontal(classes="row"):
                yield Label("Save corpus alongside index", classes="label")
                yield Switch(value=s.save_corpus, id="s_save_corpus")

            with Horizontal(id="buttons"):
                yield Button("Save", id="save_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn")

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#cancel_btn")
    def _on_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save_btn")
    def _on_save(self) -> None:
        try:
            bs = max(1, int(self.query_one("#s_batch_size", Input).value or "32"))
            msl_raw = self.query_one("#s_msl", Input).value.strip()
            msl = int(msl_raw) if msl_raw else None
            k = max(1, int(self.query_one("#s_k", Input).value or "10"))
        except ValueError as e:
            self.app.notify(f"Invalid number: {e}", severity="error")
            return
        device = self.query_one("#s_device", Select).value
        if device is Select.BLANK:
            device = "auto"
        new = Settings(
            device=str(device),
            batch_size=bs,
            max_seq_length=msl,
            default_k=k,
            save_corpus=self.query_one("#s_save_corpus", Switch).value,
        )
        self.dismiss(new)


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
    IndexingPanel #detail { color: $text-muted; padding-bottom: 1; }
    IndexingPanel #progress { width: 100%; margin-bottom: 1; display: none; }
    IndexingPanel #progress.-active { display: block; }
    IndexingPanel #spinner.-hidden { display: none; }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        yield Static("", id="detail")
        # Both are mounted; we toggle visibility via the -active / -hidden classes
        # based on whether we have a known total to track.
        yield ProgressBar(total=100, show_eta=True, id="progress")
        yield LoadingIndicator(id="spinner")


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
            # Initial value overwritten by SpladeTUI._load_and_switch_to_search()
            # from settings.default_k.
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
    Sidebar #settings_btn { width: 100%; margin-top: 0; }
    Sidebar .empty { color: $text-muted; padding: 1 0; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Indexes (0)", id="title")
        yield ListView(id="index_list")
        yield Button("+ new index", id="new_btn", variant="primary")
        yield Button("Settings", id="settings_btn")


# ---------- main app ----------


class SpladeTUI(App):
    CSS = """
    Screen { layout: horizontal; }
    Sidebar { dock: left; }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_index", "New", show=True),
        Binding("ctrl+r", "refresh_sidebar", "Refresh", show=True),
        # F2 because Ctrl+, isn't reliably forwarded by most terminals.
        Binding("f2", "open_settings", "Settings", show=True),
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
        self._indexing_name: str | None = (
            None  # name of the dataset currently being indexed, or None
        )
        self._config_path = _config_path()
        self.settings: Settings = Settings.load(self._config_path)
        self._index_status_msg: str = ""
        self._index_start_time: float = 0.0
        self._index_ticker = None  # Textual Timer | None

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

    def action_open_settings(self) -> None:
        def _on_done(result: Settings | None) -> None:
            if result is not None:
                self.settings = result
                try:
                    self.settings.save(self._config_path)
                    self.notify("Settings saved", severity="information", timeout=3)
                except Exception as e:
                    self.notify(f"Failed to save settings: {e}", severity="error")

        self.push_screen(SettingsModal(self.settings), _on_done)

    @on(Button.Pressed, "#settings_btn")
    def _on_settings_btn(self) -> None:
        self.action_open_settings()

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
        # Pre-fill k from settings, then focus the query input.
        self.query_one("#k", Input).value = str(self.settings.default_k)
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
        self._set_fetch_busy(True, "Fetching configs… (large datasets may take a minute)")
        self._fetch_worker(repo)

    def _set_fetch_busy(self, busy: bool, msg: str = "") -> None:
        btn = self.query_one("#fetch", Button)
        repo = self.query_one("#repo", Input)
        btn.disabled = busy
        repo.disabled = busy
        btn.label = "Fetching…" if busy else "Fetch metadata"
        self.query_one("#error", Static).update(msg)

    @work(thread=True, exclusive=True, group="fetch")
    def _fetch_worker(self, repo: str) -> None:
        try:
            meta = _fetch_dataset_metadata(repo)
            self.call_from_thread(self._on_metadata_loaded, repo, meta)
        except Exception as e:
            self.call_from_thread(self._set_fetch_busy, False, f"Fetch failed: {e}")

    def _on_metadata_loaded(self, repo: str, meta: DatasetMeta) -> None:
        self._dataset_meta = meta
        self._set_fetch_busy(False, "")
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
        # Splits + columns are loaded lazily for the just-selected config
        self._load_config_details(first_cfg)

    @on(Select.Changed, "#config_set")
    def _config_changed(self, event: Select.Changed) -> None:
        meta = self._dataset_meta
        if meta is None or event.value is Select.BLANK:
            return
        cfg = str(event.value)
        # Use cached details if we already loaded them; otherwise fetch.
        if cfg in meta.splits_by_config:
            self._render_config_details(cfg)
        else:
            self._load_config_details(cfg)

    def _load_config_details(self, cfg: str) -> None:
        meta = self._dataset_meta
        if meta is None:
            return
        if cfg in meta.splits_by_config:
            self._render_config_details(cfg)
            return
        # Show pending state in the column label + clear splits dropdown.
        self.query_one("#split_set", Select).set_options([])
        self.query_one("#cols", SelectionList).clear_options()
        self.query_one("#cols_label", Static).update(
            f"Text columns — loading schema for {cfg or '(default)'!r}…"
        )
        repo = self.query_one("#repo", Input).value.strip()
        self._config_details_worker(repo, cfg)

    @work(thread=True, exclusive=True, group="cfg")
    def _config_details_worker(self, repo: str, cfg: str) -> None:
        try:
            splits, cols = _fetch_config_details(repo, cfg)
            self.call_from_thread(self._on_config_details_loaded, cfg, splits, cols)
        except Exception as e:
            self.call_from_thread(
                self.query_one("#cols_label", Static).update,
                f"Text columns — failed to load schema: {e}",
            )

    def _on_config_details_loaded(self, cfg: str, splits: list[str], cols: list[str]) -> None:
        meta = self._dataset_meta
        if meta is None:
            return
        meta.splits_by_config[cfg] = splits
        meta.columns_by_config[cfg] = cols
        # Only render if this is still the selected config
        current = self.query_one("#config_set", Select).value
        if current is not Select.BLANK and str(current) == cfg:
            self._render_config_details(cfg)

    def _render_config_details(self, cfg: str) -> None:
        meta = self._dataset_meta
        if meta is None:
            return
        splits = meta.splits_by_config.get(cfg, [])
        split_set = self.query_one("#split_set", Select)
        if splits:
            split_set.set_options([(s, s) for s in splits])
            split_set.value = splits[0]
        else:
            split_set.set_options([])

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
            label.update("Text columns — none reported by this config")
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
        if self._indexing_name is not None:
            self.query_one("#error", Static).update(
                f"Already indexing '{self._indexing_name}'. Wait for it to finish, "
                "or quit and restart to drop it."
            )
            return
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
        self._indexing_name = name
        self.query_one("#index_btn", Button).disabled = True
        self.query_one("#main", ContentSwitcher).current = "indexing"
        self.query_one("#status", Static).update(f"Indexing [b]{name}[/b]…")
        self.query_one("#detail", Static).update("Loading rows from HuggingFace…")
        self._index_worker(params)

    @work(thread=True, exclusive=True, group="index")
    def _index_worker(self, params: dict) -> None:
        progress = self.query_one("#progress", ProgressBar)
        try:
            t0 = time.time()
            self.call_from_thread(self._start_index_ticker)
            if params["max_docs"]:
                self.call_from_thread(self._show_progress, params["max_docs"])
            else:
                self.call_from_thread(self._show_spinner)
            self.call_from_thread(
                self._set_index_status,
                "Streaming rows from HuggingFace (first parquet shard may take a minute)…",
            )

            def _on_load(done: int, total: int | None) -> None:
                if total:
                    self.call_from_thread(progress.update, progress=done)
                    self.call_from_thread(
                        self._set_index_status, f"Streamed {done:,} / {total:,} rows"
                    )
                else:
                    self.call_from_thread(self._set_index_status, f"Streamed {done:,} rows so far…")

            texts = _load_dataset_rows(
                params["repo"],
                params["config"],
                params["split"],
                params["cols"],
                params["max_docs"],
                progress_callback=_on_load,
            )
            n = len(texts)
            self.call_from_thread(
                self._set_index_status,
                f"Encoding {n:,} docs with {params['model'].split('/')[-1]}…",
            )
            self.call_from_thread(self._show_progress, n)

            def _on_progress(done: int, total: int) -> None:
                self.call_from_thread(progress.update, progress=done)
                rate = done / max(1, time.time() - t0)
                self.call_from_thread(
                    self._set_index_status,
                    f"Encoding {done:,} / {total:,} docs ({rate:.0f} docs/s)",
                )

            sparse_docs = encode_corpus(
                texts,
                model=params["model"],
                batch_size=self.settings.batch_size,
                device=self.settings.device_arg(),
                max_seq_length=self.settings.max_seq_length,
                show_progress=False,
                progress_callback=_on_progress,
            )
            self.call_from_thread(self._show_spinner)
            self.call_from_thread(self._set_index_status, "Building inverted index…")
            retriever = SpladeRetriever(model=params["model"])
            retriever.index(sparse_docs)
            target = self.indexes_dir / params["name"]
            self.call_from_thread(self._set_index_status, f"Saving to {target}…")
            corpus_arg = texts if self.settings.save_corpus else None
            retriever.save(target, corpus=corpus_arg)
            elapsed = time.time() - t0
            self.call_from_thread(self._on_index_done, params["name"], elapsed)
        except Exception as e:
            self.call_from_thread(self._on_index_failed, str(e))
        finally:
            self.call_from_thread(self._stop_index_ticker)

    # ---- detail-line ticker ----
    # During shard downloads, the streaming iterator can block for tens of
    # seconds with zero callbacks fired -- the screen looks frozen. The ticker
    # repaints the detail line every second with the latest status + elapsed
    # time so the user always has visible movement.

    def _set_index_status(self, msg: str) -> None:
        self._index_status_msg = msg
        self._repaint_index_detail()

    def _start_index_ticker(self) -> None:
        self._index_start_time = time.time()
        self._index_status_msg = ""
        if self._index_ticker is None:
            self._index_ticker = self.set_interval(1.0, self._repaint_index_detail)

    def _stop_index_ticker(self) -> None:
        if self._index_ticker is not None:
            self._index_ticker.stop()
            self._index_ticker = None

    def _repaint_index_detail(self) -> None:
        elapsed = time.time() - self._index_start_time
        elapsed_str = (
            f"{elapsed:.0f}s" if elapsed < 60 else f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"
        )
        msg = self._index_status_msg or "Working…"
        with contextlib.suppress(Exception):
            self.query_one("#detail", Static).update(f"{msg}  ·  elapsed {elapsed_str}")

    def _show_progress(self, total: int) -> None:
        progress = self.query_one("#progress", ProgressBar)
        spinner = self.query_one("#spinner", LoadingIndicator)
        progress.update(total=total, progress=0)
        progress.set_class(True, "-active")
        spinner.set_class(True, "-hidden")

    def _show_spinner(self) -> None:
        progress = self.query_one("#progress", ProgressBar)
        spinner = self.query_one("#spinner", LoadingIndicator)
        progress.set_class(False, "-active")
        spinner.set_class(False, "-hidden")

    def _on_index_done(self, name: str, elapsed: float) -> None:
        self._indexing_name = None
        with contextlib.suppress(Exception):
            self.query_one("#index_btn", Button).disabled = False
        self._refresh_sidebar()
        info = next((i for i in _list_indexes(self.indexes_dir) if i.name == name), None)
        if info is None:
            self.notify(f"Indexed '{name}' but couldn't find it on disk", severity="warning")
            return
        msg = f"Indexed '{info.name}' ({info.n_docs:,} docs in {elapsed:.1f}s)"
        # Only auto-switch if the user is still watching the indexing pane.
        # If they navigated away (e.g. to search a different index), just notify
        # so we don't yank them off whatever they're doing.
        switcher = self.query_one("#main", ContentSwitcher)
        if switcher.current == "indexing":
            self.notify(msg, severity="information")
            self._load_and_switch_to_search(info)
        else:
            self.notify(
                f"{msg} — pick it from the sidebar to search it",
                severity="information",
                timeout=6,
            )

    def _on_index_failed(self, msg: str) -> None:
        self._indexing_name = None
        with contextlib.suppress(Exception):
            self.query_one("#index_btn", Button).disabled = False
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
