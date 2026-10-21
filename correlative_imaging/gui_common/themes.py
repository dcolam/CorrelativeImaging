"""Window themes: each tool has its own (ci gui: Navy, ci reconstruct: Petrol by default), chosen and customised in the
window's Theme menu and remembered per tool. The theme is set on that napari window only (viewer.theme): napari's
own preference and other napari windows stay as they are. The image canvas stays black (fluorescence is judged
against black).

A theme is a preset plus optional overrides of four colours: background, panels, accent, text.
"""
from __future__ import annotations

import json

PRESETS = {
    "Navy": dict(background="#0f1a2b", foreground="#1b2a41", primary="#2c4466", secondary="#3d5a80",
                 highlight="#5b8bd0", text="#e6edf5", icon="#c9d6e6", current="#3a7bd5"),
    "Petrol": dict(background="#0d2226", foreground="#163338", primary="#24505a", secondary="#336b76",
                   highlight="#4fb3bf", text="#e4f1f2", icon="#c4dde0", current="#2a8f9c"),
    "Graphite": None,          # napari's dark
    "Light": None,             # napari's light
}
DEFAULT = {"ci_gui": "Navy", "ci_reconstruct": "Petrol"}
EDITABLE = {"background": "background", "foreground": "panels", "current": "accent", "text": "text"}


def _settings():
    from qtpy.QtCore import QSettings
    return QSettings("CorrelativeImaging", "themes")


def load_choice(tool: str) -> tuple[str, dict]:
    """(preset, overrides) remembered for a tool."""
    s = _settings()
    preset = s.value(f"{tool}/preset", DEFAULT.get(tool, "Navy"))
    try:
        over = json.loads(s.value(f"{tool}/overrides", "{}") or "{}")
    except (TypeError, ValueError):
        over = {}
    return (preset if preset in PRESETS else DEFAULT.get(tool, "Navy")), over


def save_choice(tool: str, preset: str, overrides: dict):
    s = _settings()
    s.setValue(f"{tool}/preset", preset)
    s.setValue(f"{tool}/overrides", json.dumps(overrides))


def colors(preset: str, overrides: dict | None = None) -> dict:
    """The theme's colours as hex (napari's field names)."""
    from napari.utils.theme import get_theme
    base = get_theme("light" if preset == "Light" else "dark")
    out = {k: getattr(base, k).as_hex() for k in ("background", "foreground", "primary", "secondary", "highlight",
                                                   "text", "icon", "current")}
    out.update(PRESETS.get(preset) or {})
    out.update({k: v for k, v in (overrides or {}).items() if k in out})
    return out


_MANAGED = []                  # (weakref to a viewer, its theme id): windows whose theme this module set


def _register(tool: str, preset: str, overrides: dict | None) -> str:
    from napari.utils.theme import get_theme, register_theme
    base = get_theme("light" if preset == "Light" else "dark")
    tid = f"ci_{tool}"
    t = base.model_copy(update=dict(id=tid, label=f"{tool} ({preset})"))
    for k, v in colors(preset, overrides).items():
        setattr(t, k, v)
    t.canvas = "black"
    register_theme(tid, t, "correlative_imaging")
    return tid


def _restore_others(viewer):
    """napari resets every open window to its default theme when a theme is registered: give the other windows
    this module themed their own theme back."""
    alive = []
    for ref, tid in _MANAGED:
        w = ref()
        if w is None:
            continue
        alive.append((ref, tid))
        if w is not viewer and w.theme != tid:
            try:
                w.theme = tid
            except Exception:
                pass
    _MANAGED[:] = alive


def apply(viewer, tool: str, preset: str | None = None, overrides: dict | None = None) -> str:
    """Register the tool's theme and set it on this window only. Returns the theme id."""
    import weakref
    from napari.utils.theme import available_themes
    if preset is None:
        preset, overrides = load_choice(tool)
    # every tool's theme registered once up front, so opening the second tool's window resets nothing
    for other in DEFAULT:
        if other != tool and f"ci_{other}" not in available_themes():
            _register(other, *load_choice(other))
    tid = _register(tool, preset, overrides)
    if viewer.theme == tid:                                  # same id, new colours: make napari restyle
        viewer.theme = "light" if preset == "Light" else "dark"
    viewer.theme = tid
    if not any(r() is viewer for r, _ in _MANAGED):
        _MANAGED.append((weakref.ref(viewer), tid))
    _restore_others(viewer)
    return tid


def add_menu(viewer, tool: str, on_change=None):
    """A Theme menu in the window's menu bar: the presets, "Customise colours…" and "Reset"."""
    from qtpy.QtWidgets import QAction, QActionGroup
    mw = viewer.window._qt_window
    menu = mw.menuBar().addMenu("Theme")
    group = QActionGroup(menu)
    group.setExclusive(True)
    cur, _ = load_choice(tool)
    acts = {}
    for name in PRESETS:
        a = QAction(name, menu, checkable=True)
        a.setChecked(name == cur)
        group.addAction(a)
        menu.addAction(a)
        acts[name] = a

        def pick(_=False, name=name):
            _, over = load_choice(tool)
            save_choice(tool, name, {})                      # a new preset starts without overrides
            apply(viewer, tool, name, {})
            if on_change:
                on_change()
        a.triggered.connect(pick)
    menu.addSeparator()
    custom = QAction("Customise colours…", menu)
    custom.triggered.connect(lambda: _customise(viewer, tool, on_change))
    menu.addAction(custom)
    reset = QAction(f"Reset to {DEFAULT.get(tool, 'Navy')}", menu)

    def do_reset():
        save_choice(tool, DEFAULT.get(tool, "Navy"), {})
        acts[DEFAULT.get(tool, "Navy")].setChecked(True)
        apply(viewer, tool)
        if on_change:
            on_change()
    reset.triggered.connect(do_reset)
    menu.addAction(reset)
    return menu


def _customise(viewer, tool: str, on_change=None):
    """Colour buttons for background, panels, accent and text; changes show at once and are kept."""
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QColorDialog, QDialog, QDialogButtonBox, QFormLayout, QPushButton
    preset, over = load_choice(tool)
    start = dict(over)
    dlg = QDialog(viewer.window._qt_window)
    dlg.setWindowTitle(f"Theme colours ({preset})")
    f = QFormLayout(dlg)
    btns = {}

    def paint(b, hexc):
        b.setText(hexc)
        b.setStyleSheet(f"QPushButton {{ background: {hexc}; color: {'#111' if QColor(hexc).lightness() > 140 else '#eee'}; }}")

    def change(key):
        c = colors(preset, over)
        got = QColorDialog.getColor(QColor(c[key]), dlg, EDITABLE[key])
        if got.isValid():
            over[key] = got.name()
            paint(btns[key], over[key])
            apply(viewer, tool, preset, over)
            if on_change:
                on_change()
    c0 = colors(preset, over)
    for key, label in EDITABLE.items():
        b = QPushButton()
        paint(b, c0[key])
        b.clicked.connect(lambda _=False, key=key: change(key))
        btns[key] = b
        f.addRow(label, b)
    box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    box.accepted.connect(dlg.accept)
    box.rejected.connect(dlg.reject)
    f.addRow(box)
    if dlg.exec_():
        save_choice(tool, preset, over)
    else:                                                    # cancelled: back to what it was
        apply(viewer, tool, preset, start)
        if on_change:
            on_change()
