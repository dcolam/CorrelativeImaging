"""Dock layout shared by the napari tools."""
from __future__ import annotations


def add_control_view(viewer, widget, name: str = "Control view", width: int = 340, right_dock=None,
                     right_width: int = 600):
    """The widget as a dock on the left, tabbed with napari's layer list and layer controls, on top. Returns the
    dock. right_dock: the tool's right dock, sized together with it once Qt has laid the window out."""
    dock = viewer.window.add_dock_widget(widget, area="left", name=name)
    try:
        from qtpy.QtCore import QTimer, Qt
        mw, qv = viewer.window._qt_window, viewer.window._qt_viewer
        for d, title in ((qv.dockLayerList, "Layers"), (qv.dockLayerControls, "Layer controls")):
            d.setWindowTitle(title)
            if hasattr(d, "title") and hasattr(d.title, "title_label"):
                d.title.title_label.setText(title)
            mw.tabifyDockWidget(dock, d)

        def settle():                          # after Qt has laid the docks out: Control view on top, sane widths
            dock.raise_()
            docks, sizes = [dock], [width]
            if right_dock is not None:
                docks.append(right_dock)
                sizes.append(right_width)
            mw.resizeDocks(docks, sizes, Qt.Horizontal)
        QTimer.singleShot(0, settle)
        QTimer.singleShot(300, settle)
    except Exception:                          # napari internals moved: the dock still works, untabbed
        pass
    return dock
