from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from backend.service import ensure_registry
from core.registry import registry


class AppLauncher(tk.Tk):
    """Simple desktop launcher used during and after migration."""

    def __init__(self) -> None:
        ensure_registry(include_scenes=True)
        super().__init__()
        self.title("EWSEngine 控制台")
        self.geometry("460x430")
        self.configure(bg="#eef1f6")
        self._build()

    def _build(self) -> None:
        title = tk.Label(
            self,
            text="电磁波仿真模块",
            font=("Microsoft YaHei", 16, "bold"),
            bg="#eef1f6",
            fg="#243447",
        )
        title.pack(pady=18)

        modules = registry.all()
        if not modules:
            tk.Label(self, text="未检测到已注册模块", bg="#eef1f6").pack(pady=12)
            return

        for module in modules:
            button = ttk.Button(
                self,
                text=module.name,
                command=lambda key=module.key: self._launch(key),
                width=34,
            )
            button.pack(pady=6)

        ttk.Button(self, text="退出", command=self.destroy, width=34).pack(side=tk.BOTTOM, pady=20)

    def _launch(self, key: str) -> None:
        scene_class = registry.get_scene(key)
        scene = scene_class()
        scene.show()


def run_desktop_app() -> None:
    app = AppLauncher()
    app.mainloop()
