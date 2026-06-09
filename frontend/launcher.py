from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from backend.service import ensure_registry
from core.registry import registry
from frontend.app_v2 import ModernAppToplevel


class AppLauncher(tk.Tk):
    """Desktop launcher that routes migrated scenes to Tk shell windows."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.api_host = host
        self.api_port = int(port)
        ensure_registry(include_scenes=True)
        super().__init__()
        self.title("EWSEngine 控制台")
        self.geometry("480x460")
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
        title.pack(pady=(18, 8))

        modules = registry.all()
        if not modules:
            tk.Label(self, text="未检测到已注册模块", bg="#eef1f6").pack(pady=12)
            return

        for module in modules:
            button = ttk.Button(
                self,
                text=module.name,
                command=lambda scene_factory=registry.get_scene(module.key): self._launch(scene_factory),
                width=36,
            )
            button.pack(pady=6)

        ttk.Button(self, text="退出", command=self.destroy, width=36).pack(side=tk.BOTTOM, pady=20)

    def _launch(self, scene_class: type[object]) -> None:
        ModernAppToplevel(self, scene_factory=scene_class, host=self.api_host, port=self.api_port)


def run_desktop_app(host: str = "127.0.0.1", port: int = 8765) -> None:
    app = AppLauncher(host=host, port=port)
    app.mainloop()
