import tkinter as tk
from tkinter import ttk

from model import CONDITION_PROFILES
from .tooltip import ToolTip


def create_condition_info_section(parent):

    frame = ttk.LabelFrame(
        parent,
        text="Condition Patterns (Educational)",
        padding=10
    )
    frame.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="ew",
        pady=(0, 10)
    )

    for index, condition in enumerate(CONDITION_PROFILES):
        column = index * 2

        ttk.Label(frame, text=condition["name"]).grid(
            row=0,
            column=column,
            sticky="w",
            padx=(5, 2),
            pady=3
        )

        info_label = tk.Label(
            frame,
            text="ⓘ",
            fg="blue",
            cursor="question_arrow"
        )
        info_label.grid(
            row=0,
            column=column + 1,
            sticky="w",
            padx=(0, 15),
            pady=3
        )

        ToolTip(info_label, condition["details"])
