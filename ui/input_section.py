import tkinter as tk
from tkinter import ttk

from model import FEATURES
from .tooltip import ToolTip


def create_input_section(parent, input_vars, result_var, analyze_callback):

    frame = ttk.LabelFrame(
        parent,
        text="Blood Analysis",
        padding=10
    )
    frame.grid(
        row=0,
        column=0,
        columnspan=2,
        sticky="ew",
        pady=(0, 10)
    )

    for index, feature in enumerate(FEATURES):
        ttk.Label(frame, text=feature["name"]).grid(
            row=index,
            column=0,
            sticky="w",
            padx=5,
            pady=3
        )

        info_label = tk.Label(
            frame,
            text="ⓘ",
            fg="blue",
            cursor="question_arrow"
        )
        info_label.grid(row=index, column=1, padx=5)

        ToolTip(
            info_label,
            (
                f"{feature['description']}\n"
                f"Typical healthy value: {feature['healthy']}\n"
                f"Unit: {feature['unit']}"
            )
        )

        ttk.Entry(
            frame,
            textvariable=input_vars[index],
            width=15
        ).grid(row=index, column=2, padx=10, pady=3)

        ttk.Label(frame, text=feature["unit"]).grid(
            row=index,
            column=3,
            sticky="w"
        )

    ttk.Button(
        frame,
        text="Analyze",
        command=analyze_callback
    ).grid(
        row=len(FEATURES),
        column=0,
        columnspan=4,
        pady=10
    )

    ttk.Label(
        frame,
        textvariable=result_var,
        font=("Arial", 12)
    ).grid(
        row=len(FEATURES) + 1,
        column=0,
        columnspan=4,
        pady=5
    )
