import tkinter as tk
from tkinter import ttk

import numpy as np


class MatrixSection:

    def __init__(self, parent):

        frame = ttk.LabelFrame(
            parent,
            text="Network Parameters",
            padding=10
        )
        frame.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="nsew"
        )

        for column in range(3):
            frame.columnconfigure(column, weight=1)
        frame.rowconfigure(1, weight=1)

        self.widgets = []
        for column, (name, shape) in enumerate(
            [("W1", "5 × 3"), ("W2", "3 × 3"), ("W3", "3 × 5")]
        ):
            ttk.Label(frame, text=f"{name} ({shape})").grid(
                row=0,
                column=column
            )
            widget = tk.Text(frame, height=8)
            widget.grid(
                row=1,
                column=column,
                sticky="nsew",
                padx=5
            )
            self.widgets.append(widget)

    def refresh(self, model):

        for widget, matrix in zip(
            self.widgets,
            [model.W1, model.W2, model.W3]
        ):
            widget.delete("1.0", tk.END)
            widget.insert(
                tk.END,
                np.array2string(
                    matrix,
                    precision=3,
                    suppress_small=True
                )
            )

