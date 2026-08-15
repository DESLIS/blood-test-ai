import tkinter as tk
from tkinter import ttk

import numpy as np

from model import NeuralNetwork
from dataset import (
    generate_dataset,
    normalize_input,
    LABELS,
    FEATURES
)


class ToolTip:

    def __init__(self, widget, text):

        self.widget = widget
        self.text = text
        self.tip = None

        widget.bind(
            "<Enter>",
            self.show
        )

        widget.bind(
            "<Leave>",
            self.hide
        )

    def show(self, event=None):

        if self.tip is not None:
            return

        x = (
            self.widget.winfo_rootx()
            + self.widget.winfo_width()
            + 5
        )

        y = (
            self.widget.winfo_rooty()
        )

        self.tip = tk.Toplevel(
            self.widget
        )

        self.tip.wm_overrideredirect(True)

        self.tip.geometry(
            f"+{x}+{y}"
        )

        label = tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5
        )

        label.pack()

    def hide(self, event=None):

        if self.tip:

            self.tip.destroy()
            self.tip = None


class BloodAnalysisUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Analyse sanguine - Réseau de neurones"
        )

        self.root.geometry(
            "1200x900"
        )

        self.root.minsize(
            1000,
            700
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.mean,
            self.std
        ) = generate_dataset()

        # ----------------------------------------------------
        # Modèle
        # ----------------------------------------------------

        self.model = NeuralNetwork()

        # ----------------------------------------------------
        # Variables Tkinter
        # ----------------------------------------------------

        self.input_vars = [
            tk.StringVar()
            for _ in FEATURES
        ]

        self.result_var = tk.StringVar(
            value="Aucune analyse effectuée"
        )

        self.epoch_var = tk.StringVar(
            value="Epoch : 0"
        )

        self.loss_var = tk.StringVar(
            value="Loss : -"
        )

        self.train_accuracy_var = tk.StringVar(
            value="Entraînement : 0 / 200"
        )

        self.test_accuracy_var = tk.StringVar(
            value="Test : 0 / 50"
        )

        # ----------------------------------------------------
        # Interface
        # ----------------------------------------------------

        main_frame = ttk.Frame(
            root,
            padding=15
        )

        main_frame.pack(
            fill="both",
            expand=True
        )

        main_frame.columnconfigure(
            0,
            weight=1
        )

        main_frame.columnconfigure(
            1,
            weight=1
        )

        main_frame.rowconfigure(
            2,
            weight=1
        )

        self.create_input_section(
            main_frame
        )

        self.create_training_section(
            main_frame
        )

        self.create_matrix_section(
            main_frame
        )

        self.refresh_matrices()

    # ========================================================
    # ANALYSE
    # ========================================================

    def create_input_section(self, parent):

        frame = ttk.LabelFrame(
            parent,
            text="Analyse sanguine",
            padding=10
        )

        frame.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 10)
        )

        for i, feature in enumerate(FEATURES):

            ttk.Label(
                frame,
                text=feature["name"]
            ).grid(
                row=i,
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

            info_label.grid(
                row=i,
                column=1,
                padx=5
            )

            ToolTip(
                info_label,
                (
                    f"{feature['description']}\n"
                    f"Valeur indicative saine : "
                    f"{feature['healthy']}\n"
                    f"Unité : {feature['unit']}"
                )
            )

            ttk.Entry(
                frame,
                textvariable=self.input_vars[i],
                width=15
            ).grid(
                row=i,
                column=2,
                padx=10,
                pady=3
            )

            ttk.Label(
                frame,
                text=feature["unit"]
            ).grid(
                row=i,
                column=3,
                sticky="w"
            )

        ttk.Button(
            frame,
            text="Analyser",
            command=self.analyser
        ).grid(
            row=len(FEATURES),
            column=0,
            columnspan=4,
            pady=10
        )

        ttk.Label(
            frame,
            textvariable=self.result_var,
            font=("Arial", 12)
        ).grid(
            row=len(FEATURES) + 1,
            column=0,
            columnspan=4,
            pady=5
        )

    # ========================================================
    # TRAINING
    # ========================================================

    def create_training_section(self, parent):

        frame = ttk.LabelFrame(
            parent,
            text="Training",
            padding=10
        )

        frame.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 10)
        )

        ttk.Label(
            frame,
            textvariable=self.epoch_var
        ).grid(
            row=0,
            column=0,
            padx=15
        )

        ttk.Label(
            frame,
            textvariable=self.loss_var
        ).grid(
            row=0,
            column=1,
            padx=15
        )

        ttk.Label(
            frame,
            textvariable=self.train_accuracy_var
        ).grid(
            row=0,
            column=2,
            padx=15
        )

        ttk.Label(
            frame,
            textvariable=self.test_accuracy_var
        ).grid(
            row=0,
            column=3,
            padx=15
        )

        ttk.Button(
            frame,
            text="Train 1 Epoch",
            command=lambda: self.train(1)
        ).grid(
            row=1,
            column=0,
            pady=10,
            padx=5
        )

        ttk.Button(
            frame,
            text="Train 1000 Epochs",
            command=lambda: self.train(1000)
        ).grid(
            row=1,
            column=1,
            pady=10,
            padx=5
        )

        ttk.Button(
            frame,
            text="Reset",
            command=self.reset_model
        ).grid(
            row=1,
            column=2,
            pady=10,
            padx=5
        )

    # ========================================================
    # MATRICES
    # ========================================================

    def create_matrix_section(self, parent):

        frame = ttk.LabelFrame(
            parent,
            text="Paramètres du réseau",
            padding=10
        )

        frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="nsew"
        )

        frame.columnconfigure(
            0,
            weight=1
        )

        frame.columnconfigure(
            1,
            weight=1
        )

        frame.columnconfigure(
            2,
            weight=1
        )

        frame.rowconfigure(
            1,
            weight=1
        )

        # ----------------------------------------------------
        # W1
        # ----------------------------------------------------

        ttk.Label(
            frame,
            text="W1 (5 × 3)"
        ).grid(
            row=0,
            column=0
        )

        self.w1_text = tk.Text(
            frame,
            height=8
        )

        self.w1_text.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=5
        )

        # ----------------------------------------------------
        # W2
        # ----------------------------------------------------

        ttk.Label(
            frame,
            text="W2 (3 × 3)"
        ).grid(
            row=0,
            column=1
        )

        self.w2_text = tk.Text(
            frame,
            height=8
        )

        self.w2_text.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=5
        )

        # ----------------------------------------------------
        # W3
        # ----------------------------------------------------

        ttk.Label(
            frame,
            text="W3 (3 × 5)"
        ).grid(
            row=0,
            column=2
        )

        self.w3_text = tk.Text(
            frame,
            height=8
        )

        self.w3_text.grid(
            row=1,
            column=2,
            sticky="nsew",
            padx=5
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self, epochs):

        for _ in range(epochs):

            self.model.train_one_epoch(
                self.X_train,
                self.y_train
            )

        self.update_training_display()

        self.refresh_matrices()

    # ========================================================
    # AFFICHAGE TRAINING
    # ========================================================

    def update_training_display(self):

        train_correct, train_total = (
            self.model.accuracy(
                self.X_train,
                self.y_train
            )
        )

        test_correct, test_total = (
            self.model.accuracy(
                self.X_test,
                self.y_test
            )
        )

        self.epoch_var.set(
            f"Epoch : {self.model.epoch}"
        )

        self.loss_var.set(
            f"Loss : {self.model.loss:.4f}"
        )

        self.train_accuracy_var.set(
            f"Entraînement : "
            f"{train_correct} / {train_total}"
        )

        self.test_accuracy_var.set(
            f"Test : "
            f"{test_correct} / {test_total}"
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset_model(self):

        self.model.reset()

        self.result_var.set(
            "Modèle réinitialisé"
        )

        self.update_training_display()

        self.refresh_matrices()

    # ========================================================
    # MATRICES
    # ========================================================

    def refresh_matrices(self):

        self.update_matrix(
            self.w1_text,
            self.model.W1
        )

        self.update_matrix(
            self.w2_text,
            self.model.W2
        )

        self.update_matrix(
            self.w3_text,
            self.model.W3
        )

    def update_matrix(self, widget, matrix):

        widget.delete(
            "1.0",
            tk.END
        )

        widget.insert(
            tk.END,
            np.array2string(
                matrix,
                precision=3,
                suppress_small=True
            )
        )

    # ========================================================
    # PRÉDICTION
    # ========================================================

    def analyser(self):

        try:

            values = [
                float(var.get())
                for var in self.input_vars
            ]

        except ValueError:

            self.result_var.set(
                "Veuillez saisir des nombres valides."
            )

            return

        # Normalisation exactement comme
        # celle utilisée pendant l'entraînement

        normalized = normalize_input(
            values,
            self.mean,
            self.std
        )

        probabilities = (
            self.model.predict(
                normalized
            )
        )

        best_index = int(
            np.argmax(probabilities)
        )

        result = (
            f"Résultat : {LABELS[best_index]}\n\n"
        )

        for i, label in enumerate(LABELS):

            result += (
                f"{label:<15} "
                f"{probabilities[i] * 100:6.2f}%\n"
            )

        self.result_var.set(
            result
        )


# ============================================================
# LANCEMENT
# ============================================================

def launch_ui():

    root = tk.Tk()

    BloodAnalysisUI(root)

    root.mainloop()


if __name__ == "__main__":

    launch_ui()