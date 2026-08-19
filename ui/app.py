import tkinter as tk
from tkinter import ttk

import numpy as np

from model import (
    FEATURES,
    LABELS,
    NeuralNetwork,
    generate_dataset,
    normalize_input
)
from .condition_section import create_condition_info_section
from .input_section import create_input_section
from .matrix_section import MatrixSection
from .training_section import create_training_section


class BloodAnalysisUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Blood Analysis - Neural Network")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 700)

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.mean,
            self.std
        ) = generate_dataset()

        self.model = NeuralNetwork()
        self.input_vars = [tk.StringVar() for _ in FEATURES]
        self.result_var = tk.StringVar(value="No analysis performed")
        self.epoch_var = tk.StringVar(value="Epoch: 0")
        self.loss_var = tk.StringVar(value="Loss: -")
        self.train_accuracy_var = tk.StringVar(
            value=f"Training: 0 / {len(self.y_train)}"
        )
        self.test_accuracy_var = tk.StringVar(
            value=f"Test: 0 / {len(self.y_test)}"
        )

        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        create_input_section(
            main_frame,
            self.input_vars,
            self.result_var,
            self.analyze
        )
        create_condition_info_section(main_frame)
        create_training_section(
            main_frame,
            self.epoch_var,
            self.loss_var,
            self.train_accuracy_var,
            self.test_accuracy_var,
            self.train,
            self.reset_model
        )

        self.matrix_section = MatrixSection(main_frame)
        self.matrix_section.refresh(self.model)

    def train(self, epochs):

        for _ in range(epochs):
            self.model.train_one_epoch(self.X_train, self.y_train)

        self.update_training_display()
        self.matrix_section.refresh(self.model)

    def update_training_display(self):

        train_correct, train_total = self.model.accuracy(
            self.X_train,
            self.y_train
        )
        test_correct, test_total = self.model.accuracy(
            self.X_test,
            self.y_test
        )

        self.epoch_var.set(f"Epoch: {self.model.epoch}")
        self.loss_var.set(f"Loss: {self.model.loss:.4f}")
        self.train_accuracy_var.set(f"Training: {train_correct} / {train_total}")
        self.test_accuracy_var.set(f"Test: {test_correct} / {test_total}")

    def reset_model(self):

        self.model.reset()
        self.result_var.set("Model reset")
        self.update_training_display()
        self.matrix_section.refresh(self.model)

    def analyze(self):

        try:
            values = [float(variable.get()) for variable in self.input_vars]
        except ValueError:
            self.result_var.set("Please enter valid numbers.")
            return

        normalized = normalize_input(values, self.mean, self.std)

        if values[4] > 5:
            result = (
                "Result: Infection\n\n"
                "Demo rule applied: CRP is above 5 mg/L.\n\n"
            )
            probabilities = np.zeros(len(LABELS))
            probabilities[LABELS.index("Infection")] = 1.0
        else:
            probabilities = self.model.predict(normalized)
            best_index = int(np.argmax(probabilities))
            result = f"Result: {LABELS[best_index]}\n\n"

        for index, label in enumerate(LABELS):
            result += f"{label:<15} {probabilities[index] * 100:6.2f}%\n"

        self.result_var.set(result)


def launch_ui():

    root = tk.Tk()
    BloodAnalysisUI(root)
    root.mainloop()
