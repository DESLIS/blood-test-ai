from tkinter import ttk


def create_training_section(
    parent,
    epoch_var,
    loss_var,
    train_accuracy_var,
    test_accuracy_var,
    train_callback,
    reset_callback
):

    frame = ttk.LabelFrame(parent, text="Training", padding=10)
    frame.grid(
        row=2,
        column=0,
        columnspan=2,
        sticky="ew",
        pady=(0, 10)
    )

    for index, variable in enumerate(
        [epoch_var, loss_var, train_accuracy_var, test_accuracy_var]
    ):
        ttk.Label(frame, textvariable=variable).grid(
            row=0,
            column=index,
            padx=15
        )

    ttk.Button(
        frame,
        text="Train 1 Epoch",
        command=lambda: train_callback(1)
    ).grid(row=1, column=0, pady=10, padx=5)

    ttk.Button(
        frame,
        text="Train 1000 Epochs",
        command=lambda: train_callback(1000)
    ).grid(row=1, column=1, pady=10, padx=5)

    ttk.Button(
        frame,
        text="Reset",
        command=reset_callback
    ).grid(row=1, column=2, pady=10, padx=5)

