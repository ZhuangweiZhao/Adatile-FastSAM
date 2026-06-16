"""Early stopping — monitors a metric and stops training when no improvement."""


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of evaluations with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: "max" (higher is better, e.g. Dice) or "min" (e.g. loss).
        verbose: Print when stopping.

    Usage:
        stopper = EarlyStopping(patience=10, mode="max")
        for epoch in range(epochs):
            val_dice = evaluate()
            if stopper.step(val_dice):
                print(f"Early stop at epoch {epoch}")
                break
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001,
                 mode: str = "max", verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Check if training should stop. Returns True if should stop."""
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"[EarlyStop] No improvement for {self.patience} checks, "
                          f"best={self.best:.4f}. Stopping.")
        return self.should_stop
