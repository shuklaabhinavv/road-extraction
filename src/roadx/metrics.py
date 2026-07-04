import torch

EPS = 1e-7


class SegMetrics:
    """Accumulates confusion counts over batches, reports IoU/F1/precision/recall."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = (torch.sigmoid(logits) > self.threshold).float()
        t = target.float()
        self.tp += (pred * t).sum().item()
        self.fp += (pred * (1 - t)).sum().item()
        self.fn += ((1 - pred) * t).sum().item()
        self.tn += ((1 - pred) * (1 - t)).sum().item()

    def compute(self) -> dict[str, float]:
        precision = self.tp / (self.tp + self.fp + EPS)
        recall = self.tp / (self.tp + self.fn + EPS)
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + EPS),
            "f1": 2 * precision * recall / (precision + recall + EPS),
            "precision": precision,
            "recall": recall,
        }
