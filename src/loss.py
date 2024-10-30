import torch.nn as nn

class ce_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_all_class, targets_all_class):
        """
        Calculates the cross-entropy loss for all labels and all samples.

        Args:
            logits_all_class (list): List of logits for each class.
                Each logits item contains logits for a single label.
            targets_all_class (list): List of target labels for each class.

        Returns:
            torch.Tensor: Average cross-entropy loss for all labels.
        """
        assert len(logits_all_class) == len(targets_all_class)

        num_samples = 0
        loss_all_class = []
        for i, logits in enumerate(logits_all_class):
            num_samples += logits.size(0)
            # loss = nn.functional.cross_entropy(logits, targets_all_class[i])
            loss = nn.functional.cross_entropy(logits_all_class[i], targets_all_class[i])
            loss_all_class.append(loss)
        return sum(loss_all_class) / num_samples

