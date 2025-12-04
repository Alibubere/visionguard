import torch
import logging


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device,
    epoch: int,
    scheduler=None,
):

    model.to(device)
    model.train()

    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):

        images = [img.to(device) for img in images]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss_value = loss.item()

        running_loss += loss_value
        num_batches += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            logging.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss_value:.4f} "
                + " ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = running_loss / max(1, num_batches)
    return avg_loss


def evaluate_one_epoch(model: torch.nn.Module, dataloader, device, epoch):
    model.to(device)

    model_was_training = model.training
    model.train()  # stay in train mode so it returns losses

    running_loss = 0.0
    num_batches = 0

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(dataloader):

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)   # now it's a dict
            loss = sum(loss_dict.values())
            loss_value = loss.item()

            running_loss += loss_value
            num_batches += 1

            if batch_idx % 10 == 0:
                logging.info(
                    f"[VAL] Epoch: [{epoch}] Batch: [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss_value:.4f} "
                    + " ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                )

    # restore previous mode (not critical right now, but clean)
    if not model_was_training:
        model.eval()

    avg_loss = running_loss / max(1, num_batches)
    return avg_loss
