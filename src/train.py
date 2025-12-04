import torch
import logging
from src.work_data.dataloader import get_dataloader


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


def evaluate_one_epoch(model: torch.nn.Module, device, dataloader, epoch):
    model.to(device)
    model.eval()

    running_loss = 0.0
    num_batches = 0

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(dataloader):

            images = [img.to(device) for img in images]

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images,targets)
            loss = sum(loss_dict.values())
            loss_value = loss.item()

            running_loss += loss_value
            num_batches += 1

            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch: [{epoch}] Batch: [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss_value:.4f} "
                    )
                
    avg_loss =  running_loss / max(1,num_batches)

    return avg_loss