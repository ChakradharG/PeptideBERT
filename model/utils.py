import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(inputs, attention_mask).squeeze(1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        # scheduler.step()  # if sch is onecycle

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    ground_truth = []
    predictions = []

    for batch in tqdm(dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.inference_mode():
            logits = model(inputs, attention_mask).squeeze(1)
            loss = criterion(logits, labels)

        total_loss += loss.item()
    
        preds = torch.where(logits > 0.5, 1, 0)
        predictions.extend(preds.cpu().tolist())
        ground_truth.extend(labels.cpu().tolist())

    total_loss = total_loss / len(dataloader)
    accuracy = 100 * accuracy_score(ground_truth, predictions)

    return total_loss, accuracy


def test(model, dataloader, device):
    model.eval()

    ground_truth = []
    predictions = []

    for batch in tqdm(dataloader):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        with torch.inference_mode():
            logits = model(inputs, attention_mask).squeeze(1)
    
        preds = torch.where(logits > 0.5, 1, 0)
        predictions.extend(preds.cpu().tolist())
        ground_truth.extend(labels.tolist())

    accuracy = 100 * accuracy_score(ground_truth, predictions)

    return accuracy
