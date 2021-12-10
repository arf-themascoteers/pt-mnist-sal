from torch.utils.data import DataLoader
import torch

def test(model, working_set):
    BATCH_SIZE = 100
    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model.eval()
    correct = 0
    total = 0
    print("Testing started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct += pred.eq(y_true.data.view_as(pred)).sum()
            total += 1

    print(f"Testing end. Accuracy: {round(correct.item()/len(working_set)*100,2)}%")
