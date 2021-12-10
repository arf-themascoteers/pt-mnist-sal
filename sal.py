from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

def test(model, working_set):
    BATCH_SIZE = 1
    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=False)
    count = 0
    for data, y_true in dataloader:
        print(y_true[0].item())
        z = data
        z = z.squeeze().squeeze()
        z = z.detach().numpy()
        plt.imshow(z)
        plt.show()

        data.requires_grad = True

        y_pred = model(data)
        index = y_pred.argmax()
        final = y_pred[0,index]
        final.backward()

        x = data.grad
        x = torch.abs(x)
        x = x.squeeze().squeeze()
        x = x.detach().numpy()
        plt.imshow(x, cmap="hot")
        plt.show()


        count += 1
        if count == 10:
            break


