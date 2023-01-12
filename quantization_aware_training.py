import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Normalize, ToTensor, Compose

from model import NetBN, Net


def quantize_aware_training(model, device, train_loader, optimizer, epoch):
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.quantize_forward(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(
                "Quantize Aware Training Epoch: {} [{}/{}]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
                )
            )


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print("\nTest set: Full Model Accuracy: {:.3f}%\n".format(100.0 * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print("\nTest set: Quant Model Accuracy: {:.3f}%\n".format(100.0 * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 64
    seed = 1
    epochs = 5
    lr = 0.001
    momentum = 0.5
    using_bn = True
    load_quant_model_file = None

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        MNIST(
            "data",
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    test_loader = DataLoader(
        MNIST(
            "data",
            train=False,
            transform=transforms.Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    if using_bn:
        model = NetBN()
        model.load_state_dict(torch.load("ckpt/mnist_cnnbn.pt", map_location="cpu"))
        save_file = "ckpt/mnist_cnnbn_qat.pt"
    else:
        model = Net()
        model.load_state_dict(torch.load("ckpt/mnist_cnn.pt", map_location="cpu"))
        save_file = "ckpt/mnist_cnn_qat.pt"
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.eval()

    full_inference(model, test_loader)

    num_bits = 6
    model.quantize(num_bits=num_bits)
    print("Quantization bit: %d" % num_bits)

    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print("Successfully load quantized model %s" % load_quant_model_file)

    model.train()

    for epoch in range(1, epochs + 1):
        quantize_aware_training(model, device, train_loader, optimizer, epoch)

    model.eval()
    torch.save(model.state_dict(), save_file)

    model.freeze()

    quantize_inference(model, test_loader)
