from matplotlib import pyplot as plt
from numpy import mean
from torch import tensor, Tensor, device, linspace, meshgrid, empty
from torch.nn import Module, Linear, ReLU, Parameter, BCEWithLogitsLoss, Tanh
from torch.optim import Adam
from tqdm import tqdm

from module import QLinear, QReLU


def plot_xor_model_output(model, x, title, a=-1, b=2, n=100):
    x1 = linspace(a, b, n)
    x2 = linspace(a, b, n)

    X1, X2 = meshgrid(x1, x2)
    Y = empty(n, n)

    for i in range(n):
        for j in range(n):
            Y[i, j] = model(Tensor([X1[i, j], X2[j, j]])).item()

    cm = plt.cm.get_cmap('viridis')
    plt.scatter(X1, X2, c=Y, cmap=cm)

    cm = plt.cm.get_cmap('viridis')
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm, edgecolors="white")
    plt.title(title)
    plt.show()


def train(model, epochs, train_dataloader, loss_function, optimizer, device=device("cpu")):
    losses = []
    for _ in tqdm(range(epochs)):
        local_losses = []
        for (X_batch, y_batch) in train_dataloader:
            optimizer.zero_grad()
            prediction = model(X_batch.to(device))
            loss = loss_function(prediction, y_batch.to(device))
            loss.backward()
            optimizer.step()
            local_losses.append(loss.item())
        losses.append(mean(local_losses))
    return losses


class XORNetwork(Module):
    def __init__(self):
        super(XORNetwork, self).__init__()
        self.linear_1 = Linear(2, 8, bias=False)
        self.relu_1 = ReLU()
        self.linear_2 = Linear(8, 1, bias=False)
        self.tanh = Tanh()

        # define the initialization for the weights
        self.linear_1.weight = Parameter(tensor([[-0.4069, -0.6801],
                                                 [0.5473, -0.6729],
                                                 [0.7219, -0.7386],
                                                 [0.2784, -0.8068],
                                                 [0.0049, -0.1208],
                                                 [0.4450, -0.9411],
                                                 [-1.4549, 1.0588],
                                                 [-0.1911, 0.1303]]))

        self.linear_2.weight = Parameter(tensor([[-0.2008, 0.6332, 0.4405, 0.5720, 0.0171, 0.3960, 0.9282, 0.1323]]))

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.tanh(x)
        return x

    def quantize(self, num_bits=8):
        self.qlinear_1 = QLinear(self.linear_1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu_1 = QReLU()
        self.qlinear_2 = QLinear(self.linear_2, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qlinear_1(x)
        x = self.qrelu_1(x)
        x = self.qlinear_2(x)
        x = self.tanh(x)
        return x

    def freeze(self):
        self.qlinear_1.freeze()
        self.qrelu_1.freeze(self.qlinear_1.qo)
        self.qlinear_2.freeze(qi=self.qlinear_1.qo)

    def quantize_inference(self, x):
        qx = self.qlinear_1.qi.quantize_tensor(x)
        qx = self.qlinear_1.quantize_inference(qx)
        qx = self.qrelu_1.quantize_inference(qx)
        qx = self.qlinear_2.quantize_inference(qx)
        qx = self.qlinear_2.qo.dequantize_tensor(qx)
        x = self.tanh(qx)
        return x


if __name__ == '__main__':
    x = Tensor([[0., 0.],
                [0., 1.],
                [1., 0.],
                [1., 1.]])

    y = Tensor([[0.],
                [1.],
                [1.],
                [0.]])

    model = XORNetwork()

    loss_function = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    model.train()
    losses = train(model, 500, [*zip(x, y)], loss_function, optimizer)

    plot_xor_model_output(model, x, "After Training FP32")
    plt.plot(losses)
    plt.title("FP32 Training Loss")
    plt.show()

    model.quantize(num_bits=8)
    losses = train(model.quantize_forward, 500, [*zip(x, y)], loss_function, optimizer)
    plot_xor_model_output(model.quantize_forward, x, "Simulated Quantized Model")
    plt.plot(losses)
    plt.title("FP32 Training Loss")
    plt.show()

    model.freeze()
    plot_xor_model_output(model.quantize_inference, x, "Quantized Freeze Model")
