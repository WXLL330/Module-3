import random

import numba

import minitorch

import time

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch, total_loss, correct, losses, avg_time, total_time):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct, "per-epoch avg time", avg_time, "total time", total_time)


def RParam(*shape, backend):
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, 2*hidden, backend)
        self.layer3 = Linear(2*hidden, 4*hidden, backend)
        self.layer4 = Linear(4*hidden, 8*hidden, backend)
        self.layer5 = Linear(8*hidden, 4*hidden, backend)
        self.layer6 = Linear(4*hidden, 2*hidden, backend)
        self.layer7 = Linear(2*hidden, hidden, backend)
        self.layer8 = Linear(hidden, 1, backend)

    def forward(self, x):
        # TODO: Implement for Task 3.5.
        # raise NotImplementedError("Need to implement for Task 3.5")
        y = self.layer1.forward(x).relu()
        y = self.layer2.forward(y).relu()
        y = self.layer3.forward(y).relu()
        y = self.layer4.forward(y).relu()
        y = self.layer5.forward(y).relu()
        y = self.layer6.forward(y).relu()
        y = self.layer7.forward(y).relu()
        y = self.layer8.forward(y).sigmoid()
        return y


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        s = minitorch.zeros((out_size,), backend=backend)
        s = s + 0.1
        self.bias = minitorch.Parameter(s)
        self.out_size = out_size

    def forward(self, x):
        # TODO: Implement for Task 3.5.
        # raise NotImplementedError("Need to implement for Task 3.5")
        y = x @ self.weights.value + self.bias.value
        return y


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        BATCH = 10
        losses = []

        total_start_time = time.time()
        for epoch in range(max_epochs):
            # epoch_start_time = time.time()
            total_loss = 0.0
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)

            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()
                X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)
                # Forward
                st = time.time()
                out = self.model.forward(X).view(y.shape[0])
                et = time.time()
                print(f"forward cost: {et-st}s\n")

                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -prob.log()

                st = time.time()
                (loss / y.shape[0]).sum().view(1).backward()
                et = time.time()
                print(f"backward cost: {et-st}s\n")

                total_loss = loss.sum().view(1)[0]

                # Update
                optim.step()

            # epoch_cost_time = time.time() - epoch_start_time
            losses.append(total_loss)
            avg_time_per_epoch = (time.time() - total_start_time) / (epoch + 1)

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])

                total_time = time.time() - total_start_time

                log_fn(epoch, total_loss, correct, losses, avg_time_per_epoch, total_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--PLOT", default=False, help="dataset")

    args = parser.parse_args()

    PTS = args.PTS

    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = minitorch.datasets["Simple"].simple(PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    FastTrain(
        HIDDEN, backend=FastTensorBackend if args.BACKEND != "gpu" else GPUBackend
    ).train(data, RATE, max_epochs=1)
