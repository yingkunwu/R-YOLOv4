import math
import torch
import numpy as np

from torch.optim.lr_scheduler import LambdaLR


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 300
    t0 =  int((epochs * 200) * 0.05)

    lf = one_cycle(1, 0.1, int(epochs))
    lr_sched = LambdaLR(optimizer, lr_lambda=lf)

    initial_lr = optimizer.param_groups[0]['initial_lr']

    accumulate = 4

    lrs = []
    for i in range(epochs):
        for j in range(200):
            global_step = i * 200 + j + 1
                        # Warmup
            if global_step <= t0:
                xi = [0, t0]  # x interp
                accumulate = max(1, np.interp(global_step, xi, [1, 4]).round())
                optimizer.param_groups[0]['lr'] = np.interp(global_step, xi, [0.0, initial_lr * lf(i)])

            if global_step % accumulate == 0:
                optimizer.step()

        lrs.append(optimizer.param_groups[0]["lr"])
        lr_sched.step()
    plt.plot(lrs)
    plt.savefig('outputs/schduler.png')
    plt.show()
