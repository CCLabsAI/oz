import torch
import math

class Reservoir:
    def __init__(self, sample_size):
        self.n = 0
        self.data = torch.zeros(sample_size)

    def f(self, k, n):
        return k / n

    def add(self, x):
        n = self.n
        k = self.data.size(0)

        if n < k:
            self.data[n] = x
        else:
            u = self.f(float(k), float(n))
            if torch.rand(1).item() < u:
                i = torch.randint(0, k, (1,), dtype=torch.int).item()
                self.data[i] = x

        self.n += 1

    def sample(self):
        return self.data[:self.n]

# reference: http://www.aclweb.org/anthology/P14-2112
class ExponentialReservoir(Reservoir):
    def __init__(self, sample_size, beta_ratio):
        super().__init__(sample_size)
        k = sample_size[0]
        beta = beta_ratio * k
        p_k = k * (1.0 - math.exp(- (1.0/beta)))
        assert 0.0 < p_k and p_k <= 1.0

        self.beta_ratio = beta_ratio
        self.p_k = p_k

    def f(self, k, n):
        return self.p_k


if __name__ == "__main__":
    r = Reservoir([10, 3])
    for i in range(1000):
        r.add(torch.full([3], i))
    print(r.sample())

    er = ExponentialReservoir([8, 2, 2], 10.0)
    for i in range(1000):
        er.add(torch.full([2, 2], i))
    print(er.sample())
