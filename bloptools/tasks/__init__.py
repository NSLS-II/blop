class Task:
    MIN_NOISE_LEVEL = 1e-4
    MAX_NOISE_LEVEL = 1e-2

    def __init__(self, key, kind="max", name=None, transform=None, **kwargs):
        self.key = key
        self.kind = kind
        self.name = name if name is not None else f"{kind}_{key}"
        self.transform = transform if transform is not None else lambda x: x

        if kind.lower() in ["min", "minimum", "minimize"]:
            self.sign = -1
        elif kind.lower() in ["max", "maximum", "maximize"]:
            self.sign = +1
        else:
            raise ValueError('"kind" must be either "min" or "max"')

    def fitness_func(self, x):
        return self.sign * self.transform(x)

    def get_fitness(self, entry):
        return self.fitness_func(getattr(entry, self.key))
