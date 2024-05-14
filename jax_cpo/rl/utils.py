class Count:
    def __init__(self, n: int):
        self.count = 0
        self.n = n

    def __call__(self):
        bingo = (self.count + 1) == self.n
        self.count = (self.count + 1) % self.n
        return bingo
