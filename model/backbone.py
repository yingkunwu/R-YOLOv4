from model.utils import *


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbm = Conv(3, 32, 3, 1, "mish")
        self.csp1 = CSP(32, 64, 1) # downsample 2x
        self.csp2 = CSP(64, 128, 2) # downsample 2x
        self.csp3 = CSP(128, 256, 8) # downsample 2x
        self.csp4 = CSP(256, 512, 8) # downsample 2x
        self.csp5 = CSP(512, 1024, 4) # downsample 2x

    def forward(self, i):
        d1 = self.cbm(i)
        d1 = self.csp1(d1)
        d1 = self.csp2(d1)
        d3 = self.csp3(d1)
        d4 = self.csp4(d3)
        d5 = self.csp5(d4)
        return d3, d4, d5
