from model.utils import *


class Backbonev4(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbm0 = Conv(3, 32, 3, 1, "mish")

        self.cbm1 = Conv(32, 64, 3, 2, "mish") # downsample 2x
        self.csp1 = CSP(64, 64, 1)

        self.cbm2 = Conv(64, 128, 3, 2, "mish") # downsample 2x
        self.csp2 = CSP(128, 128, 2)

        self.cbm3 = Conv(128, 256, 3, 2, "mish") # downsample 2x
        self.csp3 = CSP(256, 256, 8)

        self.cbm4 = Conv(256, 512, 3, 2, "mish") # downsample 2x
        self.csp4 = CSP(512, 512, 8)

        self.cbm5 = Conv(512, 1024, 3, 2, "mish") # downsample 2x
        self.csp5 = CSP(1024, 1024, 4)

        self.spp = SPP(1024, 512)

    def forward(self, x):
        x = self.cbm0(x)
        x = self.csp1(self.cbm1(x))
        x = self.csp2(self.cbm2(x))
        d3 = self.csp3(self.cbm3(x))
        d4 = self.csp4(self.cbm4(d3))
        d5 = self.csp5(self.cbm5(d4))

        d5 = self.spp(d5)

        return d3, d4, d5


class Backbonev5(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbs0 = Conv(3, 64, 6, 2, "swish") # downsample 2x
        self.cbs1 = Conv(64, 128, 3, 2, "swish") # downsample 2x
        self.csp1 = C3(128, 128, 3)

        self.cbs2 = Conv(128, 256, 3, 2, "swish") # downsample 2x
        self.csp2 = C3(256, 256, 6)

        self.cbs3 = Conv(256, 512, 3, 2, "swish") # downsample 2x
        self.csp3 = C3(512, 512, 9)

        self.cbs4 = Conv(512, 1024, 3, 2, "swish") # downsample 2x
        self.csp4 = C3(1024, 1024, 3)

        self.spp = SPPF(1024, 1024)

    def forward(self, x):
        x = self.cbs0(x)
        x = self.csp1(self.cbs1(x))
        d3 = self.csp2(self.cbs2(x))
        d4 = self.csp3(self.cbs3(d3))
        d5 = self.csp4(self.cbs4(d4))
        
        d5 = self.spp(d5)

        return d3, d4, d5


class Backbonev7(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbs0 = Conv(3, 32, 3, 1, "swish")

        self.cbs1 = Conv(32, 64, 3, 2, "swish") # downsample 2x
        self.cbs2 = Conv(64, 64, 3, 1, "swish")

        self.cbs3 = Conv(64, 128, 3, 2, "swish") # downsample 2x

        self.elan1 = ELAN1(128, 256)

        self.mc1 = MaxConv(256) # downsample 2x
        self.elan2 = ELAN1(256, 512)

        self.mc2 = MaxConv(512) # downsample 2x
        self.elan3 = ELAN1(512, 1024)

        self.mc3 = MaxConv(1024) # downsample 2x
        self.elan4 = ELAN1(1024, 1024, e1=0.25, e2=0.25)

        self.spp = SPPCSPC(1024, 512)

    def forward(self, x):
        x = self.cbs2(self.cbs1(self.cbs0(x)))
        x = self.elan1(self.cbs3(x))
        d3 = self.elan2(self.mc1(x))
        d4 = self.elan3(self.mc2(d3))
        d5 = self.elan4(self.mc3(d4))
        
        d5 = self.spp(d5)

        return d3, d4, d5
