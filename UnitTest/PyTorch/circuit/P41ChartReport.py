
class ChartReport() :

    @classmethod
    def CR0_0404_001(self, functionInfo):
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        exeFunctionLDir = "UnitTest/PyTorch/file/result/V0_04_0/9999/CR0_0404_001"
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        x = np.linspace(-5,5,100)
        y = 1 / (1+np.exp(-x))
        plt.plot(x,y)
        plt.savefig("{}/{}".format(exeFunctionLDir,"CR0_0404_001.jpg"))
        plt.close()
        return {}, {}
