import rasterio
import numpy as np
from PIL import Image

class WI():
    def __init__(self,s2):
        self.s2=s2

    def VegetationIndex(self):
        m = self.s2[7] + self.s2[3]
        return np.divide((self.s2[7]-self.s2[3]),m,where=m!=0)

    def MoistureIndex(self):
        m = self.s2[8] + self.s2[11]
        return np.divide((self.s2[8]-self.s2[11]),m,where=m!=0)

    def NDWI(self):
        m = self.s2[2] + self.s2[7]
        return np.divide((self.s2[2]-self.s2[7]),m,where=m!=0)

    def MNDWI(self):
        m = self.s2[2] + self.s2[11]
        return np.divide((self.s2[2]-self.s2[11]),m,where=m!=0)

    def AWEI_nsh(self):
        m = (0.25 * self.s2[7] + 2.75 * self.s2[12])
        return np.divide(4*(self.s2[2]-self.s2[11]),m,where=m!=0)

    def AWEI_sh(self):
        return self.s2[1] + 2.5*self.s2[2] - 1.5*(self.s2[7] + self.s2[11]) - 0.25*self.s2[12]

    def XXWI(self):
        return 1.7204+171*self.s2[2]+3*self.s2[3]-70*self.s2[7]-45*self.s2[11]-71*self.s2[12]

    def Get_All(self):
        r1=self.VegetationIndex()
        r2=self.MoistureIndex()
        r3=self.NDWI()
        r4=self.MNDWI()
        r5=self.AWEI_nsh()
        r6=self.AWEI_sh()
        r7=self.XXWI()

        return np.array([r1,r2,r3,r4,r5,r6,r7])