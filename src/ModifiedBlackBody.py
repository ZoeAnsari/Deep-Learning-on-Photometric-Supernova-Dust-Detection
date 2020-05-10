import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pprint import pprint
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time as time
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import optimize

import warnings

warnings.filterwarnings("ignore")


def TwoCompo_BB(wl, TEMPSN, RADIUSSN, TEMPDUST, MDUST , justsn):
    # 2components BlackBody formula
    h = 6.626076e-27  # plancks constant (erg s)#6.626e-34
    #     c = 2.99792e+5 #km/s#2.99792458E+10 #cm/s#3.0e+8
    k = 1.38066e-16  # boltzmann constant (erg/K)#1.38e-23
    Z = 0.001058  # redshift
    BETAL = 1.5  # slope for kappa

    MSUN = 1.98892e+33  # g
    #     CVEL= 299792.458 #km/s
    CC = 2.99792458E+10  # cm/s

    wlCM = wl * 1e-8  # wavelength from Angstrom to cm
    B1 = 2 * h * (CC ** 2)  ## #erg cm^2 s^-1
    B2 = h * CC / k  # K cm

    BX = (B1 / wlCM ** 5)  # erg s^-1 cm^-3
    #     BX=BX*1e-8 #convert from [erg s^-1 cm^-3] to [erg s^-1 cm^2 A^-1]

    D = 45.7  # Mpc
    LDS = D * 3.086e+18 * 1e+6  # luminosity distance from Mpc to cm

    KAPPASIN = 1e+4 * (wlCM / 1e-4) ** (
        -BETAL)  # [cm^2 g^-1]normalised to wavelength 1000. nm in cm #1 nm = 1.E-7 cm

    YMOD1 = BX / ((np.exp(B2 / (wlCM * TEMPSN))) - 1)  # erg s^-1 cm^-3
    YMOD11A = np.pi * YMOD1 * (((RADIUSSN) ** 2) / (LDS ** 2))  # erg s^-1 cm^-3

    YMOD11 = YMOD11A * 1e-8  # convert now from [erg s^-1 cm^-3] to [erg s^-1 cm^2 A-1]

    YMOD2 = (BX / ((np.exp(B2 / (wlCM * TEMPDUST))) - 1))
    YMOD22A = YMOD2 * KAPPASIN * ((MDUST * MSUN) / (LDS ** 2)) *justsn

    YMOD22 = YMOD22A * 1e-8  # convert now from [erg s^-1 cm^-3] to [erg s^-1 cm^2 A-1]
#     print("YMOD22", YMOD22)
#     print("YMOD11", YMOD11)

    YMOD = (YMOD11 + YMOD22)
    max_flux=np.max(YMOD)

    return YMOD , max_flux