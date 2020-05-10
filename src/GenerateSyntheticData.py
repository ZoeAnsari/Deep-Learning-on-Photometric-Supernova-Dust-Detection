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

from src.ModifiedBlackBody import TwoCompo_BB
# from src.Scaling_target import max_val_scaling
# from src.RandomForest import rf

from skimage import io, filters, feature
import os
from time import sleep
import sys

import warnings

warnings.filterwarnings("ignore")





    




# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6,7"



def gen(tempSNlist_with_dust, radiusSNlist_with_dust, tempDustlist_with_dust, massDustlist_with_dust,
    tempSNlist_without_dust ,radiusSNlist_without_dust ,tempDustlist_without_dust , massDustlist_without_dust):
    
    
    start_gen=time.time()
    # cutoff on wavelength
    # Tullenic bands and Halpha
    wl = list(range(3000, 6500))
    wl.extend(list(range(6600, 13300)))
    wl.extend(list(range(14400, 17900)))
    wl.extend(list(range(19200, 24000)))
    wl = np.array(wl)

    len_generated_flux = len(wl)
#     print("length of w:", len(wl))
    # Generating early time SNIIps

    def apply_gaussian_filter(fluxes, sigma):
        return filters.gaussian(image=fluxes, sigma=sigma, mode='reflect')

    sigma=4000

    ###data with dust
    poisson_nois=np.random.poisson(5,len_generated_flux)

    poisson_noise = poisson_nois / np.max(poisson_nois)#np.abs(scaler.fit_transform(poisson_nois))


    tempSNlist= tempSNlist_with_dust
    radiusSNlist=radiusSNlist_with_dust
    tempDustlist= tempDustlist_with_dust
    massDustlist= massDustlist_with_dust

    #
    params_list=[tempSNlist, radiusSNlist, tempDustlist, massDustlist]


    objects = {}
    objects["tempSN"] = []
    objects["radiusSN"] = []
    objects["tempDust"] = []
    objects["massDust"] = []
    objects["log_massDust"]=[]
    objects["nonDust"] = []
    objects["Dust"]=[]
    generated_fluxes = []
    # log_generated_fluxes=[]
    # log_generated_fluxes_const=[]
    # log_generated_fluxes_withnoise=[]
    # log_generated_fluxes_withnoise_const=[]
    # log_smooth_flux_const=[]
    wavelength = []
    justsn=1
    
    i_bar=0
    for i in range(len(tempSNlist)):
        
    
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[{:{}}] {:.1f}% First part of generating data".format("="*i, len(tempSNlist)-1, (100/(len(tempSNlist)-1)*i)))
        sys.stdout.flush()
        sleep(0.5)
        i_bar=i_bar+1

        

        
        
        SNt=params_list[0]
        SNr=params_list[1]
        Dt=params_list[2]
        Dm=params_list[3]


        tempSN = SNt[i]
#         print(tempSN)
        radiusSN = SNr[i]
        radiusSN = (radiusSN * 1e+14) / 1e+3
        tempDust = Dt[i] / 1e+1
        massDust = Dm[i]
        massDust = (massDust * 1e-4) /1e+3
        twobb, max_generated_flux=TwoCompo_BB(wl=wl, TEMPSN=tempSN,
                                              RADIUSSN=radiusSN, TEMPDUST=tempDust, MDUST=massDust, justsn=justsn)

        generated_fluxes.append(twobb)# + (poisson_noise * twobb * 0.1))#max_generated_flux
        # log_generated_fluxes.append(np.log10(twobb))
        # log_generated_fluxes_const.append(np.log10(twobb) + 19)
        # log_generated_fluxes_withnoise.append(np.log10(twobb + (poisson_noise * twobb * 0.1)))#max_generated_flux
        # log_generated_fluxes_withnoise_const.append(np.log10(twobb + (poisson_noise * twobb * 0.1)) + 19)
        # log_smooth_flux_const.append(np.log10(apply_gaussian_filter(twobb + (poisson_noise * twobb * 0.1), sigma=sigma)) + 19)

        # noisy_flux= twobb + (poisson_noise* (0.1)* max_generated_flux)
        # smooth_flux = apply_gaussian_filter(noisy_flux, sigma=sigma)
        # generated_fluxes.append(smooth_flux)

        objects["tempSN"].append(tempSN)
        objects["radiusSN"].append(radiusSN/1e+14)
        objects["tempDust"].append(tempDust)
        objects["massDust"].append(massDust/1e-4)
        objects["log_massDust"].append(np.log10(massDust / 1e-4))
        objects["nonDust"].append(0)
        objects["Dust"].append(1)
        wavelength.append(wl)





#     print("length of objects before adding 0-dust:", len(objects["Dust"]))

    ###data without dust

    tempSNlist = tempSNlist_without_dust
    radiusSNlist = radiusSNlist_without_dust
    tempDustlist = tempDustlist_without_dust
    massDustlist = massDustlist_without_dust
    justsn=0


    params_list1 = [tempSNlist, radiusSNlist, tempDustlist, massDustlist]

    i_bar=0
    for i in range(len(tempSNlist)):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[{:{}}] {:.1f}% Second part of generating data".format("="*i, len(tempSNlist)-1, (100/(len(tempSNlist)-1)*i)))
        sys.stdout.flush()
        sleep(0.5)
        i_bar=i_bar+1
        
        
        
        
        SNt = params_list1[0]
        SNr = params_list1[1]
        Dt = params_list1[2]
        Dm = params_list1[3]

        tempSN = SNt[i]
#         print(tempSN)
        radiusSN = SNr[i]
        radiusSN = radiusSN * 1e+14 / 1e+3
        tempDust = Dt[i] / 1e+1
        massDust = Dm[i]
        massDust = massDust * 1e-4 / 1e+3
        twobb, max_generated_flux = TwoCompo_BB(wl=wl, TEMPSN=tempSN,
                                                RADIUSSN=radiusSN, TEMPDUST=tempDust, MDUST=massDust, justsn=justsn)

        generated_fluxes.append(twobb)#+ (poisson_noise * twobb * 0.1))#max_generated_flux
        # log_generated_fluxes.append(np.log10(twobb))
        # log_generated_fluxes_const.append(np.log10(twobb) + 19)
        # log_generated_fluxes_withnoise.append(np.log10(twobb+ (poisson_noise * twobb * 0.1)))#max_generated_flux
        # log_generated_fluxes_withnoise_const.append(np.log10(twobb + (poisson_noise * twobb * 0.1)) + 19)

        # log_smooth_flux_const.append( np.log10(apply_gaussian_filter(twobb + (poisson_noise * twobb * 0.1), sigma=sigma))+19 )

        # noisy_flux= twobb + (poisson_noise* (0.1)* max_generated_flux)
        # smooth_flux_const = apply_gaussian_filter(twobb+ (poisson_noise * twobb * 0.1), sigma=sigma)
        # generated_fluxes.append(smooth_flux)

        objects["tempSN"].append(tempSN)
        objects["radiusSN"].append(radiusSN/ 1e+14)
        objects["tempDust"].append(0)
        objects["massDust"].append(0)
        objects["log_massDust"].append(1)
        objects["nonDust"].append(1)
        objects["Dust"].append(0)
        wavelength.append(wl)

    end_gen = time.time()

    tt = end_gen - start_gen

    objects["generated_fluxes"]=generated_fluxes
    # objects["log_generated_fluxes"]=log_generated_fluxes
    # objects["log_generated_fluxes_const"] = log_generated_fluxes_const
    # objects["log_generated_fluxes_withnoise"]=log_generated_fluxes_withnoise
    # objects["log_generated_fluxes_withnoise_const"]=log_generated_fluxes_withnoise_const
    # objects["log_smooth_flux_const"]=log_smooth_flux_const
    objects["wavelength"]=wavelength
    # objects["label"]=

    objects_df = pd.DataFrame(objects)

    # df = pd.DataFrame(df_["scaled_generated_fluxes_"])

    # objects_df = objects_df.log_generated_fluxes_const.apply(pd.Series) \
    #     .merge(objects_df, right_index=True, left_index=True)
    # print("objects_df before save", objects_df.head(), objects_df.columns)
    # objects_df1=objects_df[:6000]
    # objects_df2 = objects_df[6000:]


    objects_df.to_pickle("SyntheticData/generated_data_random.pkl")


#     print("length after dust-0:", len(objects_df[objects_df["Dust"]== 1]))
    end_gen = time.time()
#     print("", end_gen - start_gen)







    return "Data Generation is complete Done"
