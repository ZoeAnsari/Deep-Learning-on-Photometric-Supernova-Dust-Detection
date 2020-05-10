import pandas as pd
import numpy as np
from keras.models import load_model
import time as time
from skimage import io, filters, feature

def preprocess_spectra(df):

    start_=time.time()

    ###make df to have 3000 to 24500 wl and zero values for corresponds fluxes

    df = df.replace('NaN', 0)

    t1=df

    
    
        ####making average over unit of wavelength
    wl = list(range(3000, 6500))
    wl.extend(list(range(6600, 13300)))
    wl.extend(list(range(14400, 17900)))
    wl.extend(list(range(19200, 24000)))
    # wl = np.array(wl)
    wl=list(wl)


    t1_new={}
    t1_new["WL"]=[]
    t1_new["Flux"]=[]

    for j in wl:
        try:
            indexNames = t1[ (t1['WL'] >= j) & (t1['WL'] < j+1) ].index
            

            t1_temp=t1.loc[indexNames,:]
            Flux_mean= (t1_temp["Flux"]).mean()
            t1_new["WL"].append(j)
            t1_new["Flux"].append(Flux_mean)

        except Exception as e:
            print("no profile", e)



    t1_reform=pd.DataFrame.from_dict(t1_new)
    t1_=t1_reform
    
    
    indexNames = t1_[ (t1_['WL'] <= 3000)  ].index
    t1_.drop(indexNames , inplace=True)
#     smooth_flux_t1_0 = apply_gaussian_filter(fluxes=t1_["Flux"], sigma=sigma)


    indexNames = t1_[ (t1_['WL'] >= 6500) & (t1_['WL'] <= 6600) ].index
    t1_.drop(indexNames , inplace=True)


    indexNames = t1_[ (t1_['WL'] >= 13300) & (t1_['WL'] <= 14400) ].index
    t1_.drop(indexNames , inplace=True)


    indexNames = t1_[ (t1_['WL'] >= 17900) & (t1_['WL'] <= 19200) ].index
    t1_.drop(indexNames , inplace=True)


    indexNames = t1_[ (t1_['WL'] > 24000)  ].index
    t1_.drop(indexNames , inplace=True)

    t1_cut=t1_
    
    sigma=2000
    
    
    def apply_gaussian_filter(fluxes, sigma):
        return filters.gaussian(image=fluxes, sigma=sigma, mode='reflect')
    
    

#     ####making average over unit of wavelength
#     wl = list(range(3000, 6500))
#     wl.extend(list(range(6600, 13300)))
#     wl.extend(list(range(14400, 17900)))
#     wl.extend(list(range(19200, 24000)))
#     # wl = np.array(wl)
#     wl=list(wl)


#     t1_new={}
#     t1_new["WL"]=[]
#     t1_new["Flux"]=[]

#     for j in wl:
#         try:
#             indexNames = t1_cut[ (t1_cut['WL'] >= j) & (t1_cut['WL'] < j+1) ].index
            

#             t1_temp=t1_cut.loc[indexNames,:]
#             Flux_mean= (t1_temp["Flux"]).mean()
#             t1_new["WL"].append(j)
#             t1_new["Flux"].append(Flux_mean)

#         except Exception as e:
#             print("no profile", e)



#     t1_reform=pd.DataFrame.from_dict(t1_new)
#     data_=t1_reform
    
    smooth_flux_t1 = apply_gaussian_filter(fluxes=t1_cut["Flux"], sigma=sigma)
#     print(len(t1_cut["Flux"]), "-------------------------", len(smooth_flux_t1))
    
    smoothed_flux_t1=pd.DataFrame(smooth_flux_t1)
    smoothed_flux_t1=smoothed_flux_t1.rename(columns={0 : 'Flux'})
    smoothed_flux_t1["WL"]=t1_cut["WL"]

    
    
    data=smoothed_flux_t1#t1_cut#t1_#
    data=data.dropna()

    
    return data

