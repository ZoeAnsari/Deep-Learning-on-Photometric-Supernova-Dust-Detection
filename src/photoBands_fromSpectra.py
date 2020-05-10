import pandas as pd
import matplotlib.pyplot as plt
import pyphot
import numpy as np
import time as time
from src.photometry_filters import filter_logFlux
import astropy.units as u
import warnings
warnings.filterwarnings("ignore")



def define_filters_flux(survey_name, wavelength,wavelength_um, flux,len_data):
    lib = pyphot.get_library()
    
    df = pd.read_csv("pyphot/table.csv")
    table = df[df["name"].str.contains(survey_name)]

    ###drop 150w2 from JWST
    a=["JWST_NIRCAM_F150W2"]
    table = table[~table['name'].isin(a)]
    ###sort the table
    table = table.sort_values(by='effective wavelength')



    filter_logflux_={}

    filter_names = list(table["name"])
#     if lib[filter_names[0]].wavelength_unit == 'AA' :
#         filter_logflux_, filters_clWL_=filter_logFlux(survey_name, filter_names, wavelength,wavelength, flux,len_data)
        
#     elif lib[filter_names[0]].wavelength_unit == 'um' :
    filter_logflux_, filters_clWL_=filter_logFlux(survey_name, filter_names, wavelength_um,wavelength, flux,len_data)
        
    filter_logflux__=pd.DataFrame.from_dict(filter_logflux_)
    filters_clWL__=pd.DataFrame.from_dict(filters_clWL_)

    return filter_logflux__, filters_clWL__





# #Units of flux
# #1 Jy = 10^{-23} erg s^{-1} cm^{-2} Hz^{-1} = 10^{-26} Watts m^{-2} Hz^{-1}







