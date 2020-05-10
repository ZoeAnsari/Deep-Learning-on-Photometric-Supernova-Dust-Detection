import pandas as pd
import matplotlib.pyplot as plt
import pyphot
import numpy as np
import time as time
import astropy.units as u



def filter_logFlux(survey_name, filter_names, wavelength, wavelength_um, flux, len_data):
    wavelength_nonum=wavelength_um
    lib = pyphot.get_library()

    obj_list = list(range(0, len_data, 1))
    plt.figure()
    filter_logflux_={}
    filters_clWL_={}
#     print("len_obj_list", len(obj_list))

    for objects in obj_list:
        filters = lib.load_filters(filter_names, lamb=wavelength[objects])  # * u.aa)#*wl_unit)

        mags = []
        mags_flux = []
        filter_logflux_[objects] = []
        filters_clWL_[objects] = []
        filters_effWL = []
        filters_clWL = []
        for name, fn in zip(filter_names, filters):
            flux0 = fn.get_flux(wavelength[objects], flux[objects])
#             print("flux0",flux0)
            
            filters_effWL.append(fn.leff.magnitude )#,casting="unsafe")
            filters_clWL.append(fn.cl.magnitude *1e+4)
            f=flux0 # - ABf
#             print("f",f)
            filters_clWL_[objects].append(filters_clWL)
            filter_logflux_[objects].append(np.log10(f) +19)
#             print("filter_logflux_",filter_logflux_)
#     print("filters_clWL_",filters_clWL_)

    return filter_logflux_ , filters_clWL_
