
import pandas as pd
import matplotlib.pyplot as plt
import pyphot
import numpy as np
import time as time
import astropy.units as u


def filter_plot(survey_name, wavelength, wavelength_um, flux, index_list):
    wavelength=wavelength_um
    wavelength_nonum=wavelength_um
    df = pd.read_csv("pyphot/table.csv")
    table = df[df["name"].str.contains(survey_name)]

    ###drop 150w2 from JWST
    a = ["JWST_NIRCAM_F150W2"]
    table = table[~table['name'].isin(a)]

    ###sort the table
    table = table.sort_values(by='effective wavelength')
    filter_names = list(table["name"])
    lib = pyphot.get_library()

    obj_list=index_list
    plt.figure(figsize=(8,6))
    for objects in obj_list:
        filters = lib.load_filters(filter_names, lamb=wavelength[objects])# * u.aa)#*wl_unit)

        mags=[]
        mags_flux=[]
        filter_flux=[]
        filters_clWL=[]
        for name, fn in zip(filter_names, filters):
            flux0=fn.get_flux(wavelength[objects], flux[objects])
            filters_clWL.append(fn.cl.magnitude *1e+4)
            f= flux0 #- ABf
            filter_flux.append(f)




##########FIRST plot---------------------------

        plt.scatter(filters_clWL , filter_flux, s=10, marker='o')#, label=survey_name)
        # plt.plot(filters_clWL, np.log(filter_flux))
        plt.xlabel(r"$\lambda (\AA)$")
        plt.xlim(3000,24500)
        # plt.xscale("log")
        plt.ylabel("Flux "+r"($erg s^{-1} cm^{-2} Hz^{-1}$)")
        plt.ylim(1e-21, 1e-13)
        plt.yscale("log")
        plt.title(str(survey_name))




    plt.savefig("Photometric_SamplePlot/Flux_filter_continuum"+str(survey_name)+".png")


    ############----------------------------log flux plots
    plt.figure(figsize=(12, 8.5))
    
    for objects in obj_list:
        filters = lib.load_filters(filter_names, lamb=wavelength[objects])# * u.aa)#*wl_unit)
#         print("filters", filters)



        mags=[]
        mags_flux=[]
        filter_flux=[]
        filters_clWL=[]
        for name, fn in zip(filter_names, filters):
            flux0=fn.get_flux(wavelength[objects], flux[objects])
            filters_clWL.append(fn.cl.magnitude * 1e+4)
            f= flux0 #- ABf
            filter_flux.append(f)
            
        plt.scatter(wavelength[objects] * 1e+4, np.log10(flux[objects])+19 , s=5, marker=',') 
        plt.scatter(filters_clWL, np.log10(filter_flux) + 19, s=20, label=survey_name)
        
        plt.xlim(3000,24500)
        plt.xscale('log')
        plt.xticks(np.arange(3000,24000, step=1000), rotation=30)#, size=12)
        plt.yticks(size=18)
        plt.xlabel(r"$\lambda (\AA)$",size=18)
        plt.ylabel("log Flux "+r"($erg s^{-1} cm^{-2} Hz^{-1}$) + 19", size=20)

    plt.title(str(survey_name), size=20)
    plt.savefig("Photometric_SamplePlot/logFlux_filter"+str(survey_name)+".png")