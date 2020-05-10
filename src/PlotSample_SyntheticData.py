import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.ModifiedBlackBody import TwoCompo_BB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from time import sleep
# import sys


import warnings

warnings.filterwarnings("ignore")



###temp dust 1000,1500,2000 , vary dust masses
def plot_sample(df_objects,list_tot):
#     print(df_objects.head())
    plt.figure(figsize=(12,8.5))
#     list_tot=[1000, 2000, 3000, 4000,10000,11000]
    list2=[]
    list3=[]
    for i_list in list_tot:
        if i_list <10000:
            list2.append(i_list)#=[1000, 2000, 3000, 4000]
        else:
            list3.append(i_list)#=[10000, 11000]
    ax = plt.subplot(111)
    j=0
#     i_bar=0
    for i in list2:
#         sys.stdout.write('\r')
#         # the exact output you're looking for:
#         sys.stdout.write("[{:{}}] {:.1f}% First part plotting".format("="*i, len(list2)-1, (100/(len(list2)-1)*i)))
#         sys.stdout.flush()
#         sleep(0.5)
#         i_bar=i_bar+1
        
        cm = plt.get_cmap('gist_rainbow')
        ax.set_prop_cycle(color=[cm(1. * j / len(list_tot))])
        label_="R_SN%.1f1e+14cm" % df_objects["radiusSN"][i] + " T_SN%.1fK" %df_objects["tempSN"][i] +\
        " M_Dust%.1f1e-4Msun" %df_objects["massDust"][i] +\
        " T_Dust%.1fK" %df_objects["tempDust"][i]

        plt.scatter(df_objects["wavelength"][i],
                    np.log10(df_objects["generated_fluxes"][i])+19, label=label_, s=5)
        plt.xlim(3000,24500)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xticks(np.arange(3000,24000, step=1000), rotation=30)#, size=12)
        plt.yticks(size=18)
        plt.title(" Defined Modified Black Body curves  ", size=20)
        plt.xlabel(r"$\lambda (\AA)$",size=18)
        plt.ylabel(r"$log Flux + 19$",size=20)
        plt.legend(title=r"$M_{Dust} (10^{-4} M_{sun})$", fancybox=True)
        j = j + 1
#     print(j)

#     i_bar=0
    for i in list3:
#         sys.stdout.write('\r')
#         # the exact output you're looking for:
#         sys.stdout.write("[{:{}}] {:.1f}% Second part plotting".format("="*i, len(list3)-1, (100/(len(list3)-1)*i)))
#         sys.stdout.flush()
#         sleep(0.5)
#         i_bar=i_bar+1
        
        
        
        cm = plt.get_cmap('gist_rainbow')
        ax.set_prop_cycle(color=[cm(1. * j / len(list_tot))])

        label_="R_SN%.1f1e+14cm" % df_objects["radiusSN"][i] + " T_SN%.1fK" %df_objects["tempSN"][i] +\
            " No Dust "

        plt.scatter(df_objects["wavelength"][i],
                    np.log10(df_objects["generated_fluxes"][i])+19, label=label_, s=5)#,
                 # cmap=colors_.ListedColormap(colors))
        plt.xscale('log')
        plt.xlim(3000,24500)
        # plt.yscale('log')
        plt.xticks(np.arange(3000,24000, step=1000), rotation=30)#, size=12)
        plt.yticks(size=18)
        plt.title(" Defined BB curves  ", size=20)
        plt.xlabel(r"$\lambda (\AA)$",size=18)
        plt.ylabel(r"$log Flux + 19$",size=20)
        plt.legend(title=r"$M_{Dust} (10^{-4} M_{sun})$", fancybox=True)
        j = j + 1
    # plt.show()
    plt.savefig("SamplePlot/Flux-wl-Dust.png")
    plt.show()
    

# def plot_new_sample(tempSNlist,radiusSNlist, tempDustlist, massDustlist):
#     wl = list(range(3000, 6500))
#     wl.extend(list(range(6600, 13300)))
#     wl.extend(list(range(14400, 17900)))
#     wl.extend(list(range(19200, 24000)))
#     wl = np.array(wl)

#     len_generated_flux = len(wl)
#     poisson_nois = np.random.poisson(5, len_generated_flux)
#     poisson_noise = poisson_nois / np.max(poisson_nois)


#     params_list=[tempSNlist, radiusSNlist, tempDustlist, massDustlist]

#     justsn=1
#     dict_={}
#     dict_["generated_fluxes_"]=[]
#     dict_["wl_"]=[]
#     dict_["scaled_generated_fluxes_"]=[]

#     for i in range(len(tempSNlist)):
#         SNt = params_list[0]
#         SNr = params_list[1]
#         Dt = params_list[2]
#         Dm = params_list[3]
#         for j in range(len(tempSNlist)):
#             for k in range(len(tempSNlist)):
#                 for l in range(len(tempSNlist)):
#                     plt.figure(figsize=(16, 10))



#                     tempSN = SNt[i]
#                     radiusSN = SNr[j]
#                     radiusSN = (radiusSN * 1e+14) / 1e+3
#                     tempDust = Dt[k] / 1e+1
#                     massDust = Dm[l]
#                     massDust = (massDust * 1e-4) / 1e+3
#                     twobb, max_generated_flux = TwoCompo_BB(wl=wl, TEMPSN=tempSN,
#                                                         RADIUSSN=radiusSN, TEMPDUST=tempDust, MDUST=massDust, justsn=justsn)
#                     generated_fluxes=(twobb)# + (poisson_noise * max_generated_flux * 0.1))
#                     dict_["generated_fluxes_"].append(twobb)
#                     dict_["scaled_generated_fluxes_"].append(twobb / max_generated_flux )
#                     dict_["wl_"].append(wl)

#                     plt.plot(wl, generated_fluxes)
#                     plt.ylim(1e-19, 1e-13)
#                     # plt.xscale('log')
#                     plt.yscale('log')
#                     # plt.legend()
#                     plt.xticks(np.arange(3000,24000, step=1000), rotation=45)
#                     plt.title("T_SN: "+ str(tempSN)+ "R_SN(1e+14cm): "+ str(radiusSN/1e+14)+
#                                                          "T_Dust: "+ str(tempDust)+ "M_Dust(1e-4M_Sun): "+ str(massDust/1e-4))
#                     plt.savefig("SamplePlot/Flux-wl-Dust-Edges"+"T_SN"+ str(tempSN)+ "R_SN(1e+14cm)"+ str(radiusSN/1e+14)+
#                                                          "T_Dust"+ str(tempDust)+ "M_Dust(1e-4M_Sun)"+ str(massDust/1e-4)+".png")
#     plt.show()
                    






#     df_=pd.DataFrame.from_dict(dict_)

#     df=pd.DataFrame(df_["scaled_generated_fluxes_"])

#     data=df.scaled_generated_fluxes_.apply(pd.Series) \
#         .merge(df, right_index = True, left_index = True)

#     X_data_=data.drop(columns={ "scaled_generated_fluxes_"})

#     X_data=X_data_

#     for i in range(len(X_data)):
#         plt.figure(figsize=(16,10))
#         plt.plot(df_["wl_"][i], X_data.loc[i][:])
#         plt.ylim(0, 1)
#         # plt.xscale('log')
#         plt.yscale('log')
#         plt.savefig("SamplePlot/plot"+str(i)+".png")

