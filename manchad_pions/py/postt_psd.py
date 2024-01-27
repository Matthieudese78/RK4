#!/bin/python3
#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import os
import matplotlib.pyplot as plt
import mplcursors

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [20, -50]
# %% quel type de modele ?
lraidtimo = False
lplam = False
lplow = False
# lconefixe = True
# %% Scripts :
    # which slice ?
slice = 1

# cas la lache de la manchette avec juste le poids :
# namerep = "manchadela_weight"
# namerep = "manchadela_RSG"
# namerep = "manchadela_RSG_conefixe"

linert = False

lpion = False
lpcirc = True
Fext = 193.
vlimoden = 1.e-4
mu = 0.6
xi = 0.01
spinini = 0.
dte = 1.e-6
vlostr = int(-np.log10(vlimoden))
dtstr = int(-np.log10(dte))
xistr = int(100.*xi)
namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}'
repload = f'./pickle/{namerep}/'
rep_save = f"./fig/{namerep}/"
if (linert):
    repload = f'./pickle/{namerep}_inert/'
    rep_save = f"./fig/{namerep}_inert/"

# namerep = f"manchadela_pions_{slice}"
# repload = f"./pickle/{namerep}/"

# %%

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")
    # on trie et on reindexe :
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
# %% fenetrage en temps : 
t1 = 0.
if (not linert):
    t1 = 0.1 
t2 = 120.
df = df[(df['t']>t1) & (df['t']<=t2)]
df.reset_index(drop=True,inplace=True)
# %% 100 points par seconde 
    # nsort = 10 et x4 dans dopickle_slices :
if (not linert):
    nsort = 40
if (linert):
    nsort = 30
    # on veut un point ttes les :
discr = 1.e-3
ndiscr = int(discr/(dte*nsort))
df = df.iloc[::ndiscr]
# rows2keep = df.index % ndiscr == 0 
# df = df[rows2keep]
df.reset_index(drop=True,inplace=True)
dt = df['t'].iloc[1] - df['t'].iloc[0] 
fs = 1/dt
#%%
nt = int(np.floor(np.log(len(df['t']))/np.log(2.)))
indexpsd = df[df.index < 2**nt].index
#%% contact time interval :
# alternative :
icrit = 1.e-12
# ccone :
indi_ccone = df[np.abs(df["FN_CCONE"])>icrit].index
indni_ccone = df.drop(indi_ccone).index
if (lpion):
    # pion haut :
    indi_ph = df[(np.abs(df["FN_ph1"])>icrit) | 
                 (np.abs(df["FN_ph2"])>icrit) |
                 (np.abs(df["FN_ph3"])>icrit)].index
    # indni_ph = df[(np.abs(df["FN_ph1"])<=icrit) | 
    #              (np.abs(df["FN_ph2"])<=icrit) |
    #              (np.abs(df["FN_ph3"])<=icrit)].index
    indni_ph = df.drop(indi_ph).index

    indi_pb = df[(np.abs(df["FN_pb1"])>icrit) | 
                 (np.abs(df["FN_pb2"])>icrit) |
                 (np.abs(df["FN_pb3"])>icrit)].index
    indni_pb = df.drop(indi_pb).index

if (lpcirc):
    # pion haut :
    indi_ph = df[(np.abs(df["FN_pcirch1"])>icrit) | 
                 (np.abs(df["FN_pcirch2"])>icrit) |
                 (np.abs(df["FN_pcirch3"])>icrit)].index
    indni_ph = df.drop(indi_ph).index

    # pion bas :
    indi_pb = df[(np.abs(df["FN_pcircb1"])>icrit) | 
                 (np.abs(df["FN_pcircb2"])>icrit) |
                 (np.abs(df["FN_pcircb3"])>icrit)].index
    indni_pb = df.drop(indi_pb).index

    indi_pb1 = df[(np.abs(df["FN_pcircb1"])>icrit)].index
    indi_pb2 = df[(np.abs(df["FN_pcircb2"])>icrit)].index
    indi_pb3 = df[(np.abs(df["FN_pcircb3"])>icrit)].index

    indi_ph1 = df[(np.abs(df["FN_pcirch1"])>icrit)].index
    indi_ph2 = df[(np.abs(df["FN_pcirch2"])>icrit)].index
    indi_ph3 = df[(np.abs(df["FN_pcirch3"])>icrit)].index

#%% transition :
transition = False
if transition:
    tslice = 0.5
    ntrans = 3
    ltrans = []
    # itrans = range(1,ntrans)
    itrans = [55,57,60,62,74,86]
    for i in itrans:
        t1 = i*tslice - 1.e-2 
        t2 = i*tslice + 1.e-2 
        ltrans.append(df[(df['t']>t1) & (df['t']<t2)].index)

#%%############################################
#           PLOTS : PSD :
###############################################
repsect1 = f"{rep_save}variables_ft/PSD/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
power, freq = plt.psd(-1.e6*df['uzg_tot_ad'].iloc[indexpsd], NFFT=2**(nt-5), Fs=100, scale_by_freq=0., color=color1[2])
plt.close('all')
# get the ordinate of plt.psd :
power_density = 10. * np.log10(power)

linteractif = False
if (linteractif):
    fig, ax = plt.subplots()
    ax.plot(freq, power_density, label='Data')
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Interactive Plot with mplcursors')
    
    # Show the plot
    plt.show()

#%%
if (linert):
    lanot = [(0.88,80.93),(2.70,33.30),(4.49,23.37),(8.11,8.61),(13.33,8.25)]
if (not linert):
    lanot = [(1.07,77.61),(3.12,33.18),(5.27,19.85),(9.52,10.67),(15.58,9.41)]
    # lanot = [(1.07,77.61),(3.12,33.18),(5.27,19.85),(7.32,9.48),(9.52,10.67),(15.58,9.41)]
kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad",
    "x": freq,
    "y": power_density,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[2],
    "annotations": lanot,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
# %%
