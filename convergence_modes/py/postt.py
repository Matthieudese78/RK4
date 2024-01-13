#!/bin/python3
#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import matplotlib.pyplot as plt
import os

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
# %% Scripts :
# script1 = f"convergence_f_mu"
# %% 
rep_save = f"./fig/"

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")
#%%
manchette = True
stoia = False
if (manchette):
    script = f"manchette_nnoe_" 
    repload = f"./pickle/manchette/"
    rep_save = f"./fig/manchette/"

if (stoia):
    script = f"stoia_nnoe_" 
    repload = f"./pickle/stoia/"
    rep_save = f"./fig/stoia/"

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")
df['mode'].astype(int)
df['nnoe'].astype(int)
lnoe = df['nnoe'].drop_duplicates().values
#%%
lmode = df['mode'].drop_duplicates().values.astype(int)
lfreq = df['freq'].drop_duplicates().values.astype(int)
indnoe = [df[df['nnoe']==noei].index for noei in lnoe]
indmode = [df[df['mode']==ni].index for ni in lmode]
nnoemax = np.max(lnoe)  
# les frequences diminuent avec le nbr de noeuds :
crit = 5.e-3
critconvmu = 1. - crit
critconvfreq = 1. + crit
lfreqconv = [ critconvfreq*df[(df['nnoe']==nnoemax) & (df['mode']==ni)]['freq'].values[0] for ni in lmode ]
lmasseconv = [ critconvmu*df[(df['nnoe']==nnoemax) & (df['mode']==ni)]['masse'].values[0] for ni in lmode ]
#%% convected angular rotation speeds :
repsect1 = f"{rep_save}mu_fnnoe/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ni,modei in enumerate(lmode):
    kwargs1 = {
        "tile1": f"mus = f(nnoe) mode {ni+1}" + "\n",
        "tile_save": f"mus_fnnoe_mode_{ni+1}",
        "colx": "nnoe",
        "coly": "masse",
        "ind" : indmode[ni],
        "rep_save": repsect1,
        "label1": None,
        "labelx": r"$N_{node}$",
        "labely": r"$ \mu $",
        "color1": color1[0],
        "loc_leg": (1.01,0.),
        "scatter": True,
        "msize": 7,
        "endpoint": False,
        "xpower": 4,
        "ypower": 4,
        "lgrid": True,
        "axline": lmasseconv[ni],
        "labelxline": "converged",
    }
    traj.pltraj2d_mufgen(df, **kwargs1)

#%%
repsect1 = f"{rep_save}freq_fnnoe/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ni,modei in enumerate(lmode):
    kwargs1 = {
        "tile1": f"freq = f(nnoe) mode {ni+1}" + "\n",
        "tile_save": f"freq_fnnoe_mode_{ni+1}",
        "colx": "nnoe",
        "coly": "freq",
        "ind" : indmode[ni],
        "rep_save": repsect1,
        "label1": None,
        "labelx": r"$N_{node}$",
        "labely": r"$ f $"+" (Hz)",
        "color1": color1[0],
        "loc_leg": (1.01,0.),
        "scatter": True,
        "msize": 7,
        "endpoint": False,
        "xpower": 4,
        "ypower": 4,
        "lgrid": True,
        "axline": lfreqconv[ni],
        "labelxline": "converged",
    }
    traj.pltraj2d_mufgen(df, **kwargs1)
# %%
