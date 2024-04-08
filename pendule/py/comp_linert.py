#!/bin/python3
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import trajectories as traj
import rotation as rota
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as pltcolors
from matplotlib import ticker
import scipy
# from matplotlib.patches import PathPatch
import sys
from rich.console import Console
from matplotlib.font_manager import FontProperties
#%%
stoia = False
manchette = True
limpact = True
linert = True
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
xi = 0.5
thini = 45.
nmode = 15
#%% rep_load
repload = './pickle/'
repsave = './fig/'
if (stoia):
  repload = f'{repload}stoia/'
  repsave = f'{repsave}stoia/'
if (manchette):
  repload = f'{repload}manchette/'
  repsave = f'{repsave}manchette/'
if (limpact):
  repload = f'{repload}impact/'
  repsave = f'{repsave}impact/'
if (not limpact):
  repload = f'{repload}no_impact/'
  repsave = f'{repsave}no_impact/'

reploadlin = repload
repload = f'{repload}inert/'

repsave = f'{repsave}compare/'

repload = f'{repload}xi_{int(100.*xi)}/thini_{int(thini)}/nmode_{nmode}/'
reploadlin = f'{reploadlin}xi_{int(100.*xi)}/thini_{int(thini)}/nmode_{nmode}/'

repsave = f'{repsave}xi_{int(100.*xi)}/thini_{int(thini)}/nmode_{nmode}/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
else:
    print(f"FOLDER : {repsave} already exists.")
#%%
df = pd.read_pickle(f"{repload}result.pickle")
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)

dflin = pd.read_pickle(f"{reploadlin}result.pickle")
dflin.sort_values(by='t',inplace=True)
dflin.reset_index(drop=True,inplace=True)

dfcomp = [df,dflin]
del df
del dflin
#%%
lgrp = [[],[]]
for i,df in enumerate(dfcomp):
  col1 = 'fn'
  df['tag'] = df.loc[:,col1].abs() > 0.
  fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
  lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
  # prb1 = [(i,j) for i,j in zip(fst,lst)]
  dt = df.iloc[fst[0]+1]['t'] - df.iloc[fst[0]]['t']
  # on vire le dernier choc :
  # fst = fst[:-1]
  # lst = lst[:-1]
  instant_choc = [df.iloc[fsti]['t'] for fsti in fst]
  tchoc = [ dt*(j-i) for i,j in zip(fst,lst) ]
  # df = df[df['t']<=(instant_choc[-1]+0.1)]

  crit = 0.05
  nstchoc = 2 
  lgrp[i] = [[] for _ in range(nstchoc)]
  t0 = df.iloc[fst[0]]['t']
  lgrp[i][0].append(t0)
  igrp = 0
  ichoc = 0
  while igrp <= (nstchoc-1): 
    ichoc += 1
    if ((df.iloc[fst[ichoc]]['t'] - df.iloc[fst[ichoc-1]]['t'])>=crit):
      igrp += 1
      lgrp[i][igrp-1].append(df.iloc[lst[ichoc-1]]['t'])
      if (igrp<nstchoc): 
        lgrp[i][igrp].append(df.iloc[fst[ichoc]]['t'])
#%%
ltchoc = []
ltvol = []
tfinprec = 0.
for ichoc in np.arange(nstchoc):
  tdeb = np.min([lgrp[0][ichoc][0],lgrp[1][ichoc][0]])
  tfin = np.max([lgrp[0][ichoc][1],lgrp[1][ichoc][1]])
  ltchoc.append([tdeb,tfin])
  if (ichoc>0):
    tfinprec = ltchoc[ichoc-1][1]
  ltvol.append([tfinprec,tdeb])

#%% computation energies
for i,df in enumerate(dfcomp):
  h = df['lbar'][0]
  # jx = 1.79664E-02
  M = df['M'][0]
  g = 9.81
  hini = h * (1. - np.cos(df['thini']*np.pi/180.))
  df['epot'] = M*g*df['uzg'] + M*g*hini
  # df['ecref'] = 0.5*jx*(df['wx']**2)
  df['ecbar'] = 0.*df['edef'] 
  for i in np.arange(df['nmode'][0]):
    df['ecbar'] = df['ecbar'] + df[f'ec{i+1}']
    df[f'emode{i+1}'] = df[f'ec{i+1}'] + df[f'edef{i+1}']
  df['eintbar'] = df['edef'] + df['ecbar']
  df['etot'] = df['epot'] + df['edef'] + df['ec'] + df['ecbar']
  df['ecdef'] = df['edef'] + df['ec'] + df['ecbar']
#%% index de choc :
lindchoc = [[],[]]
lindvol = [[],[]]
for ichoc in np.arange(nstchoc):
  lindchoc[0].append(dfcomp[0][(dfcomp[0]['t']>=ltchoc[ichoc][0]) & (dfcomp[0]['t']<=ltchoc[ichoc][1])].index)
  lindchoc[1].append(dfcomp[1][(dfcomp[1]['t']>=ltchoc[ichoc][0]) & (dfcomp[1]['t']<=ltchoc[ichoc][1])].index)
  lindvol[0].append(dfcomp[0][(dfcomp[0]['t']>=ltvol[ichoc][0]) & (dfcomp[0]['t']<=ltvol[ichoc][1])].index)
  lindvol[1].append(dfcomp[1][(dfcomp[1]['t']>=ltvol[ichoc][0]) & (dfcomp[1]['t']<=ltvol[ichoc][1])].index)
#%%
repsect1 = f"{repsave}macro/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%

#%%
repsect1 = f"{repsave}chocs/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ichoc in np.arange(nstchoc): 
  kwargs = {
      "tile1": f"traj choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_ft",
      "x": [dfcomp[0].iloc[lindchoc[0][ichoc]]['t'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['t']], 
      "y": [dfcomp[0].iloc[lindchoc[0][ichoc]]['uzp2'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['uzp2']], 
      "rep_save": repsect1,
      "label1": [r"$u_z^{inert}$",r"$u_z^{lin}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_z$" + " (m)",
      "labelsol": "floor",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "sol": -2.*np.cos(df['thini'][0]*np.pi/180.)*h,
      "loc_leg": "upper right",
  }
  traj.pltraj2d_list_sol(**kwargs)
#%%
for ichoc in np.arange(nstchoc): 
  kwargs = {
      "tile1": f"enertot choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_energies_ft",
      "x": [dfcomp[0].iloc[lindchoc[0][ichoc]]['t'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['t'], 
            dfcomp[0].iloc[lindchoc[0][ichoc]]['t'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['t'], 
            dfcomp[0].iloc[lindchoc[0][ichoc]]['t'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['t']], 
      "y": [dfcomp[0].iloc[lindchoc[0][ichoc]]['etot'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['etot'], 
            dfcomp[0].iloc[lindchoc[0][ichoc]]['ec'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['ec'], 
            dfcomp[0].iloc[lindchoc[0][ichoc]]['eintbar'], 
            dfcomp[1].iloc[lindchoc[1][ichoc]]['eintbar']], 
      "rep_save": repsect1,
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label1": [r"$E_{tot}$","LTE : "+r"$E_{tot}$",r"$E_{kin}^{ref}$","LTE : "+r"$E_{kin}^{ref}$",r"$E_{vibr}$","LTE : "+r"$E_{vibr}$"],
      "labelx": r"$t \quad (s)$",
      "labely": "Energy (J)",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "loc_leg": "upper left",
  }
  traj.pltraj2d_list(**kwargs)
# %%
repsect1 = f"{repsave}vols/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ivol in np.arange(nstchoc): 
  kwargs = {
      "tile1": f"traj vol {ivol} = f(t)" + "\n",
      "tile_save": f"vol{ivol}_traj_ft",
      "x": [dfcomp[0].iloc[lindvol[0][ivol]]['t'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['t']], 
      "y": [dfcomp[0].iloc[lindvol[0][ivol]]['uzp2'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['uzp2']], 
      "rep_save": repsect1,
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label1": [r"$u_{z}^{inert}$","LTE : "+r"$u_{z}^{lin}$",r"$E_{kin}^{ref}$","LTE : "+r"$E_{kin}^{ref}$",r"$E_{vibr}$","LTE : "+r"$E_{vibr}$"],
      "labelx": r"$t \quad (s)$",
      "labely": "Energy (J)",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "loc_leg": "upper right",
  }
  traj.pltraj2d_list(**kwargs)
  kwargs = {
      "tile1": f"enertot vol {ivol} = f(t)" + "\n",
      "tile_save": f"vol{ivol}_energies_ft",
      "x": [dfcomp[0].iloc[lindvol[0][ivol]]['t'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['t'], 
            dfcomp[0].iloc[lindvol[0][ivol]]['t'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['t'], 
            dfcomp[0].iloc[lindvol[0][ivol]]['t'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['t']], 
      "y": [dfcomp[0].iloc[lindvol[0][ivol]]['etot'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['etot'], 
            dfcomp[0].iloc[lindvol[0][ivol]]['ec'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['ec'], 
            dfcomp[0].iloc[lindvol[0][ivol]]['eintbar'], 
            dfcomp[1].iloc[lindvol[1][ivol]]['eintbar']], 
      "rep_save": repsect1,
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label1": [r"$E_{tot}$","LTE : "+r"$E_{tot}$",r"$E_{kin}^{ref}$","LTE : "+r"$E_{kin}^{ref}$",r"$E_{vibr}$","LTE : "+r"$E_{vibr}$"],
      "labelx": r"$t \quad (s)$",
      "labely": "Energy (J)",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "loc_leg": "upper center",
  }
  traj.pltraj2d_list(**kwargs)
# %%
