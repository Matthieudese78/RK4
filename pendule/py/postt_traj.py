#!/bin/python3/
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import trajectories as traj
# import rotation as rota
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as pltcolors
from matplotlib import ticker
# import scipy
# from matplotlib.patches import PathPatch
import sys
# from rich.console import Console
# from matplotlib.font_manager import FontProperties
#%%
stoia = False
manchette = True
limpact = True
linert = True
lnortot = False
lnorcomp = True
lnormtot = False
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
xi = 0.
thini = 45.
nmode = 12
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
if (linert):
  repload = f'{repload}inert/'
  repsave = f'{repsave}inert/'
  if lnortot:
    repload = f'{repload}nortot/'
    repsave = f'{repsave}nortot/'
  if lnorcomp:
    repload = f'{repload}norcomp/'
    repsave = f'{repsave}norcomp/'
  if lnormtot:
    repload = f'{repload}normtot/'
    repsave = f'{repsave}normtot/'

repload = f'{repload}xi_{int(100.*xi)}/thini_{int(thini)}/nmode_{nmode}/'
repsave = f'{repsave}xi_{int(100.*xi)}/thini_{int(thini)}/nmode_{nmode}/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
else:
    print(f"FOLDER : {repsave} already exists.")
#%%
df = pd.read_pickle(f"{repload}result.pickle")
if linert:
  df=df[df['inert']==1]
else:
  df=df[df['inert']==0]

df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)

dt = df.iloc[1]['t'] - df.iloc[0]['t'] 
#%% energie potentielle :
Kchoc = 5.5E+07 
muk = 0.075
bamo = df['amor'][0]
thini = df['thini'][0]
h = df.loc[0,'lbar']
zsol = -2.*np.cos(thini*np.pi/180.)*h
# jx = 1.79664E-02
M = df['M'][0]
g = 9.81
hini = h * (1. - np.cos(df.loc[0,'thini']*np.pi/180.))
df['epot'] = M*g*df['uzg'] + M*g*hini
# df['ecref'] = 0.5*jx*(df['wx']**2)
df['ecbar'] = 0.*df['edef'] 
for i in np.arange(df['nmode'][0]):
  df['ecbar'] = df['ecbar'] + df[f'ec{i+1}']
  df[f'emode{i+1}'] = df[f'ec{i+1}'] + df[f'edef{i+1}']
df['eintbar'] = df['edef'] + df['ecbar']
df['ecdef'] = df['edef'] + df['ec'] + df['ecbar']

df['pene'] = df['uzpchoc'] - zsol
df['pene'] = df['pene'].apply(lambda x: 0 if x > 0 else x)
# zsol = 0.
# df['estock'] = 0.*df['edef']
df['estock'] = 0.5*Kchoc*(df['pene']**2)
df['etot'] = df['epot'] + df['edef'] + df['ec'] + df['ecbar']
# df['etot'] = df['estock'] + df['epot'] + df['edef'] + df['ec'] + df['ecbar']

if limpact:
  indchoc = df[df['fn'].abs()>0.]
  # indni = df.drop(indchoc).index
  df['efric'] = 0.*df['edef'] 
  df['efric'] = muk*df['ft'].abs()*dt

# for i,
#  df['']]
#%% localisation des chocs :
if limpact:
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
  # energie cinetique de reference
  # etotst = [df.iloc[fsti-1]['etot'] for fsti in fst]
  # df = df[df['t']<=(instant_choc[-1]+0.1)]
  df['ef'] = 0.*df['edef'] 

  # vols :  on isole les 4 premiers groupes de chocs :
  crit = 0.05
  nstchoc = 4 
  lgrp = [[] for _ in range(nstchoc)]
  t0 = df.iloc[fst[0]]['t']
  lgrp[0].append(t0)
  igrp = 0
  ichoc = 0
  while igrp <= (nstchoc-1): 
    ichoc += 1
    if ((df.iloc[fst[ichoc]]['t'] - df.iloc[fst[ichoc-1]]['t'])>=crit):
      igrp += 1
      lgrp[igrp-1].append(df.iloc[lst[ichoc-1]]['t'])
      if (igrp<nstchoc): 
        lgrp[igrp].append(df.iloc[fst[ichoc]]['t'])

  lindchoc = []
  lindvol = []
  tfinprec = 0.
  for ichoc in np.arange(nstchoc):
    lindchoc.append(df[(df['t']>=lgrp[ichoc][0]) & (df['t']<=lgrp[ichoc][1])].index)
    if (ichoc>0):
      tfinprec = lgrp[ichoc-1][1]
    lindvol.append(df[(df['t']>=tfinprec) & (df['t']<=lgrp[ichoc][0])].index)

  # valeur de l'energie totale avant chaque choc :
  lest = [df.iloc[fsti[0]-1]['ecdef'] for i,fsti in enumerate(lindchoc)]
  # energie dissipee : calcul par difference

  def calcef(df,**kwargs):
    #  ef = kwargs['lst'] - (df["etot"] + df["estock"]) 
     ef = kwargs['lst'] - (df["ecdef"] + df["estock"]) 
    #  print(f"ef = {ef}")
     return ef

  # def efdf(df, **kwargs):
  #     col3 = "ef"
  #     dict1 = {col3: df.apply(calcef,**kwargs, axis=1)}
  #     df1 = pd.DataFrame(dict1)
  #     df.loc[df1.index, col3] = df1
  #     return "Done"
  df['ef'] = 0.*df['edef']
  for i,indi in enumerate(lindchoc):
      kw = {'lst' : lest[i], 'ind' : indi}
      dict1 = {'ef' : df.apply(calcef, **kw,axis=1)}
      df1 = pd.DataFrame(dict1)
      df.loc[indi, 'ef'] = df1.iloc[indi]
      del df1

  # edamp : calcul via vn
  df['edamp'] = 0.*df['edef'] 
  edamp0 = 0.
  for fsti,lsti in zip(fst,lst):
    indi = range(fst[0],lst[0]-1)
    df.loc[indi,'edamp'] = edamp0 + (dt*bamo*df.loc[indi,'vn'].abs().cumsum())
    edamp0 = df.loc[indi,'edamp'].iloc[-1]

  # edamp = [df.at[i,'edamp'] + df.at[i-1,'edamp']]

#%% spin :
# kq = {"colname": "mrot", "q1": "quat1", "q2": "quat2", "q3": "quat3", "q4": "quat4"}
# rota.q2mdf(df, **kq)
# name_cols = ["spin"] 
# kwargs1 = {"mat": "mrot", "colnames": name_cols}
# rota.spinextrdf(df,**kwargs1)

# %%
repsect1 = f"{repsave}energies/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
kwargs1 = {
    "tile1": "wx sleeve = f(t)" + "\n",
    "tile_save": "wx_ft",
    "colx": ["t"],
    "coly": ["wx"],
    "rep_save": repsect1,
    "label1": [r"$W_{X}$",r"$E_{vibr}$",r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_{X}$"+" (rad/s)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "wy sleeve = f(t)" + "\n",
    "tile_save": "wy_ft",
    "colx": ["t"],
    "coly": ["wy"],
    "rep_save": repsect1,
    "label1": [r"$W_{Y}$",r"$E_{vibr}$",r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_{Y}$"+" (rad/s)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "wz sleeve = f(t)" + "\n",
    "tile_save": "wz_ft",
    "colx": ["t"],
    "coly": ["wz"],
    "rep_save": repsect1,
    "label1": [r"$W_{Z}$",r"$E_{vibr}$",r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_{Z}$"+" (rad/s)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)
#test reco :
kwargs1 = {
    "tile1": "uzpchoc vs uzp2" + "\n",
    "tile_save": "uzreco_vs_uzliai_ft",
    "colx": ["t","t"],
    "coly": ["uzpchoc","uzp2"],
    "rep_save": repsect1,
    "label1": [r"$u_{z}(P_{choc})$",r"$u_z(P_2)$",r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_z$"+" (m)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "energies sleeve = f(t)" + "\n",
    "tile_save": "energies_ft",
    "colx": ["t","t","t","t"],
    "coly": ["ec","eintbar","epot","etot"],
    "rep_save": repsect1,
    "label1": [r"$E_{kin}^{ref}$",r"$E_{vibr}$",r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{kin}^{ref},E_{vibr},E_{pot},E_{tot}$"+" (J)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
    "alpha" : [0.7]*10,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "energies sleeve = f(t)" + "\n",
    "tile_save": "ec_ft",
    "colx": ["t"],
    "coly": ["ec","edef","epot","etot"],
    "rep_save": repsect1,
    "label1": [r"$E_{kin}$",r"$E_{def}$",r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{kin}$"+" (J)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "energies sleeve = f(t)" + "\n",
    "tile_save": "edef_ft",
    "colx": ["t"],
    "coly": ["edef","epot","etot"],
    "rep_save": repsect1,
    "label1": [r"$E_{def}$",r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{def}$"+" (J)",
    "color1": [color1[1]],
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "energies sleeve = f(t)" + "\n",
    "tile_save": "epot_ft",
    "colx": ["t"],
    "coly": ["epot","etot"],
    "rep_save": repsect1,
    "label1": [r"$E_{pot}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{pot}$"+" (J)",
    "color1": [color1[2]],
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

# for i in np.arange(df['nmode'][0]):
for i in np.arange(4):
  kwargs1 = {
      "tile1": f"energies mode{i+1} = f(t)" + "\n",
      "tile_save": f"energies_mode{i+1}",
      "colx": ["t","t"],
      "coly": [f"ec{i+1}",f"edef{i+1}"],
      "rep_save": repsect1,
      "label1": [r"$E_{kin}^{%d}$" % (i+1),r"$E_{def}^{%d}$" % (i+1),r"$E_{pot}$",r"$E_{tot}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$E_{kin},E_{def},E_{pot},E_{tot}$"+" (J)",
      "color1": color1,
      "endpoint": [False,False,False,False],
      "xpower": 5,
      "ypower": 5,
      "alpha" : [0.7]*10,
  }
  traj.pltraj2d(df, **kwargs1)
  kwargs1 = {
      "tile1": f"energy mode{i+1} = f(t)" + "\n",
      "tile_save": f"eint_mode{i+1}",
      "colx": ["t"],
      "coly": [f"emode{i+1}",f"edef{i+1}"],
      "rep_save": repsect1,
      "label1": [r"$E_{int}^{%d}$" % (i+1),r"$E_{def}^{%d}$" % (i+1),r"$E_{pot}$",r"$E_{tot}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$E_{kin},E_{def},E_{pot},E_{tot}$"+" (J)",
      "color1": color1,
      "endpoint": [False,False,False,False],
      "xpower": 5,
      "ypower": 5,
  }
  traj.pltraj2d(df, **kwargs1)

# %%
repsect1 = f"{repsave}chocs/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ichoc in np.arange(nstchoc): 
  indexchoc = df[(df['t']>=(lgrp[ichoc][0]-1.e-3)) & (df['t']<=(lgrp[ichoc][1]+1.e-3))].index
  kwargs = {
      "tile1": f"traj choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_ft",
      "x": df.iloc[indexchoc]['t'], 
      "y": df.iloc[indexchoc]['uzpchoc'], 
      # "y": df.iloc[indexchoc]['pene'], 
      "rep_save": repsect1,
      "label1": r"$u_z$",
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_z$" + " (m)",
      "labelsol": "ground",
      "color1": color1[0],
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "sol": zsol,
      # "sol": 0.,
      "loc_leg": "upper right",
  }
  traj.pltraj2d_list_sol(**kwargs)

#%% lecture des arguments :
for ichoc in np.arange(nstchoc): 
  indexchoc = df[(df['t']>=(lgrp[ichoc][0]-1.e-3)) & (df['t']<=(lgrp[ichoc][1]+1.e-3))].index

  kwargs1 = {
      "tile1": "uzpchoc vs uzp2" + "\n",
      "tile_save": f"uzreco_vs_uzliai_{ichoc}",
      "colx": ["t","t"],
      "coly": ["uzpchoc","uzp2"],
      "rep_save": repsect1,
      "label1": [r"$u_{z}(P_{choc})$",r"$u_z(P_2)$",r"$E_{pot}$",r"$E_{tot}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_z$"+" (m)",
      "color1": color1,
      "endpoint": [False,False,False,False],
      "xpower": 5,
      "ypower": 5,
  }
  traj.pltraj2d(df.iloc[indexchoc], **kwargs1)

  kwargs = {
      "tile1": f"traj+ener choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_ener_ft",
      "x": df.iloc[indexchoc]['t'], 
      "y": df.iloc[indexchoc]['uzpchoc'], 
      # "y": df.iloc[indexchoc]['pene'], 
      "y2": [df.iloc[indexchoc]['ec'], 
      df.iloc[indexchoc]['eintbar'],
      df.iloc[indexchoc]['etot'], 
      df.iloc[indexchoc]['estock']], 
      "rep_save": repsect1,
      "label1": r"$u_z$",
      "labelsol": "floor",
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label2": [r"$E_{kin}^{ref}$",r"$E_{vibr}$",r"$E_{tot}$",r"$E_{stock}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_z$" + " (m)",
      "labely2": "Energy (J)",
      "color1": color1[0],
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "sol": zsol,
      "loc_leg": "upper left",
      "loc_leg2": "upper right",
  }
  traj.pltraj2d_list_2axes(**kwargs)
  kwargs = {
      "tile1": f"ener choc {ichoc} = f(t)" + "\n",
      "tile_save": f"enerchoc{ichoc}_ft",
      "x": [df.iloc[indexchoc]['t']]*4, 
      "y": [df.iloc[indexchoc]['ec'], 
            df.iloc[indexchoc]['eintbar'],
            df.iloc[indexchoc]['estock'], 
            df.iloc[indexchoc]['etot']], 
      "rep_save": repsect1,
      "label1": [r"$E_{kin}^{ref}$",r"$E_{vibr}$",r"$E_{stock}$",r"$E_{tot}$"],
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
      "tile1": f"traj+ener choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_ener2_ft",
      "x": [df.iloc[indexchoc]['t']]*3, 
      "y": [ 
      df.iloc[indexchoc]['edamp'], 
      df.iloc[indexchoc]['efric'], 
      df.iloc[indexchoc]['ef']], 
      "rep_save": repsect1,
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label1": [r"$E_{damp}$",r"$E_{fric}$",r"$E_{f}$"],
      "labelx": r"$t \quad (s)$",
      "labely": "Energy (J)",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "loc_leg": "upper left",
  }
  traj.pltraj2d_list(**kwargs)

  kwargs = {
      "tile1": f"pusure choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_pusure_ft",
      "x": [df.iloc[indexchoc]['t']], 
      "y": [df.iloc[indexchoc]['pusure']], 
      "rep_save": repsect1,
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label1": [r"$P_{wear}$",r"$E_{fric}$",r"$E_{f}$"],
      "labelx": r"$t \quad (s)$",
      "labely": "Wear Power (W)",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "loc_leg": "upper left",
  }
  traj.pltraj2d_list(**kwargs)
  kwargs = {
      "tile1": f"vitesse normale choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_vn_ft",
      "x": [df.iloc[indexchoc]['t']], 
      "y": [df.iloc[indexchoc]['vn']], 
      "rep_save": repsect1,
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label1": [r"$v_{n}$",r"$E_{fric}$",r"$E_{f}$"],
      "labelx": r"$t \quad (s)$",
      "labely": "Normal speed (m/s)",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "loc_leg": "upper left",
  }
  traj.pltraj2d_list(**kwargs)

  # kwargs = {
  #     "tile1": f"force normale choc {ichoc} = f(t)" + "\n",
  #     "tile_save": f"choc{ichoc}_fn_ft",
  #     "x": [df[(df['t']>=lgrp[ichoc][0]) & (df['t']<=lgrp[ichoc][1])]['t']], 
  #     "y": [df[(df['t']>=lgrp[ichoc][0]) & (df['t']<=lgrp[ichoc][1])]['fn']], 
  #     "rep_save": repsect1,
  #     # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
  #     "label1": [r"$F_{n}$",r"$E_{fric}$",r"$E_{f}$"],
  #     "labelx": r"$t \quad (s)$",
  #     "labely": "Normal Force (N)",
  #     "color1": color1,
  #     "endpoint": False,
  #     "xpower": 5,
  #     "ypower": 5,
  #     "loc_leg": "upper left",
  # }
  # traj.pltraj2d_list(**kwargs)

  kwargs = {
      "tile1": f"traj+fn choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_uzfn_ft",
      "x": df.iloc[indexchoc]['t'], 
      "y": df.iloc[indexchoc]['uzpchoc'], 
      "y2": [df.iloc[indexchoc]['fn']], 
      "rep_save": repsect1,
      "label1": r"$u_z$",
      "labelsol": "floor",
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label2": [r"$F_{n}$",r"$E_{vibr}$",r"$E_{tot}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_z$" + " (m)",
      "labely2": "Normal Force (N)",
      "color1": color1[0],
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "sol": zsol,
      # "sol": None,
      "loc_leg": "upper left",
      "loc_leg2": "upper right",
  }
  traj.pltraj2d_list_2axes(**kwargs)
# %%
repsect1 = f"{repsave}vols/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ichoc in np.arange(nstchoc-1): 
  indexvol = df[(df['t']>=lgrp[ichoc][1]) & (df['t']<=lgrp[ichoc+1][0])].index
  kwargs = {
      "tile1": f"traj vol {ichoc} = f(t)" + "\n",
      "tile_save": f"vol{ichoc}_ft",
      "x": df.iloc[indexvol]['t'], 
      "y": df.iloc[indexvol]['uzpchoc'], 
      "rep_save": repsect1,
      "label1": r"$u_z$",
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_z$" + " (m)",
      "labelsol": "floor",
      "color1": color1[0],
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "sol": zsol,
      "loc_leg": "upper right",
  }
  traj.pltraj2d_list_sol(**kwargs)

#%% lecture des arguments :
for ichoc in np.arange(nstchoc-1): 
  indexvol = df[(df['t']>=lgrp[ichoc][1]) & (df['t']<=lgrp[ichoc+1][0])].index
  kwargs = {
      "tile1": f"traj+ener vol {ichoc} = f(t)" + "\n",
      "tile_save": f"vol{ichoc}_ener_ft",
      "x": df.iloc[indexvol]['t'], 
      "y": df.iloc[indexvol]['uzpchoc'], 
      "y2": [df.iloc[indexvol]['ec'], 
      df.iloc[indexvol]['eintbar'], 
      df.iloc[indexvol]['etot']], 
      "rep_save": repsect1,
      "label1": r"$u_z$",
      "labelsol": "floor",
      # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
      "label2": [r"$E_{kin}^{ref}$",r"$E_{vibr}$",r"$E_{tot}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_z$" + " (m)",
      "labely2": "Energy (J)",
      "color1": color1[0],
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "sol": zsol,
      "loc_leg": "upper left",
      "loc_leg2": "upper right",
  }
  traj.pltraj2d_list_2axes(**kwargs)
#%%
sys.exit()
#%%
title1 = kwargs["tile1"]
title_save = kwargs["tile_save"]
X = kwargs["x"]
Y = kwargs["y"]
rep_save = kwargs["rep_save"]
label1 = kwargs["label1"]
labelx = kwargs["labelx"]
labely = kwargs["labely"]
color1 = kwargs["color1"]
loc_leg = kwargs["loc_leg"]

lw = 0.8  # linewidth
f = plt.figure(figsize=(8, 6), dpi=600)
axes = f.gca()

axes.set_title(title1)
if type(X) == list:
    if type(Y) != list:
        print(f"Les 3 inputs doivent etre des lists!")
    x_data = [xi for xi in X]
    axes.set_xlim(xmin=np.min(x_data[0]),xmax=np.max(x_data[0]))
else:
    x_data = X
    axes.set_xlim(xmin=np.min(X),xmax=np.max(X))

if type(Y) == list:
    y_data = [yi for yi in Y]
else:
    y_data = Y

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-kwargs['xpower'], kwargs['xpower']))
axes.xaxis.set_major_formatter(formatter)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-kwargs['ypower'], kwargs['ypower']))
axes.yaxis.set_major_formatter(formatter)

if type(X) != list:
    plt.plot(X, Y, label=label1, linewidth=lw, color=color1)
else:
    [
        axes.plot(xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i])
        for i, xi in enumerate(x_data)
    ]
if isinstance(kwargs['sol'],float):
  axes.plot(kwargs['x'],[kwargs['sol']]*len(kwargs['x']),c='black',linestyle='--',label=kwargs['labelsol'])
# axes.set_facecolor('None')
axes.set_facecolor("white")
axes.grid(False)
axes.set_xlabel(labelx)
axes.set_ylabel(labely)
axes.legend(loc=kwargs['loc_leg'])

if isinstance(kwargs['y2'],list):
  ax2 = axes.twinx()
  for i,yi in enumerate(kwargs['y2']):
    ax2.plot(kwargs['x'],yi,linestyle='-.',label=kwargs['label2'][i])
  ax2.legend(loc=kwargs['loc_leg2'])

f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
plt.close("all")
# %%
