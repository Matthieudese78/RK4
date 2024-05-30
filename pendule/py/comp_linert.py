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
trig = True
limpact = True
linert = True
lnorcomp = True
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
bamo = 0.
Kchoc = 5.5e07
nstchoctr = 3
g = 9.81
if stoia:
  M = 0.59868
  h = 0.6
  Jx = 0.07185
if manchette:
  M = 9.2774
  h = 1.5349
  Jx = 1.8255
if trig:
  vimpact = 4.
xi = bamo / (2.0 * M * (np.sqrt(Kchoc / M)))

thinc = 45.
nmode = 80
# long calcul ?
macro = True

#%% rep_load
repload = './pickle/'
repsave = './fig/'
if (stoia):
  repload = f'{repload}stoia/'
  repsave = f'{repsave}stoia/'
if (manchette):
  repload = f'{repload}manchette/'
  repsave = f'{repsave}manchette/'
if trig: 
  repload = f"{repload}trig/"
  repsave = f"{repsave}trig/"
if (limpact):
  repload = f'{repload}impact/'
  repsave = f'{repsave}impact/'
if (not limpact):
  repload = f'{repload}no_impact/'
  repsave = f'{repsave}no_impact/'

repsave = f'{repsave}compare/xi_{int(100.*xi)}/thinc_{int(thinc)}/nmode_{nmode}/vc_{int(vimpact)}/'

reploadlin = repload
repload = f'{repload}inert/'
if lnorcomp:
  repload = f'{repload}norcomp/'

repload    = f'{repload}xi_{int(100.*xi)}/thinc_{int(thinc)}/'
reploadlin = f'{reploadlin}xi_{int(100.*xi)}/thinc_{int(thinc)}/'
if macro:
  repload    = f'{repload}calc_3s/'
  reploadlin = f'{reploadlin}calc_3s/'

repload    = f'{repload}nmode_{nmode}/vc_{int(vimpact)}/'
reploadlin = f'{reploadlin}nmode_{nmode}/vc_{int(vimpact)}/'

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

#%% chocs / vols
crit = 0.01
#%% P1
lindchoc = [[],[]]
lindvol = [[],[]]
lgrp = [[],[]]
nbchoc = [[],[]]
ltchoc = [[],[]]
lfst = [[],[]]
llst = [[],[]]
for i,df in enumerate(dfcomp):
  nstchoc = 200 
  col1 = 'fn'
  df['tag'] = df.loc[:,col1].abs() > 0.
  fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
  lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
  # prb1 = [(i,j) for i,j in zip(fst,lst)]
  dt = df.iloc[1]['t'] - df.iloc[0]['t']
  # on vire le dernier choc :
  # fst = fst[:-1]
  # lst = lst[:-1]
  instant_choc = [df.iloc[fsti]['t'] for fsti in fst]
  tchoc = [ dt*(j-i) for i,j in zip(fst,lst) ]
  nbchoc[i].append(len(fst))
  ltchoc[i] = tchoc
  lfst[i] = fst
  llst[i] = lst
  # df = df[df['t']<=(instant_choc[-1]+0.1)]

  # chocs
  lgrp[i] = [[] for _ in range(nstchoc)]
  t0 = df.iloc[fst[0]]['t']
  lgrp[i][0].append(t0)
  lgrp[i][0].append(df.iloc[lst[-1]]['t'])
  igrp = 0
  for ichoc,tci in enumerate(fst):
    ichoc += 1
    if (ichoc<=(len(fst)-1)):
      if ((df.iloc[fst[ichoc]]['t'] - df.iloc[lst[ichoc-1]]['t'])>=crit):
        igrp += 1
        lgrp[i][igrp-1].append(df.iloc[lst[ichoc-1]]['t'])
        if ((igrp<nstchoc) and (ichoc<=(len(fst)-1))): 
          lgrp[i][igrp].append(df.iloc[fst[ichoc]]['t'])
    if ((ichoc==(len(fst)-1)) and (igrp<=0)):
      lgrp[i][igrp].append(df.iloc[lst[ichoc]]['t'])

  if (igrp>0):
    lgrp[i][0] = [lgrp[i][0][0],lgrp[i][0][2]]
  nstchoc = np.min([nstchoc,igrp])
  lgrp[i] = lgrp[i][:nstchoc]

  # vols
for i,df in enumerate(dfcomp):
  # lindchoc = []
  # lindvol = []
  tfinprec = 0.
  for ichoc in np.arange(nstchoctr):
    tdeb = np.min([lgrp[0][ichoc][0],lgrp[1][ichoc][0]])
    tfin = np.max([lgrp[0][ichoc][1],lgrp[1][ichoc][1]])
    indc = df[(df['t']>=tdeb) & (df['t']<=tfin)].index
    lindchoc[i].append(indc)
    # lindchoc.append([tdeb,tfin])
    if (ichoc>0):
      tfinprec = df.iloc[lindchoc[i][ichoc-1][-1]]['t']
    indv = df[(df['t']>=tfinprec) & (df['t']<=tdeb)].index
    lindvol[i].append(indv)
    # lindvol.append([tfinprec,tdeb])



#%% P2
lindchoc2 = [[],[]]
lindvol2 = [[],[]]
lgrp2 = [[],[]]
nbchoc2 = [[],[]]
ltchoc2 = [[],[]]
lfst2 = [[],[]]
llst2 = [[],[]]
for i,df in enumerate(dfcomp):
  nstchoc = 200 
  col1 = 'fn2'
  df['tag'] = df.loc[:,col1].abs() > 0.
  fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
  lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
  # prb1 = [(i,j) for i,j in zip(fst,lst)]
  dt = df.iloc[1]['t'] - df.iloc[0]['t']
  # on vire le dernier choc :
  # fst = fst[:-1]
  # lst = lst[:-1]
  instant_choc = [df.iloc[fsti]['t'] for fsti in fst]
  tchoc = [ dt*(j-i) for i,j in zip(fst,lst) ]
  nbchoc2[i].append(len(fst))
  ltchoc2[i] = tchoc
  lfst2[i] = fst
  llst2[i] = lst
  # df = df[df['t']<=(instant_choc[-1]+0.1)]

  # chocs
  lgrp2[i] = [[] for _ in range(nstchoc)]
  t0 = df.iloc[fst[0]]['t']
  lgrp2[i][0].append(t0)
  lgrp2[i][0].append(df.iloc[lst[-1]]['t'])
  igrp = 0
  for ichoc,tci in enumerate(fst):
    ichoc += 1
    if (ichoc<=(len(fst)-1)):
      if ((df.iloc[fst[ichoc]]['t'] - df.iloc[lst[ichoc-1]]['t'])>=crit):
        igrp += 1
        lgrp2[i][igrp-1].append(df.iloc[lst[ichoc-1]]['t'])
        if ((igrp<nstchoc) and (ichoc<=(len(fst)-1))): 
          lgrp2[i][igrp].append(df.iloc[fst[ichoc]]['t'])
    if ((ichoc==(len(fst)-1)) and (igrp<=0)):
      lgrp2[i][igrp].append(df.iloc[lst[ichoc]]['t'])

  if (igrp>0):
    lgrp2[i][0] = [lgrp2[i][0][0],lgrp2[i][0][2]]
  nstchoc = np.min([nstchoc,igrp])
  lgrp2[i] = lgrp2[i][:nstchoc]

  # vols
for i,df in enumerate(dfcomp):
  # lindchoc2 = []
  # lindvol2 = []
  tfinprec = 0.
  for ichoc in np.arange(nstchoctr):
    tdeb = np.min([lgrp2[0][ichoc][0],lgrp2[1][ichoc][0]])
    tfin = np.max([lgrp2[0][ichoc][1],lgrp2[1][ichoc][1]])
    indc = df[(df['t']>=tdeb) & (df['t']<=tfin)].index
    lindchoc2[i].append(indc)
    # lindchoc2.append([tdeb,tfin])
    if (ichoc>0):
      tfinprec = df.iloc[lindchoc2[i][ichoc-1][-1]]['t']
    indv = df[(df['t']>=tfinprec) & (df['t']<=tdeb)].index
    lindvol2[i].append(indv)
    # lindvol2.append([tfinprec,tdeb])

#%% first macro impact at P1 :
imacrop1 = np.where(dfcomp[0].iloc[llst[0]]['t']<=lgrp[0][0][1])
imax_p1 = np.max(imacrop1) 
tmean_imacrop1 = np.mean(ltchoc[0][:imax_p1])
# P2 :
imacrop2 = np.where(dfcomp[0].iloc[llst2[0]]['t']<=lgrp2[0][0][1])
imax_p2 = np.max(imacrop2) 
tmean_imacrop2 = np.mean(ltchoc2[0][:imax_p2])
#%% intersection des vols :
linter = [[],[]]
for i,df in enumerate(dfcomp):
  # lindchoc = []
  # lindvol = []
    i1 = lindvol[i][2]
    i2 = lindvol2[i][1]
    indinter = i1.intersection(i2)
    linter[i].append(indinter)
#%% computation energies
consener = []
for i,df in enumerate(dfcomp):
  Kchoc = 5.5E+07 
  muk = 0.075
  bamo = df['amor'][0]
  thini = df['thini'][0]
  M = df['M'][0]
  g = 9.81
  h = df['lbar'][0]
  hini = h * (1. - np.cos(df['thini']*np.pi/180.))
  zsol = -h*(1.+ np.cos(thini*np.pi/180.))
  if trig:
    yp10 = 0.
    yp20 = h
    tc = (vimpact/g)
    zsol  = (df.loc[0,'uypchoc'] + yp10) - g*(tc**2)/2.
    zsol2 = zsol - h

  df['epot'] = M*g*df['uyg'] + M*g*hini
  # df['ecref'] = 0.5*jx*(df['wx']**2)
  df['ecbar'] = 0.*df['edef'] 
  for j in np.arange(df['nmode'][0]):
    df['ecbar'] = df['ecbar'] + df[f'ec{j+1}']
    df[f'emode{j+1}'] = df[f'ec{j+1}'] + df[f'edef{j+1}']
  df['eintbar'] = df['edef'] + df['ecbar']
  df['ecdef'] = df['edef'] + df['ec'] + df['ecbar']

  # estock in the ground :
  df['pene'] = df['uypchoc'] - zsol
  df['pene2'] = df['uypchoc2'] - zsol2
  df['pene'] = df['pene'].apply(lambda x: 0 if x > 0 else x)
  df['pene2'] = df['pene2'].apply(lambda x: 0 if x > 0 else x)
  df['estock'] = 0.5*Kchoc*(df['pene']**2) + 0.5*Kchoc*(df['pene2']**2)

  df['etot'] = df['epot'] + df['edef'] + df['ec'] + df['ecbar']
  df['etotstock'] = 0.*df['edef'] 
  df['etotstock'] = df['etot'] + df['estock']
  consener.append(100.*(df.iloc[0]['etot'] - df.iloc[-1]['etot']) / df.iloc[0]['etot'])
  if limpact:
    indchoc = df[df['fn'].abs()>0.]
    # indni = df.drop(indchoc).index
    df['efric'] = 0.*df['edef'] 
    df['efric'] = muk*df['ft'].abs()*dt

    lest = [df.iloc[fsti[0]-1]['etot'] for j,fsti in enumerate(lindchoc[i])]
    lest2 = [df.iloc[fsti[0]-1]['etot'] for j,fsti in enumerate(lindchoc2[i])]
    # energie dissipee : calcul par difference
    def calcef(df,**kwargs):
       ef = kwargs['lst'] - (df["etot"] + df["estock"]) 
       return ef
    df['ef'] = 0.*df['edef']
    for j,indi in enumerate(lindchoc[i]):
        kw = {'lst' : lest[j], 'ind' : indi}
        dict1 = {'ef' : df.apply(calcef,**kw,axis=1)}
        df1 = pd.DataFrame(dict1)
        df.loc[indi, 'ef'] = df1.iloc[indi]
        del df1
    for j,indi in enumerate(lindchoc2[i]):
        kw = {'lst' : lest2[j], 'ind' : indi}
        dict1 = {'ef' : df.apply(calcef,**kw,axis=1)}
        df1 = pd.DataFrame(dict1)
        df.loc[indi, 'ef'] = df1.iloc[indi]
        del df1
# moyenne de l'energie totale a la fin du calcul :    
moylast = np.mean(dfcomp[0].iloc[-1000:]['etot']+dfcomp[0].iloc[-1000:]['estock'])
etot0 = dfcomp[0].iloc[0]['etot']
np.abs(etot0 - moylast)*100./etot0
#%% index de choc :
# lindchoc = [[],[]]
# lindvol = [[],[]]
# for ichoc in np.arange(nstchoc):
#   lindchoc[0].append(dfcomp[0][(dfcomp[0]['t']>=ltchoc[ichoc][0]) & (dfcomp[0]['t']<=ltchoc[ichoc][1])].index)
#   lindchoc[1].append(dfcomp[1][(dfcomp[1]['t']>=ltchoc[ichoc][0]) & (dfcomp[1]['t']<=ltchoc[ichoc][1])].index)
#   lindvol[0].append(dfcomp[0][(dfcomp[0]['t']>=ltvol[ichoc][0]) & (dfcomp[0]['t']<=ltvol[ichoc][1])].index)
#   lindvol[1].append(dfcomp[1][(dfcomp[1]['t']>=ltvol[ichoc][0]) & (dfcomp[1]['t']<=ltvol[ichoc][1])].index)
#%%
repsect1 = f"{repsave}macro/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
kwargs = {
    "tile1": f"ener tot beam = f(t)" + "\n",
    "tile_save": f"enertot_beam_ft",
    "x": [dfcomp[0]['t'],
          dfcomp[1]['t']], 
    "y": [dfcomp[0]['etotstock'],
          dfcomp[1]['etotstock']], 
    "rep_save": repsect1,
    # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
    # "label1": [r"$E_{tot}^{inert}+E_{stock}$",r"$E_{tot}^{LTE}+E_{stock}$"],
    "label1": [r"$\mathcal{E}_{inert}$",r"$\mathcal{E}_{LTE}$"],
    "labelx": r"$t \quad (s)$",
    "labely": "Energy (J)",
    "color1": color1,
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "loc_leg": (1.01,0.85),
    "alpha" : 0.7,
}
traj.pltraj2d_list(**kwargs)

kwargs = {
    "tile1": f"ener friction = f(t)" + "\n",
    "tile_save": f"enerfric_ft",
    "x": [dfcomp[0]['t'],
          dfcomp[1]['t']], 
    "y": [dfcomp[0]['efric'],
          dfcomp[1]['efric']], 
    "rep_save": repsect1,
    # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
    "label1": [r"$E_{fric}^{inert}$",r"$E_{fric}^{LTE}$"],
    "labelx": r"$t \quad (s)$",
    "labely": "Energy (J)",
    "color1": color1,
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "loc_leg": (1.01,0.85),
}
traj.pltraj2d_list(**kwargs)

kwargs = {
    "tile1": f"ener kin beam {ichoc} = f(t)" + "\n",
    "tile_save": f"ekinref_ft",
    "x": [dfcomp[0]['t'],
          dfcomp[1]['t']], 
    "y": [dfcomp[0]['ec'],
          dfcomp[1]['ec']], 
    "rep_save": repsect1,
    # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
    "label1": [r"$(E_{kin}^{ref})_{inert}$",r"$(E_{kin}^{ref})_{LTE}$"],
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
    "tile1": f"ener pot beam {ichoc} = f(t)" + "\n",
    "tile_save": f"epot_ft",
    "x": [dfcomp[0]['t'],
          dfcomp[1]['t']], 
    "y": [dfcomp[0]['epot'],
          dfcomp[1]['epot']], 
    "rep_save": repsect1,
    # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
    "label1": [r"$(E_{pot})_{inert}$",r"$(E_{pot})_{LTE}$"],
    "labelx": r"$t \quad (s)$",
    "labely": "Energy (J)",
    "color1": color1,
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "loc_leg": "upper right",
}
traj.pltraj2d_list(**kwargs)

kwargs = {"tile1" : "energies lte vs inert coupling",
          "tile_save" : "energies_ft",
          "colx" : ["t"]*5,
          "coly" : ["ec","eintbar","epot","etot","estock"],
          "rep_save" : repsect1,
          "color1" : color1,
          "label1" : [r"$E_{kin}^{ref}$",r"$E_{vibr}$",r"$E_{pot}$",r"$E_{tot}$",r"$E_{stock}$"] ,
          "labelx" : "t (s)" ,
          "labely" : "Energies (J)",
          "linestyle1" : ['-','--'],
          "lslabel" : ["Inertia coupling", "LTE"],
          "title_ls" : "Integration scheme : ",
          "title_col" : "Energies : ",
          "title_both" : None,
          "leg_col" : True,
          "leg_ls" : True,
          "leg_both" : False,
          # "loc_col" : "upper right",
          "loc_col" :  (1.01,0.5),
          # "loc_ls" : "upper center",
          "loc_ls" :   (1.01,0.85),
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp, **kwargs)

kwargs = {"tile1" : "rotation speeds" + "\n",
          "tile_save" : f"Ws_ft",
          "colx" : ["t"]*3,
          "coly" : ["wx","wy","wz"],
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "labely": "Rotation Speed (rad/s)",
          "color1" : ['red','green','blue'],
          "label1" : [r"$W_x$",r"$W_y$",r"$W_z$"] ,
          "labely2": r"$u_y$" + " (m)",
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  False,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : True,
          "loc_col" :  (1.01,0.8),
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : (1.01,0.6),
          "title_col" : "Altitude : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$W_x$",r"$W_y$",r"$W_z$"],
          "colorcust" : ["red","green","blue"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)

kwargs = {"tile1" : "ux pchoc" + "\n",
          "tile_save" : f"uxpchoc_ft",
          "colx" : ["t"],
          "coly" : ["uxpchoc"],
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "labely": r"$u_x$"+" (m)",
          "color1" : ['red'],
          "label1" : [r"$u_x$"] ,
          "labely2": r"$u_y$" + " (m)",
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  (1.01,0.74),
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : (1.01,0.6),
          "title_col" : "Displacement : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$u_x$"],
          "colorcust" : ["red"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)

kwargs = {"tile1" : "uy p1" + "\n",
          "tile_save" : f"uyp1_ft",
          "colx" : "t",
          "coly" : "uypchoc",
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "labely": r"$u_y$"+" (m)",
          "color1" : ['red','green'],
          "label1" :   ["Inertia coupling", "LTE"],
          "labely2": r"$u_y$" + " (m)",
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   False,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  "upper right",
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : "upper right",
          "title_col" : "Scheme : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$u_y(P_1)$"],
          "colorcust" : ["red"],
          "linestyle1" : ['-','-'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.7,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)

kwargs = {"tile1" : "uy p2" + "\n",
          "tile_save" : f"uyp2_ft",
          "colx" : "t",
          "coly" : "uypchoc2",
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "labely": r"$u_y$"+" (m)",
          "color1" : ['red','green'],
          "label1" :   ["Inertia coupling", "LTE"],
          "labely2": r"$u_y$" + " (m)",
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   False,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  "upper right",
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : "upper right",
          "title_col" : "Scheme : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$u_y(P_2)$"],
          "colorcust" : ["red"],
          "linestyle1" : ['-','-'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.7,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)

kwargs = {"tile1" : "uy g" + "\n",
          "tile_save" : f"uyg_ft",
          "colx" : "t",
          "coly" : "uyg",
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "labely": r"$u_y$"+" (m)",
          "color1" : ['red','green'],
          "label1" :   ["Inertia coupling", "LTE"],
          "labely2": r"$u_y$" + " (m)",
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   False,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  "upper right",
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : "upper right",
          "title_col" : "Scheme : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$u_y(P_2)$"],
          "colorcust" : ["red"],
          "linestyle1" : ['-','-'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.7,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)


kwargs = {"tile1" : "uz pchoc" + "\n",
          "tile_save" : f"uypchoc_ft",
          "colx" : ["t"],
          "coly" : ["uypchoc"],
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "labely": r"$u_y$"+" (m)",
          "color1" : ['red'],
          "label1" : [r"$u_y$"] ,
          "labely2": r"$u_y$" + " (m)",
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  (1.01,0.74),
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : (1.01,0.6),
          "title_col" : "Displacement : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$u_y$"],
          "colorcust" : ["red"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)

kwargs = {"tile1" : "energy first 3 modes" + "\n",
          "tile_save" : f"emode3fst_ft",
          "colx" : ["t"]*3,
          "coly" : ["emode1","emode2","emode3"],
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "color1" : ["green","blue","turquoise"],
          "label2" : None,
          "label1" : [r"$E(\varphi_1)$",r"$E(\varphi_2)$",r"$E(\varphi_3)$"],
          "labely" : "Energies (J)",
          "labely2": None,
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  (1.01,0.65),
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : (1.01,0.5),
          "title_col" : "Energies : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Energies : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : None,
          "color2" :    ["green","blue","turquoise"],
          "colorcust" : ["red","green","blue","turquoise"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)

kwargs = {"tile1" : "energy mode x" + "\n",
          "tile_save" : f"emodex_ft",
          "colx" : ["t"]*3,
          "coly" : ["emode2","emode4","emode6"],
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "color1" : ["green","blue","turquoise"],
          "label2" : None,
          "label1" : [r"$E(\varphi_1^x)$",r"$E(\varphi_2^x)$",r"$E(\varphi_3^x)$"],
          "labely" : "Energies (J)",
          "labely2": None,
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  (1.01,0.65),
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : (1.01,0.5),
          "title_col" : "Energies : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Energies : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : None,
          "color2" :    ["green","blue","turquoise"],
          "colorcust" : ["red","green","blue","turquoise"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)

kwargs = {"tile1" : "energy mode y" + "\n",
          "tile_save" : f"emodey_ft",
          "colx" : ["t"]*3,
          "coly" : ["emode1","emode3","emode5"],
          "x2" : "t",
          "y2" : None,
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "color1" : ["green","blue","turquoise"],
          "label2" : None,
          "label1" : [r"$E(\varphi_1^y)$",r"$E(\varphi_2^y)$",r"$E(\varphi_3^y)$"],
          "labely" : "Energies (J)",
          "labely2": None,
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  True,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : False,
          "loc_col" :  (1.01,0.65),
          "loc_ls" :   (1.01,0.85),
          "loc_cust" : (1.01,0.5),
          "title_col" : "Energies : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Energies : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : None,
          "color2" :    ["green","blue","turquoise"],
          "colorcust" : ["red","green","blue","turquoise"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs(dfcomp,**kwargs)
#%%
repsect1 = f"{repsave}chocs/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ichoc in np.arange(nstchoctr): 
  #
  indexchoc    = dfcomp[0][(dfcomp[0]['t']>=(lgrp[0][ichoc][0]-1.e-3)) & (dfcomp[0]['t']<=(lgrp[0][ichoc][1]+1.e-3))].index
  indexchoclin = dfcomp[1][(dfcomp[1]['t']>=(lgrp[1][ichoc][0]-1.e-3)) & (dfcomp[1]['t']<=(lgrp[1][ichoc][1]+1.e-3))].index
  indexchoc2    = dfcomp[0][(dfcomp[0]['t']>=(lgrp2[0][ichoc][0]-1.e-3)) & (dfcomp[0]['t']<=(lgrp2[0][ichoc][1]+1.e-3))].index
  indexchoclin2 = dfcomp[1][(dfcomp[1]['t']>=(lgrp2[1][ichoc][0]-1.e-3)) & (dfcomp[1]['t']<=(lgrp2[1][ichoc][1]+1.e-3))].index
  # indexchoc    = df[(df['t']>=(lgrp[0][ichoc][0]-1.e-3)) & (df['t']<=(lgrp[0][ichoc][1]+1.e-3))].index
  # indexchoclin = df[(df['t']>=(lgrp[1][ichoc][0]-1.e-3)) & (df['t']<=(lgrp[1][ichoc][1]+1.e-3))].index
  #
  kwargs = {
      "tile1": f"traj choc {ichoc} = f(t)" + "\n",
      "tile_save": f"choc{ichoc}_ft",
      "x": [dfcomp[0].iloc[indexchoc]['t'], 
            dfcomp[1].iloc[indexchoclin]['t']], 
      "y": [dfcomp[0].iloc[indexchoc]['uypchoc'], 
            dfcomp[1].iloc[indexchoclin]['uypchoc']], 
      "rep_save": repsect1,
      "label1": [r"$u_z^{inert}$",r"$u_z^{LTE}$"],
      "labelx": r"$t \quad (s)$",
      "labely": r"$u_y$" + " (m)",
      "labelsol": "floor",
      "color1": color1,
      "endpoint": False,
      "xpower": 5,
      "ypower": 5,
      "sol": zsol,
      # "sol": -2.*np.cos(df['thini'][0]*np.pi/180.)*h,
      "loc_leg": "upper right",
  }
  traj.pltraj2d_list_sol(**kwargs)
  #
  # kwargs = {
  #     "tile1": f"ener damping choc {ichoc} = f(t)" + "\n",
  #     "tile_save": f"choc{ichoc}_edamp_ft",
  #     "x": [dfcomp[0].iloc[indexchoc]['t'],
  #           dfcomp[1].iloc[indexchoclin]['t']], 
  #     "y": [dfcomp[0].iloc[indexchoc]['ef'],
  #           dfcomp[1].iloc[indexchoclin]['ef']], 
  #     "rep_save": repsect1,
  #     # "label2": [r"$E_{kin}^{ref}$",r"$E_{bar}$"],
  #     "label1": [r"$E_{damp}^{inert}$",r"$E_{damp}^{LTE}$"],
  #     "labelx": r"$t \quad (s)$",
  #     "labely": "Energy (J)",
  #     "color1": color1,
  #     "endpoint": False,
  #     "xpower": 5,
  #     "ypower": 5,
  #     "loc_leg": "upper left",
  # }
  # traj.pltraj2d_list(**kwargs)

  kwargs = {"tile1" : "altitude and damping energy" + "\n",
            "tile_save" : f"uzedamp_p1_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["ef"],
            "rep_save" : repsect1,
            "color2" : ['blue'],
            "label2" : [r"$E_{f}$"] ,
            "labelx" : "t (s)" ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Eneregy (J)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$u_y(P1)$"] ,
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P1)$",r"$E_{f}$"],
            "color1" :    color1,
            "colorcust" : [color1[0],"blue"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "vibr and ref energy p1" + "\n",
            "tile_save" : f"uzevibref_p1_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["ec","eintbar"],
            "rep_save" : repsect1,
            "color2" : ['blue','green'],
            "label2" : [r"$E_{kin}^{ref}$",r"$E_{vibr}$"] ,
            "labelx" : "t (s)" ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Eneregy (J)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.6),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$u_y(P1)$"] ,
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P1)$",r"$E_{kin}^{ref}$",r"$E_{vibr}$"],
            "color1" :    color1,
            "colorcust" : ['red','blue','green'],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "vibr and ref energy p2" + "\n",
            "tile_save" : f"uzevibref_p2_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc2"],
            "x2" : "t",
            "y2" : ["ec","eintbar"],
            "rep_save" : repsect1,
            "color2" : ['blue','green'],
            "label2" : [r"$E_{kin}^{ref}$",r"$E_{vibr}$"] ,
            "labelx" : "t (s)" ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Eneregy (J)",
            "loc_leg2": "lower right",
            "sol": zsol2,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.6),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$u_y(P2)$"] ,
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P2)$",r"$E_{kin}^{ref}$",r"$E_{vibr}$"],
            "color1" :    color1,
            "colorcust" : ['red','blue','green'],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc2],dfcomp[1].iloc[indexchoc2]],**kwargs)

  kwargs = {"tile1" : "altitude and damping energy" + "\n",
            "tile_save" : f"uzedamp_p2_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc2"],
            "x2" : "t",
            "y2" : ["ef"],
            "rep_save" : repsect1,
            "color2" : ['blue'],
            "label2" : [r"$E_{f}$"] ,
            "labelx" : "t (s)" ,
            "labely": r"$u_y(P2)$" + " (m)",
            "labely2": "Eneregy (J)",
            "loc_leg2": "lower right",
            "sol": zsol2,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$u_y(P2)$"] ,
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P2)$",r"$E_{f}$"],
            "color1" :    color1,
            "colorcust" : [color1[0],"blue"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc2],dfcomp[1].iloc[indexchoc2]],**kwargs)

  kwargs = {"tile1" : "altitude and normal shock force" + "\n",
            "tile_save" : f"uzfn_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["fn"],
            "rep_save" : repsect1,
            "color2" : ['blue'],
            "label2" : [r"$F_n$"] ,
            "labelx" : "t (s)" ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Normal Force (N)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$u_y$"] ,
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$F_n$"],
            "color1" :    color1,
            "colorcust" : [color1[0],"blue"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "normal speed" + "\n",
            "tile_save" : f"vn_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["vn"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['green'],
            "label2" : [r"$v_n$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Normal Speed (m/s)",
            "loc_leg2": "lower right",
            "labely": r"$u_y$" + " (m)",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$v_n$"] ,
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$v_n$"],
            "color1" :    color1,
            "colorcust" : [color1[0],"green"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "CDM speed" + "\n",
            "tile_save" : f"vg_p1_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["vyg"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['green'],
            "label2" : [r"$v_{G}$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Normal Speed (m/s)",
            "loc_leg2": "lower right",
            "labely": r"$u_y$" + " (m)",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$v_{G}$"] ,
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P_1)$",r"$v_{G}$"],
            "color1" :    color1,
            "colorcust" : [color1[0],"green"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "CDM speed" + "\n",
            "tile_save" : f"vg_p2_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc2"],
            "x2" : "t",
            "y2" : ["vyg"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['green'],
            "label2" : [r"$v_{G}$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Normal Speed (m/s)",
            "loc_leg2": "lower right",
            "labely": r"$u_y$" + " (m)",
            "sol": zsol2,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" :    [r"$v_{G}$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P_2)$",r"$v_{G}$"],
            "color1" :    color1,
            "colorcust" : [color1[0],"green"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc2],dfcomp[1].iloc[indexchoc2]],**kwargs)

  kwargs = {"tile1" : "energy mode y" + "\n",
            "tile_save" : f"emodey_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["emode1","emode3","emode5"],
            "label2" : [r"$E(\varphi_1^y)$",r"$E(\varphi_2^y)$",r"$E(\varphi_3^y)$"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ["green","blue","turquoise"],
            "color1" : ['red'],
            "labely2" : "Energies (J)",
            "labely": r"$u_y$" + " (m)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.15,0.85),
            "loc_cust" : (1.15,0.5),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$E(\varphi_1^y)$",r"$E(\varphi_2^y)$",r"$E(\varphi_3^y)$"],
            "colorcust" : ["red","green","blue","turquoise"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "energy mode x" + "\n",
            "tile_save" : f"emodex_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "y2" : ["emode2","emode4","emode6"],
            "x2" : "t",
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color1" : ['red'],
            "label1" : [r"$u_y$"] ,
            "label2" : [r"$E(\varphi_1^x)$",r"$E(\varphi_2^x)$",r"$E(\varphi_3^x)$"],
            "labely2" : "Energies (J)",
            "labely": r"$u_y$" + " (m)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.15,0.85),
            "loc_cust" : (1.15,0.5),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$E(\varphi_1^x)$",r"$E(\varphi_2^x)$",r"$E(\varphi_3^x)$"],
            "color2" :    ["green","blue","turquoise"],
            "colorcust" : ["red","green","blue","turquoise"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "rotation speed x" + "\n",
            "tile_save" : f"Wx_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["wx"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['purple'],
            "label2" : [r"$W_x$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Rotation Speed (rad/s)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$W_x$"],
            "color1" :    ["red"],
            "colorcust" : ["red","purple"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "rotation speed y" + "\n",
            "tile_save" : f"Wy_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["wy"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['purple'],
            "label2" : [r"$W_y$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Rotation Speed (rad/s)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$W_y$"],
            "color1" :    ["red"],
            "colorcust" : ["red","purple"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "rotation speed z" + "\n",
            "tile_save" : f"Wz_p1_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["wz"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['purple'],
            "label2" : [r"$W_z$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Rotation Speed (rad/s)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y(P_1)$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P_1)$",r"$W_z$"],
            "color1" :    ["red"],
            "colorcust" : ["red","purple"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "rotation speed z" + "\n",
            "tile_save" : f"Wz_p2_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc2"],
            "x2" : "t",
            "y2" : ["wz"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['purple'],
            "label2" : [r"$W_z$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Rotation Speed (rad/s)",
            "loc_leg2": "lower right",
            "sol": zsol2,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y(P_2)$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P_2)$",r"$W_z$"],
            "color1" :    ["red"],
            "colorcust" : ["red","purple"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc2],dfcomp[1].iloc[indexchoc2]],**kwargs)

  kwargs = {"tile1" : "energy first 3 modes" + "\n",
            "tile_save" : f"emode3fst_p1_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["emode1","emode2","emode3"],
            "rep_save" : repsect1,
            "labelx" : "t (s)",
            "color1" : ["red"],
            "color2" : ["green","blue","turquoise"],
            "label2" : [r"$E(\varphi_1)$",r"$E(\varphi_2)$",r"$E(\varphi_3)$"],
            "label1" : [r"$u_y(P1)$"],
            "labely" : r"$u_y$" + " (m)",
            "labely2": "Energies (J)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.65),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.4),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Qauantities : ',
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P_1)$",r"$E(\varphi_1)$",r"$E(\varphi_2)$",r"$E(\varphi_3)$"],
            "colorcust" : ["red","green","blue","turquoise"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc],dfcomp[1].iloc[indexchoc]],**kwargs)

  kwargs = {"tile1" : "energy first 3 modes" + "\n",
            "tile_save" : f"emode3fst_p2_choc{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc2"],
            "x2" : "t",
            "y2" : ["emode1","emode2","emode3"],
            "rep_save" : repsect1,
            "labelx" : "t (s)",
            "color1" : ["red"],
            "color2" : ["green","blue","turquoise"],
            "label2" : [r"$E(\varphi_1)$",r"$E(\varphi_2)$",r"$E(\varphi_3)$"],
            "label1" : [r"$u_y(P1)$"],
            "labely" : r"$u_y$" + " (m)",
            "labely2": "Energies (J)",
            "loc_leg2": "lower right",
            "sol": zsol2,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.65),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.4),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Qauantities : ',
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y(P_2)$",r"$E(\varphi_1)$",r"$E(\varphi_2)$",r"$E(\varphi_3)$"],
            "colorcust" : ["red","green","blue","turquoise"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexchoc2],dfcomp[1].iloc[indexchoc2]],**kwargs)
#%%
repsect1 = f"{repsave}vols/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
for ichoc in np.arange(nstchoctr-1): 
  indexvol = dfcomp[0][(dfcomp[0]['t']>=lgrp[0][ichoc][1]) & (dfcomp[0]['t']<=lgrp[0][ichoc+1][0])].index
  indexvollin = dfcomp[1][(dfcomp[1]['t']>=lgrp[1][ichoc][1]) & (dfcomp[1]['t']<=lgrp[1][ichoc+1][0])].index

  indexvol2    = dfcomp[0][(dfcomp[0]['t']>=(lgrp2[0][ichoc][1])) & (dfcomp[0]['t']<=(lgrp2[0][ichoc+1][0]))].index
  indexvollin2 = dfcomp[1][(dfcomp[0]['t']>=(lgrp2[0][ichoc][1])) & (dfcomp[1]['t']<=(lgrp2[1][ichoc+1][0]))].index

  indexinter = indexvol.intersection(indexvol2)

#%%
  kwargs = {"tile1" : "energy mode y" + "\n",
            "tile_save" : f"emodey_vol{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["emode1","emode3","emode5"],
            "label2" : [r"$E(\varphi_1^y)$",r"$E(\varphi_2^y)$",r"$E(\varphi_3^y)$"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ["green","blue","turquoise"],
            "color1" : ['red'],
            "labely2" : "Energies (J)",
            "labely": r"$u_y$" + " (m)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.15,0.85),
            "loc_cust" : (1.15,0.5),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$E(\varphi_1^y)$",r"$E(\varphi_2^y)$",r"$E(\varphi_3^y)$"],
            "colorcust" : ["red","green","blue","turquoise"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexvol],dfcomp[1].iloc[indexvol]],**kwargs)

  kwargs = {"tile1" : "energy mode x" + "\n",
            "tile_save" : f"emodex_vol{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "y2" : ["emode2","emode4","emode6"],
            "x2" : "t",
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color1" : ['red'],
            "label1" : [r"$u_y$"] ,
            "label2" : [r"$E(\varphi_1^x)$",r"$E(\varphi_2^x)$",r"$E(\varphi_3^x)$"],
            "labely2" : "Energies (J)",
            "labely": r"$u_y$" + " (m)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.15,0.85),
            "loc_cust" : (1.15,0.5),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$E(\varphi_1^x)$",r"$E(\varphi_2^x)$",r"$E(\varphi_3^x)$"],
            "color2" :    ["green","blue","turquoise"],
            "colorcust" : ["red","green","blue","turquoise"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexvol],dfcomp[1].iloc[indexvol]],**kwargs)

  kwargs = {"tile1" : "rotation speed x" + "\n",
            "tile_save" : f"Wx_vol{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["wx"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['purple'],
            "label2" : [r"$W_x$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Rotation Speed (rad/s)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$W_x$"],
            "color1" :    ["red"],
            "colorcust" : ["red","purple"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexvol],dfcomp[1].iloc[indexvol]],**kwargs)
  kwargs = {"tile1" : "rotation speed y" + "\n",
            "tile_save" : f"Wy_vol{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["wy"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['purple'],
            "label2" : [r"$W_y$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Rotation Speed (rad/s)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$W_y$"],
            "color1" :    ["red"],
            "colorcust" : ["red","purple"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexvol],dfcomp[1].iloc[indexvol]],**kwargs)
  kwargs = {"tile1" : "rotation speed y" + "\n",
            "tile_save" : f"Wz_vol{ichoc}_ft",
            "colx" : ["t"],
            "coly" : ["uypchoc"],
            "x2" : "t",
            "y2" : ["wz"],
            "rep_save" : repsect1,
            "labelx" : "t (s)" ,
            "color2" : ['purple'],
            "label2" : [r"$W_z$"] ,
            "labely": r"$u_y$" + " (m)",
            "labely2": "Rotation Speed (rad/s)",
            "loc_leg2": "lower right",
            "sol": zsol,
            "labelsol" : None,
            # "loc_ls" : "upper center",
            "leg_col" :  False,
            "leg_ls" :   True,
            "leg_both" : False,
            "cust_leg" : True,
            "loc_col" :  (1.1,0.8),
            "loc_ls" :   (1.1,0.85),
            "loc_cust" : (1.1,0.7),
            "title_col" : "Altitude : ",
            "title_ls" :  "Scheme : ",
            "title_both" : None,
            "title_cust" : 'Quantities : ',
            "label1" : [r"$u_y$"],
            "lslabel" :   ["Inertia coupling", "LTE"],
            "labelcust" : [r"$u_y$",r"$W_z$"],
            "color1" :    ["red"],
            "colorcust" : ["red","purple"],
            "linestyle1" : ['-','--'],
            "lscust" :     ["-","-","-","-"],
            "alpha" : 0.5,
            "endpoint" : False,
           }
  traj.pltraj2d_dfs([dfcomp[0].iloc[indexvol],dfcomp[1].iloc[indexvol]],**kwargs)

#%%
kwargs = {"tile1" : "vg inter" + "\n",
          "tile_save" : f"vg_inter_ft",
          "colx" : ["t"],
          "coly" : ["uypchoc"],
          "x2" : "t",
          "y2" : ["vyg"],
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "color2" : ['green'],
          "label2" : [r"$v_z(G)$"] ,
          "labely": r"$u_y$" + " (m)",
          "labely2": "Speed (m/s)",
          "loc_leg2": "lower right",
          "sol": zsol,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  False,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : True,
          "loc_col" :  (1.1,0.8),
          "loc_ls" :   (1.1,0.85),
          "loc_cust" : (1.1,0.7),
          "title_col" : "Altitude : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "label1" : [r"$u_y$"],
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$u_y(P1)$",r"$v_z(G)$"],
          "color1" :    ["red"],
          "colorcust" : ["red","green"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs([dfcomp[0].iloc[linter[0][0]],dfcomp[1].iloc[linter[0][0]]],**kwargs)

kwargs = {"tile1" : "wz inter" + "\n",
          "tile_save" : f"wz_inter_ft",
          "colx" : ["t"],
          "coly" : ["uypchoc"],
          "x2" : "t",
          "y2" : ["wz"],
          "rep_save" : repsect1,
          "labelx" : "t (s)",
          "color2" : ['purple'],
          "label2" : [r"$W_z$"],
          "labely": r"$u_y$" + " (m)",
          "labely2": "Rotation Speed (rad/s)",
          "loc_leg2": "lower right",
          "sol": zsol,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  False,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : True,
          "loc_col" :  (1.1,0.8),
          "loc_ls" :   (1.1,0.85),
          "loc_cust" : (1.1,0.7),
          "title_col" : "Altitude : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Quantities : ',
          "label1" : [r"$u_y$"],
          "lslabel" :   ["Inertia coupling","LTE"],
          "labelcust" : [r"$u_y(P1)$",r"$W_z$"],
          "color1" :    ["red"],
          "colorcust" : ["red","purple"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs([dfcomp[0].iloc[linter[0][0]],dfcomp[1].iloc[linter[0][0]]],**kwargs)

kwargs = {"tile1" : "energy first 3 modes" + "\n",
          "tile_save" : f"emode3fst_volinter_ft",
          "colx" : ["t"],
          "coly" : ["uypchoc"],
          "x2" : "t",
          "y2" : ["emode1","emode2","emode3"],
          "rep_save" : repsect1,
          "labelx" : "t (s)" ,
          "color1" : ["red"],
          "color2" : ["green","blue","turquoise"],
          "label2" : [r"$E(\varphi_1)$",r"$E(\varphi_2)$",r"$E(\varphi_3)$"],
          "label1" : [r"$u_y(P1)$"],
          "labely" : r"$u_y$" + " (m)",
          "labely2": "Energies (J)",
          "loc_leg2": "lower right",
          "sol": None,
          "labelsol" : None,
          # "loc_ls" : "upper center",
          "leg_col" :  False,
          "leg_ls" :   True,
          "leg_both" : False,
          "cust_leg" : True,
          "loc_col" :  (1.1,0.65),
          "loc_ls" :   (1.1,0.85),
          "loc_cust" : (1.1,0.55),
          "title_col" : "Altitude : ",
          "title_ls" :  "Scheme : ",
          "title_both" : None,
          "title_cust" : 'Qauantities : ',
          "lslabel" :   ["Inertia coupling", "LTE"],
          "labelcust" : [r"$u_y$",r"$E(\varphi_1)$",r"$E(\varphi_2)$",r"$E(\varphi_3)$"],
          "color2" :    ["green","blue","turquoise"],
          "colorcust" : ["red","green","blue","turquoise"],
          "linestyle1" : ['-','--'],
          "lscust" :     ["-","-","-","-"],
          "alpha" : 0.5,
          "endpoint" : False,
         }
traj.pltraj2d_dfs([dfcomp[0].iloc[linter[0][0]],dfcomp[1].iloc[linter[0][0]]],**kwargs)
#%%
sys.exit()

#%%