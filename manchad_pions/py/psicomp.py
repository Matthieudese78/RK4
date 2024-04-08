#!/bin/python3
#%%
import numpy as np
# import numpy.linalg as LA
import pandas as pd
import trajectories as traj
# import rotation as rota
import repchange as rc
import os
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as pltcolors
import rotation as rota
# import mplcursors
# import sys
#%%
rep_save = f"./fig/psicomp/"
color1 = ["red", "blue", "green", "orange", "purple", "pink"]
view = [20, -50]
  # 1 : xp, 2 : computation : 
alpha = [0.7,1.,1.]
#%%
f1 = 2.
f2 = 20.
exb = np.array([0.0, -1.0, 0.0])
eyb = np.array([0.0, 0.0, 1.0])
ezb = np.array([-1.0, 0.0, 0.0])
base2 = [exb, eyb, ezb]
#%%
linert = False
lraidtimo = False
raidiss = True
lamode = True
#%%
    # pour b_lam = 5.5
# Fext = 79.44
    # pour b_lam = 6.5
# Fext = 0.72*2.*79.44
    # pour b_lam = 7.5
# Fext = 0.83*2.*79.44
    # pour b_lam = 8
# Fext = 2.*79.44
    # pour b_lam = 9
# Fext = 2.*79.44
# Fext = [79.44]
# Fext = [2.*79.44]
Fext = [79.44,2.*79.44]
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 5.e-6

h_lam = 50.e-3
b_lam = [5.5e-3,9.e-3]
# b_lam = [9.e-3]
# b_lam = [5.5e-3]
lspring = 45.e-2

vlostr = int(-np.log10(vlimoden))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
hlstr = int(h_lam*1.e3)
lspringstr = int(lspring*1.e2)
amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))
dfs = []
i=0
for bl,F0 in zip(b_lam,Fext):
  blstr = int(bl*1.e3)
  namecase = f'calc_fext_{int(F0)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_bl_{blstr}_lspr_{lspringstr}'
  if lamode:
    namecase = f'{namecase}_amodem_{amodemstr}_amodead_{amodeadstr}'
  if (linert):
    namecase = f'{namecase}_inert'
  if (lraidtimo):
    namecase = f'{namecase}_raidtimo'
  if (raidiss):
    namecase = f'{namecase}_raidiss'
  repload = f'/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/{namecase}/'
  df = pd.read_pickle(f"{repload}2048/result.pickle")
    # on trie et on reindexe :
  df.sort_values(by='t',inplace=True)
  df.reset_index(drop=True,inplace=True)

  # discr = 1.e-3
  # dtsort = df.iloc[1]['t'] - df.iloc[0]['t']
  # print(f"dtsort = {dtsort}")
  # ndiscr = int(discr/(dtsort))
  # print(f"ndiscr = {ndiscr}")
  # df = df.iloc[::ndiscr]
  # df.reset_index(drop=True,inplace=True)

  ttot = df.iloc[-1]['t']
  # matrice de roation :
  kq = {"colname": "mrot", "q1": "quat1", "q2": "quat2", "q3": "quat3", "q4": "quat4"}
  rota.q2mdf(df, **kq)
  # extraction du spin :
      # matrice de rotation de la manchette dans le repere utilisateur :
  name_cols = ["mrotu"]
  kwargs1 = {"base2": base2, "mat": "mrot", "name_cols": name_cols}
  rc.repchgdf_mat(df, **kwargs1)
      # extraction du spin : 
  name_cols = [f"spin{i}"] 
  kwargs1 = {"mat": "mrotu", "colnames": name_cols}
  rota.spinextrdf(df,**kwargs1)
  df[f'spin{i}'] = df[f'spin{i}'] * 180. / np.pi
      # on drop la column "mrotu" qui prend de la place : 
  df.drop(['mrotu'],inplace=True,axis=1)
  df[f'freq{i}'] = f1 + ((f2-f1)/ttot)*df['t'] 
  # print(f"fmin = {df.iloc[0][f'freq{i}']}")
  # print(f"fmax = {df.iloc[-1][f'freq{i}']}")
  # on append dfs :
  df = df[[f't',f'freq{i}',f'spin{i}']]
  dfs.append(df)
  del df
  i += 1

#%%
# df = pd.concat(dfs,axis=1)
df = pd.merge(dfs[0],dfs[1],on='t',how='outer')
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
del dfs
#%% on rajoute la courbe xp
lcasneuf = [
 '20201127_1350',
 '20201127_1406',
 '20201127_1409',
 '20201127_1413',
 '20201127_1416',
 '20201127_1419',
 '20201127_1422',
 '20201127_1427',
 '20201127_1429',
 '20201127_1436',
 '20201127_1448',
 '20201127_1454']
# O1 : pion en face du pot :
icas = 0

O10 = 84. * np.pi/180.
O1 = O10
if icas < 1 :
    O1=O10
elif icas < 7 :
    # O1=O10
    O1=O10-180
elif icas < 14 :
    O1=O10

essai = lcasneuf[icas]

repload = '/home/matthieu/Documents/EDF/mesures/data/donneesLaser/'
filename = f'{repload}{essai}_laser.pickle'
dfxp = pd.read_pickle(f"{filename}")
dfxp = pd.DataFrame(dfxp)
dfxp.sort_values(by='tL',inplace=True)
dfxp.reset_index(drop=True,inplace=True)
def unwrap(signal,ecart) :
    buffer=np.zeros(1000)+signal[0]
    vafter = np.zeros(len(signal))
    for i in range(len(signal)) :
        e=signal[i]
        m=np.mean(buffer)
        if abs(e-m)<ecart :
            vafter[i]=e
        elif abs(e+18-m)<ecart :
            vafter[i]=e+18
        elif abs(e-18-m)<ecart :
            vafter[i]=e-18
        else :
            vafter[i]=m
        if i>1000 :
            buffer=vafter[i-1000:i]
    return vafter

dfxp['L4'] = unwrap(dfxp['L4'],1)
# dfxp['t'] = dfxp['tL']
dfxp['spinxp'] = dfxp['L4']*10. * np.pi/180. - O1 
dfxp['spinxp'] = dfxp['spinxp'][0] - (dfxp['spinxp'] - dfxp['spinxp'][0])
dfxp['spinxp'] = dfxp['spinxp'] * 180. / np.pi
  # freq :
ttot = dfxp.iloc[-1]['tL']
dfxp['freqxp'] = f1 + ((f2-f1)/ttot)*dfxp['tL'] 
dfxp = dfxp[['tL','freqxp','spinxp']]
  # on concat avec le num :
df = pd.concat([df,dfxp],axis=1)
del dfxp
#%%
repsect1 = f"{rep_save}"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

kwargs1 = {
    "tile1": "spin = f(f)" + "\n",
    "tile_save": "spin_f_comp12",
    "colx": ["freqxp","freq0","freq1"],
    "coly": ["spinxp","spin0","spin1"],
    # "colx": ["freq0"],
    # "coly": ["spin0"],
    "rep_save": repsect1,
    "label1": ["Experiment","Model 1","Model 2"],
    # "labelx": r"$t \quad (s)$",
    "labelx": "Loading Frequency" + " (Hz)",
    # "labely": "x-axis rotation "+r"$(\degree)$",
    "labely": r"$\psi \quad (\degree)$",
    "color1": color1,
    "endpoint": [False,False,False],
    "xpower": 5,
    "ypower": 5,
    "alpha" : alpha,
    "loc_leg" : "lower left",
}
traj.pltraj2d(df, **kwargs1)

# kwargs1 = {
#     "tile1": "Spin = f(f)" + "\n",
#     "tile_save": "psi_f",
#     "colx": "freq",
#     "coly": "spindeg",
#     "rep_save": repsect1,
#     "label1": r"$\psi$",
#     # "labelx": r"$t \quad (s)$",
#     "labelx": "Loading Frequency" + " (Hz)",
#     "labely": r"$\psi \quad (\degree)$",
#     "color1": color1[2],
#     "endpoint": False,
#     "xpower": 5,
#     "ypower": 5,
# }
# traj.pltraj2d(df, **kwargs1)
# %%
