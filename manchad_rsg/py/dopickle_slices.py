#!/bin/python3
#%%
import os
# from csv_to_pickle import csv2pickle
import numpy as np
import pandas as pd
import glob
import shutil
import matplotlib.pyplot as plt
#%% repertoire data
  # on discretise ?
ndiscr = 1
l2048 = False
linert = False
lraidtimo = False
raidiss = True
lamode = False
lkxp = False
# Fext = 387.
# Fext = 193.
# Fext = 35.
  # pour b_lam = 6.5
# Fext = 0.72*2.*79.44
  # pour b_lam = 7.5
# Fext = 0.83*2.*79.44
Fext = 0.
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 1.e-6
  # real dte : on prend le multiple de 2 superieur :
# ndte = int((np.log(1./dte))/(np.log(2.))) + 1
# dte = (1./(2.**ndte)) 

h_lam = 50.e-3
b_lam = 9.e-3
# b_lam = 9.e-3
lspring = 45.e-2

vlostr = int(-np.log10(vlimoden))
# dtstr = int(-np.log10(dte))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
hlstr = int(h_lam*1.e3)
blstr = int(b_lam*1.e3)
lspringstr = int(lspring*1.e2)
namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_bl_{blstr}_lspr_{lspringstr}'

amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))
if lamode:
  namerep = f'{namerep}_amodem_{amodemstr}_amodead_{amodeadstr}'

if (lkxp):
  namerep = f'{namerep}_kxp'
if (linert):
  namerep = f'{namerep}_inert'
if (lraidtimo):
  namerep = f'{namerep}_raidtimo'
if (raidiss):
  namerep = f'{namerep}_raidiss'

repload = f'../{namerep}/pickle/'
repsave = f'./pickle/{namerep}/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")

if l2048:
  if not os.path.exists(f"{repsave}2048/"):
    os.makedirs(f"{repsave}2048/")
    print(f"FOLDER : {repsave} created.")
#%% concatenating slices :
all_files = glob.glob(repload + "*.pickle")

#%%
dfs = []
# i1 = int(len(all_files)/2.)
# all_files_1 = all_files[:i1]
# all_files_2 = all_files[i1:-1]
nsplit = 1

listglob = np.split(np.array(all_files),nsplit)

#%%
for il,lfiles in enumerate(listglob): 
  for i,filename in enumerate(lfiles):
      print(f"i = {i}")
      print(f"file = {filename}")
      df = pd.read_pickle(filename)
      # si ce n'est pas la 1ere slice on enleve la 1ere ligne
      # if (i != 0): 
      df.sort_values(by='t',inplace=True)
      df.reset_index(drop=True,inplace=True)
      # df.drop(0,inplace=True)
      if l2048:
        dt = df['t'].iloc[1] - df['t'].iloc[0] 
        dtobj = 1./2048.
        ndiscr2048 = int(dtobj/dt)
        print(f"ndiscr2048 = {ndiscr2048}")
        df = df.iloc[::ndiscr2048]
      if (ndiscr>1):
        # rows2keep = df.index % ndiscr == 0 
        # df = df[rows2keep]
        df = df.iloc[::ndiscr]
        df.reset_index(drop=True,inplace=True)
      dfs.append(df)
      # if i % 50:
      #   combined_df = pd.concat(dfs,ignore_index=True)
      #   dfs = [combined_df]

  combined_df = pd.concat(dfs,ignore_index=True)
  combined_df.sort_values(by='t',inplace=True)
  combined_df.reset_index(drop=True,inplace=True)
  combined_df.to_pickle(f"{repsave}result{il}.pickle")
  del combined_df
  del dfs
  dfs = []
#%%
all_pickles = glob.glob(repsave + "*.pickle")
dfs = []
for il,fpickle in enumerate(all_pickles):
  dfs.append((pd.read_pickle(fpickle)))
  os.remove(fpickle)

dftot = pd.concat(dfs,ignore_index=True)
del dfs
dftot.sort_values(by='t',inplace=True)
dftot.reset_index(drop=True,inplace=True)
dtfinal = dftot['t'][1] - dftot['t'][0] 
print(f"time step saved data = dtfinal = {dtfinal}")
if l2048:
  dftot.to_pickle(f"{repsave}2048/result.pickle")
else:
  dftot.to_pickle(f"{repsave}result.pickle")

# del dftot

# #%% moving the slice 0 to result_0.pickle :
# shutil.copy(f"{repload}slice_0/result.pickle",f"{repsave}result_0.pickle")
# # shutil.move(f"{repload}slice_0/result.pickle",f"{repsave}result_0.pickle")

# %%
