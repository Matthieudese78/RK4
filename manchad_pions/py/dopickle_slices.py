#%%
import os
# from csv_to_pickle import csv2pickle
import numpy as np
import pandas as pd
import glob
import shutil
#%% repertoire data
  # on discretise ?
ndiscr = 1
linert = True
lamode = True
lkxp = False
Fext = 193.
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-4
spinini = 0.
dte = 1.e-6
vlostr = int(-np.log10(vlimoden))
dtstr = int(-np.log10(dte))
xistr = int(100.*xi)
namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}'

amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))
if lamode:
  namerep = f'{namerep}_amodem_{amodemstr}_amodead_{amodeadstr}'

if (lkxp):
  namerep = f'{namerep}_kxp'
if (linert):
  namerep = f'{namerep}_inert'

repload = f'../{namerep}/pickle/'
repsave = f'./pickle/{namerep}/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
#%% concatenating slices :
all_files = glob.glob(repload + "*.pickle")
dfs = []
for i,filename in enumerate(all_files):
    print(f"i = {i}")
    print(f"file = {filename}")
    df = pd.read_pickle(filename)
    # si ce n'est pas la 1ere slice on enleve la 1ere ligne
    if (i != 0): 
      df.sort_values(by='t',inplace=True)
      df.reset_index(drop=True,inplace=True)
      # df.drop(0,inplace=True)
      if (ndiscr>1):
        rows2keep = df.index % ndiscr == 0 
        df = df[rows2keep]
        df.reset_index(drop=True,inplace=True)
    dfs.append(df)
    if i % 50:
      combined_df = pd.concat(dfs,ignore_index=True)
      dfs = [combined_df]

combined_df = pd.concat(dfs,ignore_index=True)
combined_df.sort_values(by='t',inplace=True)
combined_df.reset_index(drop=True,inplace=True)

#%%
combined_df.to_pickle(f"{repsave}result.pickle")

#%% moving the slice 0 to result_0.pickle :
shutil.copy(f"{repload}slice_0/result.pickle",f"{repsave}result_0.pickle")
# shutil.move(f"{repload}slice_0/result.pickle",f"{repsave}result_0.pickle")

# %%
