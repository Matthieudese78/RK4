#%%
import os
import numpy as np
from csv_to_pickle import csvs2pickle

#%%
stoia = True
manchette = False
trig = True
limpact = True
linert = True
lnortot = False
lnorcomp = True
lnormtot = False
if stoia:
  M = 0.59868
bamo = 2.e7
Kchoc = 5.5e07
# xi = 0.
xi = bamo / (2.0 * M * (np.sqrt(Kchoc / M)))
thinc = 10.
nmode = 26
#%%
repload = '../data/'
# repload = '../calc/data/'
repsave = './pickle/'
#%% rep_load
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

repload = f'{repload}xi_{int(100.*xi)}/thinc_{int(thinc)}/nmode_{nmode}/'
repsave = f'{repsave}xi_{int(100.*xi)}/thinc_{int(thinc)}/nmode_{nmode}/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
else:
    print(f"FOLDER : {repsave} already exists.")

#%%
# kwargs1 = {'rep_load' : f"{repload}{script1}", \
kwargs1 = {'rep_load' : f"{repload}", \
           'rep_save' : f"{repsave}"}
csvs2pickle(**kwargs1)

# %%
