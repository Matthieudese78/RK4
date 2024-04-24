#%%
import os
from csv_to_pickle import csvs2pickle

#%%
stoia = True
manchette = False
limpact = True
linert = True
lnortot = False
lnorcomp = True
lnormtot = False
xi = 0.
thini = 45.
nmode = 6
#%%
repload = '../data/'
repsave = './pickle/'
#%% rep_load
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
# kwargs1 = {'rep_load' : f"{repload}{script1}", \
kwargs1 = {'rep_load' : f"{repload}", \
           'rep_save' : f"{repsave}"}
csvs2pickle(**kwargs1)

# %%
