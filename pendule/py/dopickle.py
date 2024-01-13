#%%
import os
from csv_to_pickle import csvs2pickle

#%%
stoia = True
manchette = False
limpact = True
#%%
#%% rep_load
if (limpact):
    if (stoia):
        repload = f'../data/impact/stoia/'
        repsave = f'./pickle/impact/stoia/'
    if (manchette):
        repload = f'../data/impact/manchette/'
        repsave = f'./pickle/impact/manchette/'

if (not limpact):
    if (stoia):
        repload = f'../data/no_impact/stoia/'
        repsave = f'./pickle/no_impact/stoia/'
    if (manchette):
        repload = f'../data/no_impact/manchette/'
        repsave = f'./pickle/no_impact/manchette/'

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
