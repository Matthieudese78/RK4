#%%
import os
from csv_to_pickle import csv2pickle

#%%
fnonly = False
# script1 = 'manchadela_weight' 
slice = 1 
script1 = f'manchadela_pions_{slice}' 
# script1 = 'manchadela_pion' 
# script1 = 'manchadela_RSG' 

# t = 2 
# f = 8
# F = 20
#%% rep_load
repload = '../data/'
repsave = f'./pickle/'

if (fnonly):
    script1 = f"slice_{slice}"
    repload = f"../data/fnonly/"
    repsave = f"./pickle/fnonly/"

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
else:
    print(f"FOLDER : {repsave} already exists.")

#%%
# kwargs1 = {'rep_load' : f"{repload}{script1}", \
kwargs1 = {'rep_load' : f"{repload}", \
           'rep_save' : f"{repsave}{script1}/"}
csv2pickle(**kwargs1)

# %%
