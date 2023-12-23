#%%
import os
from csv_to_pickle import csv2pickle

#%%
script = f"cb"
#%% rep_load
repload = f'../data_{script}/'
repsave = f'./pickle/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
else:
    print(f"FOLDER : {repsave} already exists.")

#%%
# kwargs1 = {'rep_load' : f"{repload}{script1}", \
kwargs1 = {'rep_load' : f"{repload}", \
           'rep_save' : f"{repsave}{script}/"}
csv2pickle(**kwargs1)

# %%
