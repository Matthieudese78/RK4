#%%
import os
from csv_to_pickle import csvs2pickle

#%%
script = f"fast_top"
# script = f"slow_top"
# script = f"fsb"
# script = f"bt"
# script = f"cb"
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
csvs2pickle(**kwargs1)

# %%
