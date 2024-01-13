#%%
import os
from csv_to_pickle import csvs2pickle

#%%
#%% rep_load
manchette = False
stoia = True
if (manchette):
  script = "manchette"
if (stoia):
  script = "stoia"

repload = f'../data/{script}/'
repsave = f'./pickle/{script}/'

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
