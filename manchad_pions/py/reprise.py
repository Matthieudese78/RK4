#!/bin/python3 
  # goal : Creation d'un tableau de reprise
#%%
import numpy as np
import pandas as pd
import os
import json
import rotation
#%% lecture du dataframe  :
  # slice num ? 
slice = 2 
scriptload = f"manchadela_pion_{slice - 1}"
repload = f"./pickle/{scriptload}/"

scriptsave = f"reprise_{slice}"
rep_save = f"./reprise/{scriptsave}/"
# %%
if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")
# %% trnasfo quaternion to vecteur de rotation : 
q = np.array([df['q1'].tail(),df['q2'].tail(),df['q3'].tail(),df['q4'].tail()])
vect = rotation.quat2vect(q)

# %% 
  # manchette & adaptateur : 4 modes elastiques. 
dict_rep = {
  #          ddls ela : manchette
             'qf1' : df['qf1'].tail(),
             'qf2' : df['qf2'].tail(),
             'qf3' : df['qf3'].tail(),
             'qf4' : df['qf4'].tail(),
             'vf1' : df['vf1'].tail(),
             'vf2' : df['vf2'].tail(),
             'vf3' : df['vf3'].tail(),
             'vf4' : df['vf4'].tail(),
  #          ddls rig : manchette vect de rotation 
             'vectx' : vect[0],
             'vecty' : vect[1],
             'vectz' : vect[2],
  #          ddls rig : vitesse de rotation 
             'wx' : df['WX'].tail(),
             'wy' : df['WY'].tail(),
             'wz' : df['WZ'].tail(),
  #          ddls ela : adapter 
             'q1ad' : df['q1_ad'].tail(),
             'q2ad' : df['q2_ad'].tail(),
             'q3ad' : df['q3_ad'].tail(),
             'q4ad' : df['q4_ad'].tail(),
             'v1ad' : df['v1_ad'].tail(),
             'v2ad' : df['v2_ad'].tail(),
             'v3ad' : df['v3_ad'].tail(),
             'v4ad' : df['v4_ad'].tail(),
  #          frequence de reprise  
            }
# %%
with open(f'reprise_{slice}.csv', 'w') as filehandle:
    json.dump(dict_rep, filehandle)