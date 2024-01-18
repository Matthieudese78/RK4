
#!/bin/python3
#%%
import numpy as np
# import numpy.linalg as LA
import pandas as pd
# import trajectories as traj
# import rotation as rota
# import repchange as rc
import os

# #%% usefull parameters :
# color1 = ["red", "green", "blue", "orange", "purple", "pink"]
# view = [20, -50]
# # %% quel type de modele ?
# lraidtimo = False
# lconefixe = True
# %% Scripts :
    # which slice ?
slice = 1
script1 = f"slice_{slice}"
repload = f"./pickle/fnonly/{script1}/"
    # PDT : 
pdt = 50.e-6
# %%
rep_save = f"./fig/{script1}/"

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")

#%% contact time interval :
  # pion bas 1 :
df['tag'] = df['FN_pb1'] < 0
fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
prb1 = [(i,j) for i,j in zip(fst,lst)]
tchocpb1 = [ pdt*(j-i+1) for i,j in zip(fst,lst) ]
meantchoc_pb1 = np.mean(tchocpb1)

  # pion bas 2 :
df['tag'] = df['FN_pb2'] < 0
fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
prb2 = [(i,j) for i,j in zip(fst,lst)]
tchocpb2 = [ pdt*(j-i+1) for i,j in zip(fst,lst) ]
meantchoc_pb2 = np.mean(tchocpb2)

  # pion bas 3 :
df['tag'] = df['FN_pb3'] < 0
fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
prb3 = [(i,j) for i,j in zip(fst,lst)]
tchocpb3 = [ pdt*(j-i+1) for i,j in zip(fst,lst) ]
meantchoc_pb3 = np.mean(tchocpb3)

  # ccone :
df['tag'] = df['FN_CCONE'] > 0
fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
prccone = [(i,j) for i,j in zip(fst,lst)]
tchocccone = [ pdt*(j-i+1) for i,j in zip(fst,lst) ]
meantchoc_ccone = np.mean(tchocccone)

print(f"mean tchoc pb1 = {meantchoc_pb1}")
print(f"mean tchoc pb2 = {meantchoc_pb2}")
print(f"mean tchoc pb3 = {meantchoc_pb3}")
print(f"mean tchoc ccone = {meantchoc_ccone}")
# %%
