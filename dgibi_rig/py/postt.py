#!/bin/python3
#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import matplotlib.pyplot as plt
import os

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [20, -50]

#%% 1 ou plsrs calculs ? 
lindiv = False
#%% Cas test ? :
icas1 = 3
if (icas1==2):
    n1 = 2
    n2 = 3
    n3 = 4

if (icas1==3):
    n1 = 7
    n2 = 7
    n3 = 7
#%% point d'observation
# pobs = np.array([0.2,0.2,0.2])
pobs = np.array([1.,1.,1.])
# %% quel type de modele ?
lraidtimo = False
# %% Scripts :
lscript = [f"fast_top",f"slow_top",f"fsb",f"bt",f"cb"]
lalgo = ['sw','nmb','rkmk4']
script1 = lscript[icas1]
repload = f"./pickle/{script1}/"
# %% 
rep_save = f"./fig/{script1}/"

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")

# %% matrice de rotation de la manchette :
kq = {"colname": "mrot", "q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4"}
rota.q2mdf(df, **kq)

# %% trajectoire du point d'observation :
kp = {"mat": "mrot", "point": pobs, "colnames": ["uxpobs","uypobs","uzpobs"]}
rota.recopointdf(df, **kp)
# %%          COLORATION : 
    # en fonction du pdt :
kcol = {'colx' : 'n', 'ampl' : 200., 'logcol' : False}
dfcolpus = traj.color_from_value(df,**kcol)

#%% pour un algo : 
lindcas1 = [ [ df[(df['ialgo']==ialgi) & (df['icas']==(icas1+1)) & (df['n']==nj) ].index for i,ialgi in enumerate([1,2,3]) ] for j,nj in enumerate([n1,n2,n3]) ]

#%%############################################
#           PLOTS : grandeures temporelles :
###############################################
repsect1 = f"{rep_save}variables_ft/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

# vitesses de rotations body frame :
kwargs1 = {
    "tile1": "Wx = f(t)" + "\n",
    "tile_save": "Wx_t",
    "colx": "t",
    "coly": "wx",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_X \quad (rad/s)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Wy = f(t)" + "\n",
    "tile_save": "Wy_t",
    "colx": "t",
    "coly": "wy",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_Y \quad (rad/s)$",
    "color1": color1[1],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Wz = f(t)" + "\n",
    "tile_save": "Wz_t",
    "colx": "t",
    "coly": "wz",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_Z \quad (rad/s)$",
    "color1": color1[2],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Ecin = f(t)" + "\n",
    "tile_save": "Ecin_t",
    "colx": "t",
    "coly": "ec",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{kin} \quad (J)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Emag = f(t)" + "\n",
    "tile_save": "Emag_t",
    "colx": "t",
    "coly": "edef",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{mag} \quad (J)$",
    "color1": color1[1],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Etot = f(t)" + "\n",
    "tile_save": "Etot_t",
    "colx": "t",
    "coly": "et",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E \quad (J)$",
    "color1": color1[2],
}
traj.pltraj2d(df, **kwargs1)

#%%############################################
#           PLOTS : trajectoires 3d :
###############################################
repsect1 = f"{rep_save}traj_3d_pdts/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

ialgo = 0
kwargs1 = {
    "tile1": f"traj. 3d cas {lscript[icas1]} ialgo = {lalgo[ialgo]}" + "\n",
    "tile_save": f"traj3d_pdts_{lscript[icas1]}_{lalgo[ialgo]}",
    "ind": lindcas1[ialgo],
    "colx": "uxpobs",
    "coly": "uypobs",
    "colz": "uzpobs",
    "rep_save": repsect1,
    "label1": [fr"$h = 1/2^{n1}$", fr"$h = 1/2^{n2}$", fr"$h = 1/2^{n3}$"],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1,
    "view": view,
}
traj.pltraj3d_ind(df, **kwargs1)
#%%
ialgo = 1
kwargs1 = {
    "tile1": f"traj. 3d cas {lscript[icas1]} ialgo = {lalgo[ialgo]}" + "\n",
    "tile_save": f"traj3d_pdts_{lalgo[ialgo]}",
    "ind": lindcas1[ialgo],
    "colx": "uxpobs",
    "coly": "uypobs",
    "colz": "uzpobs",
    "rep_save": repsect1,
    "label1": [fr"$h = 1/2^{n1}$", fr"$h = 1/2^{n2}$", fr"$h = 1/2^{n3}$"],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1,
    "view": view,
}
traj.pltraj3d_ind(df, **kwargs1)

ialgo = 2
kwargs1 = {
    "tile1": f"traj. 3d cas {lscript[icas1]} ialgo = {lalgo[ialgo]}" + "\n",
    "tile_save": f"traj3d_pdts_{lalgo[ialgo]}",
    "ind": lindcas1[ialgo],
    "colx": "uxpobs",
    "coly": "uypobs",
    "colz": "uzpobs",
    "rep_save": repsect1,
    "label1": [fr"$h = 1/2^{n1}$", fr"$h = 1/2^{n2}$", fr"$h = 1/2^{n3}$"],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1,
    "view": view,
}
traj.pltraj3d_ind(df, **kwargs1)
#%%############################################
#           PLOTS : trajectoires relatives 3d :
###############################################
if (lindiv):
    repsect1 = f"{rep_save}traj_3d_indiv/"
    if not os.path.exists(repsect1):
        os.makedirs(repsect1)
        print(f"FOLDER : {repsect1} created.")
    else:
        print(f"FOLDER : {repsect1} already exists.")
    #
    kwargs1 = {
        "tile1": "traj. pobs" + "\n",
        "tile_save": "traj3d_pobs",
        "colx": "uxpobs",
        "coly": "uypobs",
        "colz": "uzpobs",
        "rep_save": repsect1,
        "label1": r"$P_{obs}$",
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "labelz": r"$Z \quad (m)$",
        "color1": color1[1],
        "view": view,
    }
    traj.pltraj3d(df, **kwargs1)

# %%
