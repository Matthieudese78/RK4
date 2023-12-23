#!/bin/python3
#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import os

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [20, -50]
#%% point d'observation
# pobs = np.array([0.2,0.2,0.2])
pobs = np.array([1.,1.,1.])
# %% quel type de modele ?
lraidtimo = False
# %% Scripts :
script1 = f"cb"
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
#           PLOTS : trajectoires relatives 2d:
###############################################
# repsect1 = f"{rep_save}traj_relatives_2d/"
# if not os.path.exists(repsect1):
#     os.makedirs(repsect1)
#     print(f"FOLDER : {repsect1} created.")
# else:
#     print(f"FOLDER : {repsect1} already exists.")

# # centre du cercle - vis a vis adapter : 
# kwargs1 = {
#     "tile1": "traj. relative PCcirc / PCad" + "\n",
#     "tile_save": "traj2d_Ccirc",
#     "colx": "uxcerela",
#     "coly": "uycerela",
#     "rep_save": repsect1,
#     "label1": r"$C_{circ}$",
#     "labelx": r"$X \quad (m)$",
#     "labely": r"$Y \quad (m)$",
#     "color1": 'black',
#     "view": view,
# }
# traj.pltraj2d(df, **kwargs1)


#%%############################################
#           PLOTS : trajectoires relatives 3d :
###############################################
repsect1 = f"{rep_save}traj_3d/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#
# kwargs1 = {
#     "tile1": "traj. relative p2" + "\n",
#     "tile_save": "traj3d_p2",
#     "colx": "uxp2",
#     "coly": "uyp2",
#     "colz": "uzp2",
#     "rep_save": repsect1,
#     "label1": r"$P_{2}$",
#     "labelx": r"$X \quad (m)$",
#     "labely": r"$Y \quad (m)$",
#     "labelz": r"$Z \quad (m)$",
#     "color1": color1[1],
#     "view": view,
# }
# traj.pltraj3d(df, **kwargs1)

kwargs1 = {
    "tile1": "traj. relative pobs" + "\n",
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

#%%############################################
#           PLOTS : forces de chocs :
###############################################
repsect1 = f"{rep_save}forces_de_choc/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
# pb 1 : 
kwargs1 = {
    "tile1": "FN pb1 = f(t)" + "\n",
    "tile_save": "fn_pb1",
    "colx": "t",
    "coly": "FN_pb1",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{n}^l \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

# pb 2 : 
kwargs1 = {
    "tile1": "FN pb2 = f(t)" + "\n",
    "tile_save": "fn_pb1",
    "colx": "t",
    "coly": "FN_pb2",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{n}^l \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

# pb 3 : 

kwargs1 = {
    "tile1": "FN pb3 = f(t)" + "\n",
    "tile_save": "fn_pb3",
    "colx": "t",
    "coly": "FN_pb3",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{n}^l \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

# ph 1 : 
kwargs1 = {
    "tile1": "FN ph1 = f(t)" + "\n",
    "tile_save": "fn_ph1",
    "colx": "t",
    "coly": "FN_ph1",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{n}^l \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

# ph 2 : 
kwargs1 = {
    "tile1": "FN ph2 = f(t)" + "\n",
    "tile_save": "fn_ph2",
    "colx": "t",
    "coly": "FN_ph2",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{n}^l \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

# ph 3 : 
kwargs1 = {
    "tile1": "FN ph3 = f(t)" + "\n",
    "tile_save": "fn_ph3",
    "colx": "t",
    "coly": "FN_ph3",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$P_{W}^l \quad (W)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

#%%############################################
#           PLOTS : Puissance d'usure :
###############################################
repsect1 = f"{rep_save}Pusure/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

kwargs1 = {
    "tile1": "Pus pb1 = f(t)" + "\n",
    "tile_save": "pus_pb1",
    "colx": "t",
    "coly": "pusure_pb1",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$P_{W}^l \quad (W)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Pus pb2 = f(t)" + "\n",
    "tile_save": "pus_pb2",
    "colx": "t",
    "coly": "pusure_pb2",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$P_{W}^l \quad (W)$",
    "color1": color1[1],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Pus pb3 = f(t)" + "\n",
    "tile_save": "pus_pb3",
    "colx": "t",
    "coly": "pusure_pb3",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$P_{W}^l \quad (W)$",
    "color1": color1[2],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Pus ph1 = f(t)" + "\n",
    "tile_save": "pus_ph1",
    "colx": "t",
    "coly": "pusure_ph1",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$P_{W}^u \quad (W)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Pus ph2 = f(t)" + "\n",
    "tile_save": "pus_ph2",
    "colx": "t",
    "coly": "pusure_ph2",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$P_{W}^u \quad (W)$",
    "color1": color1[1],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Pus ph3 = f(t)" + "\n",
    "tile_save": "pus_ph3",
    "colx": "t",
    "coly": "pusure_ph3",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$P_{W}^u \quad (W)$",
    "color1": color1[2],
}
traj.pltraj2d(df, **kwargs1)
#%%############################################
#           PLOTS : points de choc :
###############################################
repsect1 = f"{rep_save}point_choc/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
# %%
kwargs1 = {
    "tile1": "point d'impact" + "\n",
    "tile_save": "pchocA_circ2d",
    "colx": "uxpicircA",
    "coly": "uypicircA",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": 'red',
    "msize" : 5,
}
traj.scat2d(df.loc[indchoc], **kwargs1)

kwargs1 = {
    "tile1": "point d'impact" + "\n",
    "tile_save": "pchocA_circ3d",
    "colx": "uxpicircA",
    "coly": "uypicircA",
    "colz": "uzpicircA",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1[1],
    "view": view,
}
traj.scat3d(df.loc[indchoc], **kwargs1)

# %%
kwargs1 = {
    "tile1": "point d'impact" + "\n",
    "tile_save": "pchocB_circ3d",
    "colx": "UXpincid",
    "coly": "UYpincid",
    "colz": "UZpincid",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1[1],
    "view": view,
}
traj.scat3d(df.loc[indchoc], **kwargs1)
# %%
kwargs1 = {
    "tile1": "point d'impact" + "\n",
    "tile_save": "pchocB_cone2d",
    "colx": "uxpisc",
    "coly": "uypisc",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": color1[1],
}
traj.scat2d(df.loc[indchoc], **kwargs1)
kwargs1 = {
    "tile1": "point d'impact" + "\n",
    "tile_save": "pchocB_cone3d",
    "colx": "uxpisc",
    "coly": "uypisc",
    "colz": "uzpisc",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1[1],
    "view": view,
}
traj.scat3d(df.loc[indchoc], **kwargs1)

#%%
kwargs1 = {
    "tile1": "point d'impact color pusure" + "\n",
    "tile_save": "pchocB_cone2d_colorbar_pusure",
    "colx": "uxpisc",
    "coly": "uypisc",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "mtype" : 'o',
    "msize" : 4,
    "colcol" : 'pusure_ccone',
    "ampl" : 200.,
    "title_colbar" : r"$log_{10}(1+$"+"Wear Power (W)"+r"$)$",
    "leg" : False,
    "logcol" : True
}
traj.scat2d_df_colorbar(df.loc[indchoc], **kwargs1)
#%%
kwargs1 = {
    "tile1": "point d'impact color pusure" + "\n",
    "tile_save": "pchocB_cone2d_colorbar_thmax",
    "colx": "uxpisc",
    "coly": "uypisc",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "mtype" : 'o',
    "msize" : 4,
    "colcol" : 'THMAX',
    "ampl" : 200.,
    # "title_colbar" : r"$log_{10}(1+$"+"Angular width ("+ r"$\degree$"+")"+r"$)$",
    "title_colbar" : "Angular width ("+ r"$\degree$"+")",
    "leg" : False,
    "logcol" : False
}
traj.scat2d_df_colorbar(df.loc[indchoc], **kwargs1)
# %%
repsect1 = f"{rep_save}point_choc/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
kwargs1 = {
    "tile1": "point d'impact color pctg glisadh" + "\n",
    "tile_save": "pchocB_cone2d_colorbar_glisad",
    "colx": "uxcerela",
    "coly": "uycerela",
    "rep_save": repsect1,
    "label1": r"$P_{choc}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "mtype" : 'o',
    "msize" : 4,
    "colcol" : 'PCTG_GLIS_ADH',
    "ampl" : 200.,
    # "title_colbar" : r"$log_{10}(1+$"+"Angular width ("+ r"$\degree$"+")"+r"$)$",
    "title_colbar" : "Glide/Adhesion " + r"$Glide/Adhesion %$",
    "leg" : False,
    "logcol" : False
}
traj.scat2d_df_colorbar(df.loc[indchoc], **kwargs1)
