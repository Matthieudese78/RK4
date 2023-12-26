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
# quel cas ?
icas1 = 0
# quel algo ?
lialgo = [1,2,3]
if (icas1==0):
    # SW
    # n1 = 8 
    # n2 = 10
    # n3 = 12
    # NMB 
    n1 = 6
    n2 = 8
    n3 = 10


if (icas1==1):
    n1 = 8
    n2 = 10
    n3 = 12

if (icas1==2):
    n1 = 3
    n2 = 4
    n3 = 5

if (icas1==3):
    n1 = 10
    n2 = 11
    n3 = 13

if (icas1==4):
    n1 = 10
    n2 = 11
    n3 = 12

ln = [n1, n2, n3]
# ln = [n1]
labelh = [ r"$h = 2^{-%d}$" % ni for ni in ln ]
#%% point d'observation
# pobs = np.array([0.2,0.2,0.2])
pobs = np.array([1.,1.,1.])
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

# %% pour la toupie : calcul de l'energie potentielle :
if ((icas1==0) or (icas1==1)):
    kq = {"colname": "edef", "altitude": "uzg", "masse" : 2.}
    rota.epotdf(df, **kq)
#%% pour tous les cas :
kq = {"colname": "et", "ecin": "ec", "epot": "edef"}
rota.etotdf(df,**kq)
# %% trajectoire du point d'observation :
kp = {"mat": "mrot", "point": pobs, "colnames": ["uxpobs","uypobs","uzpobs"]}
rota.recopointdf(df, **kp)
# %%          COLORATION : 
    # en fonction du pdt :
kcol = {'colx' : 'n', 'ampl' : 200., 'logcol' : False}
dfcolpus = traj.color_from_value(df,**kcol)

#%% pour un algo : 
lindcas1 = [ [ df[(df['ialgo']==ialgi) & (df['icas']==(icas1+1)) & (df['n']==nj) ].index for i,ialgi in enumerate(lialgo) ] for j,nj in enumerate(ln) ]
# on passe ialgo en indice et on maj la liste de noms :
lialgo = list(map(lambda x: x - 1, lialgo))
lalgo = [ lalgo[ialgi] for ialgi in lialgo ]

for i,indi in enumerate(lindcas1):
    for j in np.arange(len(indi)): 
        ialij = df.iloc[lindcas1[i][j]]['ialgo'].drop_duplicates().values
        nij = df.iloc[lindcas1[i][j]]['n'].drop_duplicates().values
        print(f"lindcas1 [{i},{j}] : ")
        print(f"ialgo = {ialij}, n = {nij}")

#%%############################################
#           PLOTS : grandeures temporelles :
###############################################
repsect1 = f"{rep_save}variables_ft/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%% convected angular rotation speeds :
for ialg,alg in enumerate(lialgo):
    kwargs1 = {
        "tile1": f"Ws = f(t) cas {lscript[icas1]} ialgo = {lalgo[ialg]}" + "\n",
        "tile_save": f"Ws_pdts_{lscript[icas1]}_{lalgo[ialg]}",
        "colx": "t",
        "coly": ["wx","wy","wz"],
        "ind" : lindcas1[:][ialg],
        "rep_save": repsect1,
        "label1": labelh,
        "labelx": [r"$t \quad (s)$"] * 3,
        "labely": [r"$W_X \quad (rad/s)$", r"$W_Y \quad (rad/s)$",r"$W_Z \quad (rad/s)$"],
        "color1": color1,
        "loc_leg": (1.01,0.),
    }
    traj.pltsub2d_ind(df,**kwargs1)

#%% energies :
for ialg,alg in enumerate(lialgo):
    kwargs1 = {
        "tile1": f"Es = f(t) cas {lscript[icas1]} ialgo = {lalgo[ialg]}" + "\n",
        "tile_save": f"Es_pdts_{lscript[icas1]}_{lalgo[ialg]}",
        "colx": "t",
        "coly": ["ec","edef","et"],
        "ind" : lindcas1[:][ialg],
        "rep_save": repsect1,
        "label1": labelh,
        "labelx": [r"$t \quad (s)$"] * 3,
        "labely": [r"$E_{kin} \quad (J)$", r"$E_{pot} \quad (J)$",r"$E_{tot} \quad (J)$"],
        "color1": color1,
        "loc_leg": (1.01,0.),
    }
    traj.pltsub2d_ind(df,**kwargs1)


#%% moment cinetique : 
for ialg,alg in enumerate(lialgo):
    kwargs1 = {
        "tile1": f"Pis = f(t) cas {lscript[icas1]} ialgo = {lalgo[ialg]}" + "\n",
        "tile_save": f"Pis_pdts_{lscript[icas1]}_{lalgo[ialg]}",
        "colx": "t",
        "coly": ["pix","piy","piz"],
        "ind" : lindcas1[:][ialg],
        "rep_save": repsect1,
        "label1": labelh,
        "labelx": [r"$t \quad (s)$"] * 3,
        "labely": [r"$\Pi_{x} \quad (m^2.s^{-1})$", r"$\Pi_{y} \quad (m^2.s^{-1})$", r"$\Pi_{z} \quad (m^2.s^{-1})$"],
        "color1": color1,
        "loc_leg": (1.01,0.),
    }
    traj.pltsub2d_ind(df,**kwargs1)

#%%############################################
#           PLOTS : trajectoires 3d :
###############################################
repsect1 = f"{rep_save}traj_3d_pdts/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

for ialg,alg in enumerate(lialgo):
    kwargs1 = {
        "tile1": f"traj. 3d cas {lscript[icas1]} ialgo = {lalgo[ialg]}" + "\n",
        "tile_save": f"traj3d_pdts_{lscript[icas1]}_{lalgo[ialg]}",
        "ind": lindcas1[ialg],
        "colx": "uxg",
        "coly": "uyg",
        "colz": "uzg",
        "rep_save": repsect1,
        "label1": labelh,
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
        "label1": ["SW","NMB","RKMK4"],
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