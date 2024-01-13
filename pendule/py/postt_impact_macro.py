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
import shutil

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [30, -45]
#%%  
limpact = True 

stoia = True 

manchette = False

linert = True

motstoia = "faux"
motmanchette = "faux"
motimpact = "faux"
if (stoia):
    motstoia = "vrai"
if (manchette):
    motmanchette = "vrai"
if (limpact):
    motimpact = "vrai"

# %% Scripts :
if (limpact):
    if (stoia):
        repload = f'./pickle/impact/stoia/'
        repsave = f'./fig/impact/stoia/'
    if (manchette):
        repload = f'./data/impact/manchette/'
        repsave = f'./fig/impact/manchette/'

if (not limpact):
    if (stoia):
        repload = f'./pickle/no_impact/stoia/'
        repsave = f'./fig/no_impact/stoia/'
    if (manchette):
        repload = f'./pickle/no_impact/manchette/'
        repsave = f'./fig/no_impact/manchette/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
else:
    print(f"FOLDER : {repsave} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")

# %% detection des chocs :
dt = 1.e-6
df['tag'] = df['fn'] < 0
fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
prb1 = [(i,j) for i,j in zip(fst,lst)]
tchoc = [ dt*(j-i+1) for i,j in zip(fst,lst) ]
meantchoc = np.mean(tchoc)
instants_chocs = df.iloc[fst]['t']

#%%
filename = 'pendule_timo.dgibi'
for i,fsti in enumerate(fst[:4]):
    # creation du repo :
    repchoc1 = f'../chocs/choc{i+1}/data/'
    if not os.path.exists(repchoc1):
        os.makedirs(repchoc1)
        print(f"FOLDER : {repchoc1} created.")
    else:
        print(f"FOLDER : {repchoc1} already exists.")
    # on copie le script dans choci :
    source = f'../{filename}'
    destination = f'../chocs/choc{i+1}/'
    shutil.copy(source,f"{destination}{filename}")
    # on copie l'executable :
    if (linert):
        shutil.copy('../cast_inert/cast_64_21',f"{destination}/cast_64_21")
    if (not linert):
        shutil.copy('../cast/cast_64_21',f"{destination}/cast_64_21")

    # Sample DataFrame
    nmode = df.iloc[fst[i]]['nmode']
    dt = [df.iloc[fst[i]]['dt']], 
    dictini = {"wxini" : [df.iloc[fst[i]-1]['wx']],
               "wyini" : [df.iloc[fst[i]-1]['wy']], 
               "wzini" : [df.iloc[fst[i]-1]['wz']], 
               "quat1" : [df.iloc[fst[i]-1]['quat1']], 
               "quat2" : [df.iloc[fst[i]-1]['quat2']], 
               "quat3" : [df.iloc[fst[i]-1]['quat3']], 
               "quat4" : [df.iloc[fst[i]-1]['quat4']], 
               "t" : [tchoc[i] + 5.*dt], 
               "nmode_ela" : [df.iloc[fst[i]]['nmode']], 
               "nnoeuds" : [df.iloc[fst[i]]['nmode']], 
               "stoia" : [motstoia], 
               "manchette" : [motmanchette], 
               "limpact" : [motimpact], 
               "bamo" : [df.iloc[fst[i]]['amor']], 
               "dte" : [df.iloc[fst[i]]['dt']], 
               "lbar" : [df.iloc[fst[i]]['lbar']], 
              } 
    # on ajoute toutes les coord modales :
    for i in range(nmode): 
        dictini[f'q{i+1}'] = df.iloc[fst[i]][f'q{i+1}']
        dictini[f'q{i+1}v'] = df.iloc[fst[i]][f'q{i+1}v']

    dfini = pd.DataFrame(dictini)

    # Read the content of the existing file
    with open(f"{destination}{filename}", 'r') as file:
        lines = file.readlines()

    # Update the values based on the DataFrame
    # for index, row in dfini.iterrows():
    for colname, colvalue in dfini.iteritems():
        # Assuming the column names match the tags in the file (e.g., val1, val2)
        tag = f'*# {colname} :'
        print(tag)
        replacement = f'{colname} = {colvalue.values[0]} ;'
        print(replacement)

        # Find and replace the line with the updated value
        for i in range(len(lines)):
            if lines[i].strip() == tag:
                lines[i + 1] = f'{replacement}\n'
                break

    # Write the modified content back to the file
    with open(f"{destination}{filename}", 'w') as file:
        file.writelines(lines)

#%%############################################
#           PLOTS : grandeures temporelles :
###############################################
repsect1 = f"{repsave}variables_ft/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%%  :
n = 6
kwargs1 = {
    "tile1": f"zp2 = f(t),"+r"$h = 10^{-%d}$" % n + "\n",
    "tile_save": f"zp2_ft_n_{n}",
    "colx": "t",
    "coly": "uzp2",
    "ind" : df.index,
    "rep_save": repsect1,
    "label1": r"$h = 10^{-%d}$" % n,
    "labelx": r"$t \quad (s)$",
    "labely": r"$z \quad (m)$",
    "color1": color1[0],
    "loc_leg": (1.01,0.),
    "scatter": False,
    "endpoint": False,
    "msize": 2.,
    }
traj.pltraj2d_ind(df,**kwargs1)
#%%
kwargs1 = {
    "tile1": f"fn = f(t),"+r"$h = 10^{-%d}$" % n + "\n",
    "tile_save": f"fn_ft_n_{n}",
    "colx": "t",
    "coly": "fn",
    "ind" : df.index,
    "rep_save": repsect1,
    "label1": r"$h = 10^{-%d}$" % n,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_n \quad (N)$",
    "color1": color1[0],
    "loc_leg": (1.01,0.),
    "scatter": False,
    "endpoint": False,
    "msize": 2.,
    }
traj.pltraj2d_ind(df,**kwargs1)
#%% energies :
for ialg,alg in enumerate(lialgo):
    kwargs1 = {
        "tile1": f"Es = f(t) cas {lscript[icas1]} ialgo = {lalgo[ialg]}" + "\n",
        "tile_save": f"Es_pdts_{lscript[icas1]}_{lalgo[ialg]}",
        "colx": "t",
        "coly": ["ec","edef","et"],
        "ind" : lindcas1[ialg],
        "rep_save": repsect1,
        "label1": labelh[ialg],
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
        "ind" : lindcas1[ialg],
        "rep_save": repsect1,
        "label1": labelh[ialg],
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
        "tile1": f"CDM traj. 3d cas {lscript[icas1]} ialgo = {lalgo[ialg]}" + "\n",
        "tile_save": f"cdm_traj3d_pdts_{lscript[icas1]}_{lalgo[ialg]}",
        "ind": lindcas1[ialg],
        "colx": "uxg",
        "coly": "uyg",
        "colz": "uzg",
        "rep_save": repsect1,
        "label1": labelh[ialg],
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "labelz": r"$Z \quad (m)$",
        "color1": color1,
        "view": view,
    }
    traj.pltraj3d_ind(df, **kwargs1)

for ialg,alg in enumerate(lialgo):
    kwargs1 = {
        "tile1": f"Pobs traj. 3d cas {lscript[icas1]} ialgo = {lalgo[ialg]}" + "\n",
        "tile_save": f"pobs_traj3d_pdts_{lscript[icas1]}_{lalgo[ialg]}",
        "ind": lindcas1[ialg],
        "colx": "uxpobs",
        "coly": "uypobs",
        "colz": "uzpobs",
        "rep_save": repsect1,
        "label1": labelh[ialg],
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "labelz": r"$Z \quad (m)$",
        "color1": color1,
        "view": view,
    }
    traj.pltraj3d_ind(df, **kwargs1)

#%%############################################
#   comparaison des resultats converges :
###############################################
repsect1 = f"{rep_save}comp_converged/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

repsect2 = f"{repsect1}variables_ft/"
if not os.path.exists(repsect2):
    os.makedirs(repsect2)
    print(f"FOLDER : {repsect2} created.")
else:
    print(f"FOLDER : {repsect2} already exists.")

kwargs1 = {
    "tile1": f"converged results, Ws = f(t) cas {lscript[icas1]}" + "\n",
    "tile_save": f"converged_Ws_pdts_{lscript[icas1]}",
    "colx": "t",
    "coly": ["wx","wy","wz"],
    "ind" : lindconv,
    "rep_save": repsect2,
    "label1": labelconv,
    "labelx": [r"$t \quad (s)$"] * 3,
    "labely": [r"$W_X \quad (rad/s)$", r"$W_Y \quad (rad/s)$",r"$W_Z \quad (rad/s)$"],
    "color1": color1,
    "loc_leg": (1.01,0.),
}
traj.pltsub2d_ind(df,**kwargs1)

kwargs1 = {
    "tile1": f"converged results, Es = f(t) cas {lscript[icas1]}" + "\n",
    "tile_save": f"converged_Es_pdts_{lscript[icas1]}",
    "colx": "t",
    "coly": ["ec","edef","et"],
    "ind" : lindconv,
    "rep_save": repsect2,
    "label1": labelconv,
    "labelx": [r"$t \quad (s)$"] * 3,
    "labely": [r"$E_{kin} \quad (J)$", r"$E_{pot} \quad (J)$",r"$E_{tot} \quad (J)$"],
    "color1": color1,
    "loc_leg": (1.01,0.),
}
traj.pltsub2d_ind(df,**kwargs1)

kwargs1 = {
    "tile1": f"converged results, Pis = f(t) cas {lscript[icas1]}" + "\n",
    "tile_save": f"converged_Pis_pdts_{lscript[icas1]}",
    "colx": "t",
    "coly": ["pix","piy","piz"],
    "ind" : lindconv,
    "rep_save": repsect2,
    "label1": labelconv,
    "labelx": [r"$t \quad (s)$"] * 3,
    "labely": [r"$\pi_{x} \quad (m^2.s^{-1})$", r"$\pi_{y} \quad (m^2.s^{-1})$", r"$\pi_{z} \quad (m^2.s^{-1})$"],
    "color1": color1,
    "loc_leg": (1.01,0.),
}
traj.pltsub2d_ind(df,**kwargs1)

#%%
repsect2 = f"{repsect1}traj_3d_pdts/"
if not os.path.exists(repsect2):
    os.makedirs(repsect2)
    print(f"FOLDER : {repsect2} created.")
else:
    print(f"FOLDER : {repsect2} already exists.")

kwargs1 = {
    "tile1": f"converged results, CDM traj. 3d cas {lscript[icas1]}" + "\n",
    "tile_save": f"converged_cdm_traj3d_pdts_{lscript[icas1]}",
    "ind": lindconv,
    "colx": "uxg",
    "coly": "uyg",
    "colz": "uzg",
    "rep_save": repsect2,
    "label1": labelconv,
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1,
    "view": view,
}
traj.pltraj3d_ind(df, **kwargs1)
#%%
kwargs1 = {
    "tile1": f"converged results, Pobs traj. 3d cas {lscript[icas1]}" + "\n",
    "tile_save": f"converged_pobs_traj3d_pdts_{lscript[icas1]}",
    "ind": lindconv,
    "colx": "uxpobs",
    "coly": "uypobs",
    "colz": "uzpobs",
    "rep_save": repsect2,
    "label1": labelconv,
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