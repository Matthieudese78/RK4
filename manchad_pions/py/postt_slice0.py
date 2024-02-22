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
# %% quel type de modele ?
lraidtimo = True
lraidiss = True
lplam = False
lplow = False

ltest = False
# %% Scripts :
    # which slice ?
slice = 1
# cas la lache de la manchette avec juste le poids :
# namerep = "manchadela_weight"
# namerep = "manchadela_RSG"
# namerep = "manchadela_RSG_conefixe"
linert = True
lamode = True
dte = 5.e-6
# Fext = 193
Fext = 35
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
amodemstr = str(int(100.*amode_m))
amodeadstr = str(int(100.*amode_ad))
vlimoden = 1.e-5
spinini = 0.
h_lam = 50.e-3
lspring = 45.e-2

hlstr = int(h_lam*1.e3)
lspringstr = int(lspring*1.e2)

vlostr = int(-np.log10(vlimoden))
# dtstr = int(-np.log10(dte))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)

namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_lspr_{lspringstr}'
if (lamode):
    namerep = f'{namerep}_amodem_{amodemstr}_amodead_{amodeadstr}'
if (linert):
    namerep = f'{namerep}_inert'
if (lraidtimo):
  namerep = f'{namerep}_raidtimo'
if (lraidiss):
  namerep = f'{namerep}_raidiss'

repload = f'./pickle/{namerep}/'
# namerep = f"manchadela_pions_{slice}"
# repload = f"./pickle/{namerep}/"

# %%
rep_save = f"./fig/{namerep}/slice_0/"

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

if ltest:
    repload = f'./pickle/slice_0/'
    rep_save = f"./fig/slice_0/"

    if not os.path.exists(rep_save):
        os.makedirs(rep_save)
        print(f"FOLDER : {rep_save} created.")
    else:
        print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result_0.pickle")

#%% contact time interval :
# df['tag'] = df['FN_pb1'] < 0
# fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
# lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
# prb1 = [(i,j) for i,j in zip(fst,lst)]
# %% On change de repere pour le tracer des trajectoires :
exb = np.array([0.0, -1.0, 0.0])
eyb = np.array([0.0, 0.0, 1.0])
ezb = np.array([-1.0, 0.0, 0.0])
base2 = [exb, eyb, ezb]

name_cols = ["uxg_tot_ad", "uyg_tot_ad", "uzg_tot_ad"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["uxg_m", "uyg_m", "uzg_m"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

# grandeur ccone :
name_cols = ["FX_CCONE", "FY_CCONE", "FZ_CCONE"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

# %% trajectoires relatives
### calculs
lk = []
lk.append({"col1": "uxg_m", "col2": "uxg_tot_ad", "col3": "uxgmadrela"})
lk.append({"col1": "uyg_m", "col2": "uyg_tot_ad", "col3": "uygmadrela"}) 
lk.append({"col1": "uzg_m", "col2": "uzg_tot_ad", "col3": "uzgmadrela"}) 
          
[traj.rela(df, **ki) for i, ki in enumerate(lk)]

# %% matrice de rotation de la manchette :
kq = {"colname": "mrot", "q1": "quat1", "q2": "quat2", "q3": "quat3", "q4": "quat4"}
rota.q2mdf(df, **kq)

# %% extraction du spin :
    #%% matrice de rotation de la manchette dans le repere utilisateur :
name_cols = ["mrotu"]
kwargs1 = {"base2": base2, "mat": "mrot", "name_cols": name_cols}
rc.repchgdf_mat(df, **kwargs1)

#%%############################################
#           PLOTS : grandeures temporelles :
###############################################
repsect1 = f"{rep_save}variables_ft/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%% forces d'excitation :
kwargs1 = {
    "tile1": "fz ccone = f(t)" + "\n",
    "tile_save": "fzccone_ft",
    "colx": "t",
    "coly": "FZ_CCONE",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{z} \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

# vitesses de rotations body frame :
kwargs1 = {
    "tile1": "uzgmadrela = f(t)" + "\n",
    "tile_save": "uzgmadrela_ft",
    "colx": "t",
    "coly": "uzgmadrela",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_z(G_s) - u_z(G_{ad}) \quad (m)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uz G manchette = f(t)" + "\n",
    "tile_save": "uzgm_t",
    "colx": "t",
    "coly": "uzg_m",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_z(G_s) \quad (m)$",
    "color1": color1[1],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uz G adapter = f(t)" + "\n",
    "tile_save": "uzgad_ft",
    "colx": "t",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_z(G_{ad}) \quad (m)$",
    "color1": color1[1],
}
traj.pltraj2d(df, **kwargs1)

#%%############################################
#           PLOTS : forces de chocs :
###############################################
repsect1 = f"{rep_save}forces_de_choc/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
kwargs1 = {
    "tile1": "FZ ccone = f(t)" + "\n",
    "tile_save": "fzccone_ft",
    "colx": "t",
    "coly": "FZ_CCONE",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{z} \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)
