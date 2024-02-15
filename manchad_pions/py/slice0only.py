#%%
import os
# from csv_to_pickle import csv2pickle
import numpy as np
import pandas as pd
import rotation as rota
import trajectories as traj
import repchange as rc
import glob
import shutil
from csv_to_pickle import csv2pickle
#%%
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
#%% repertoire data
  # on discretise ?
ndiscr = 1
linert = True
lamode = True
lkxp = False
# Fext = 387.
Fext = 193.
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 5.e-6
h_lam = 50.e-3
lspring = 45.e-2

vlostr = int(-np.log10(vlimoden))
# dtstr = int(-np.log10(dte))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
hlstr = int(h_lam*1.e3)
lspringstr = int(lspring*1.e2)
namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_lspr_{lspringstr}'

amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))
if lamode:
  namerep = f'{namerep}_amodem_{amodemstr}_amodead_{amodeadstr}'

if (lkxp):
  namerep = f'{namerep}_kxp'
if (linert):
  namerep = f'{namerep}_inert'

repload = f'../{namerep}/pickle/'
repsave = f'./pickle/slice_0/'

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")

repglob = f'../{namerep}/'
#%%
filename = 'manchadela_pions.dgibi'
rawname = filename.split('.')[0]
slice = 0
print(f"saving {slice}th calc...")
kwpi = {'rep_load' : f"{repglob}calc_{slice}/data/", 
        'rep_save' : f"{repsave}",
        'name_save' : f"result_{slice}"}
csv2pickle(**kwpi)
# %%
df = pd.read_pickle(f"{repsave}result_0.pickle")

# rep_save = f"./fig/{namerep}/slice_0/"

rep_save = f"./fig/slice_0/"
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
