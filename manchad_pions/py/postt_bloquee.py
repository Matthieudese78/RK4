#!/bin/python3
#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import os
import matplotlib.pyplot as plt
import mplcursors
import sys
#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [20, -50]
#%% usefull parameters : pins
dint_a = 70.e-3 
d_ext = 63.5*1.e-3 
d_pion = 68.83*1.e-3 
h_pion = ((d_pion - d_ext)/2.) 
ray_circ = ((dint_a/2.) - (d_ext/2.))
spinz = 0.
excent = (0., h_pion)
# excent = (0. ,-h_pion)
# secteur angulaire pris par un pion : 
sect_pion_deg = (19. / (68.83 * 2.*np.pi ) ) * 360.
sect_pion_rad = (19. / (68.83 * 2.*np.pi ) ) * 2.*np.pi

jeumax = ray_circ - h_pion*np.sin(np.pi/6.) 
    # max clearance 
xcmax = h_pion*np.cos(np.pi/6.)
ycmax = np.sqrt((ray_circ**2) - (xcmax**2))
cmax = ycmax - (h_pion*np.sin(np.pi/6.))
# %% quel type de modele ?
lraidtimo = False
lplam = False
lplow = False
# lconefixe = True
# %% Scripts :
    # which slice ?
slice = 1
# pstt des chocs ?
lchoc = False
# cas la lache de la manchette avec juste le poids :
# namerep = "manchadela_weight"
# namerep = "manchadela_RSG"
# namerep = "manchadela_RSG_conefixe"

linert = True
lbfin = False
lamode = True
lkxp = False
lpion = False
lpcirc = True
Fext = 193.
mu = 0.6
xi = 0.05

amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 1.e-6
vlostr = int(-np.log10(vlimoden))
dtstr = int(-np.log10(dte))
xistr = int(100.*xi)

namerep = "/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/manchadela_pions_1"


rep_save = f"./fig/manchadela_bloquee/"

repload = f'{namerep}/'

if lbfin:
    # namerep = f'{namerep}_bfin'
    rep_save = f'{rep_save}bfin/'
    repload = f'./pickle/bfin/manchadela_pions_1/'

# %%
if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")
    # on trie et on reindexe :
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
# %% fenetrage en temps : 
t1 = 0.
if (not linert):
    t1 = 0.1 
t2 = 128.

# df = df[(df['t']>=t1) & (df['t']<=t2)]
# df.reset_index(drop=True,inplace=True)

# %% 100 points par seconde 
discr = 1.e-3
dtsort = df.iloc[1]['t'] - df.iloc[0]['t']
    # on veut un point ttes les :
ndiscr = int(discr/(dtsort))
df = df.iloc[::ndiscr]
# rows2keep = df.index % ndiscr == 0 
# df = df[rows2keep]
df.reset_index(drop=True,inplace=True)
dt = df['t'].iloc[1] - df['t'].iloc[0] 
fs = 1/dt
print(f"dt = {dt}")
print(f"fs = {fs}")
# %% frequency = f(t) : 
f1 = 2.
f2 = 20.
ttot = df.iloc[-1]['t']
df['freq'] = f1 + ((f2-f1)/ttot)*df['t'] 

#%%
nt = int(np.floor(np.log(len(df['t']))/np.log(2.)))
indexpsd = df[df.index < 2**nt].index
# %% On change de repere pour le tracer des trajectoires :
exb = np.array([0.0, -1.0, 0.0])
eyb = np.array([0.0, 0.0, 1.0])
ezb = np.array([-1.0, 0.0, 0.0])
base2 = [exb, eyb, ezb]
lrepchg = True
if lrepchg:
    # centre du cercle et son vis a vis :
    name_cols = ["uxp2", "uyp2", "uzp2"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["uxg_tot_ad", "uyg_tot_ad", "uzg_tot_ad"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["uxplow", "uyplow", "uzplow"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["uxph", "uyph", "uzph"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["uxpb", "uypb", "uzpb"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

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
lfext = np.ones_like(df['t'].values) * Fext
lt = df['t'].values
lfreq = np.linspace(2.,20.,len(df['t'].values))
#%%
kwargs1 = {
    "tile1": "amplitude Fext = f(t)" + "\n",
    "tile_save": "AFext_ft",
    "x": lt,
    "y": lfext,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{0} \quad (N)$",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d_list(**kwargs1)
kwargs1 = {
    "tile1": "frequence chargmt = f(t)" + "\n",
    "tile_save": "freq_ft",
    "x": lt,
    "y": lfreq,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$Frequency \quad (Hz)$",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d_list(**kwargs1)

# vitesses de rotations body frame :

kwargs1 = {
    "tile1": "ux(G) adapter = f(f)" + "\n",
    "tile_save": "uxgad_f",
    "colx": "freq",
    "coly": "uxg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$Loading Frequency$" + " (Hz)",
    "labely": r"$u_x(G_ad)$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "ux(G) adapter = f(t)" + "\n",
    "tile_save": "uxgad_t",
    "colx": "t",
    "coly": "uxg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$t$" + " (s)",
    "labely": r"$u_x(G_ad)$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)
kwargs1 = {
    "tile1": "uy(G) adapter = f(f)" + "\n",
    "tile_save": "uygad_f",
    "colx": "freq",
    "coly": "uyg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$Loading Frequency$" + " (Hz)",
    "labely": r"$u_y(G_ad)$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(G) adapter = f(t)" + "\n",
    "tile_save": "uygad_t",
    "colx": "t",
    "coly": "uyg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$t$" + " (s)",
    "labely": r"$u_y(G_ad)$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uz(G) adapter = f(f)" + "\n",
    "tile_save": "uzgad_f",
    "colx": "freq",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$Loading Frequency$" + " (Hz)",
    "labely": r"$u_z(G_ad)$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uz(G) adapter = f(t)" + "\n",
    "tile_save": "uzgad_t",
    "colx": "t",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$t$" + " (s)",
    "labely": r"$u_z(G_ad)$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PB) sleeve = f(f)" + "\n",
    "tile_save": "uypb_f",
    "colx": "freq",
    "coly": "uypb",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$Loading Frequency$" + " (Hz)",
    "labely": r"$u_y(P_{pin}^{l})$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PH) sleeve = f(f)" + "\n",
    "tile_save": "uyph_t",
    "colx": "freq",
    "coly": "uyph",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": r"$Loading Frequency$" + " (Hz)",
    "labely": r"$u_y(P_{pin}^{u})$"+" (m)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

#%%
sys.exit()
#%% energies :
repsect1 = f"{rep_save}energies/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

kwargs1 = {
    "tile1": "kinetic energy sleeve = f(t)" + "\n",
    "tile_save": "ekin_ft",
    "colx": "t",
    "coly": "EC",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{kin}$"+" (J)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "deformation energy sleeve = f(t)" + "\n",
    "tile_save": "edef_ft",
    "colx": "t",
    "coly": "edef",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{def}$"+" (J)",
    "color1": color1[1],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "weight potential energy sleeve = f(t)" + "\n",
    "tile_save": "epotw_ft",
    "colx": "t",
    "coly": "epot",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{w}$"+" (J)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "energies sleeve = f(t)" + "\n",
    "tile_save": "energies_ft",
    "colx": ["t","t","t","t"],
    "coly": ["EC","edef","epot","etot"],
    "rep_save": repsect1,
    "label1": [r"$E_{kin}$",r"$E_{def}$",r"$E_{w}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{kin},E_{def},E_{w},E_{tot}$"+" (J)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

# adapter :
kwargs1 = {
    "tile1": "kinetic energy  adapter = f(t)" + "\n",
    "tile_save": "ekin_ft_ad",
    "colx": "t",
    "coly": "ecad",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{kin}$"+" (J)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "deformation energy adapter = f(t)" + "\n",
    "tile_save": "edef_ft_ad",
    "colx": "t",
    "coly": "edefad",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{def}$"+" (J)",
    "color1": color1[1],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "weight potential adapter = f(t)" + "\n",
    "tile_save": "epotw_ft_ad",
    "colx": "t",
    "coly": "epotad",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{w}$"+" (J)",
    "color1": color1[2],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "energies sleeve adapter = f(t)" + "\n",
    "tile_save": "energies_ft_ad",
    "colx": ["t","t","t","t"],
    "coly": ["ecad","edefad","epotad","etotad"],
    "rep_save": repsect1,
    "label1": [r"$E_{kin}$",r"$E_{def}$",r"$E_{w}$",r"$E_{tot}$"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$E_{kin},E_{def},E_{w},E_{tot}$"+" (J)",
    "color1": color1,
    "endpoint": [False,False,False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)
#%%############################################
#           PLOTS : trajectoires relatives :
###############################################
repsect1 = f"{rep_save}traj_relatives_2d/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%% centre du cercle - vis a vis adapter : 
kwargs1 = {
    "tile1": "traj. relative PCcirc / PCad" + "\n",
    "tile_save": "traj2d_Ccirc",
    "colx": "uxcerela",
    "coly": "uycerela",
    "rep_save": repsect1,
    "label1": r"$C_{circ}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": 'black',
    "view": view,
}
traj.pltraj2d(df, **kwargs1)


# pion haut : 
kwargs1 = {
    "tile1": "traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_pionh",
    "colx": "uxphrela",
    "coly": "uyphrela",
    "rep_save": repsect1,
    "label1": r"$P_{pin}^u$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": 'black',
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz     
}
traj.pltraj2d_circs(df, **kwargs1)
#%%
kwargs1 = {
    "tile1": "traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_pionh_colimpact_ind",
    "ind": [indni_ph,indi_ph],
    "colx": "uxphrela",
    "coly": "uyphrela",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": ['no contact','contact'],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": ['black','orange'],
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [True,False],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

#%% melange des 2 :
kwargs1 = {
    "tile1": "traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_pionh_circ_colimpact",
    "ind": [indni_ph,indi_ph],
    "colx": "uxphrela",
    "coly": "uyphrela",
    "rep_save": repsect1,
    "label1": ['no contact','contact'],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": ['black','orange'],
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz,     
    "scatter" : True,
    "msize" : 0.1,
    "endpoint" : [True,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

#%%
kwargs1 = {
    "tile1": "traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj2d_pionb_circ_colimpact",
    "ind": [indni_pb,indi_pb],
    "colx": "uxpbrela",
    "coly": "uypbrela",
    "rep_save": repsect1,
    "label1": ['no contact','contact'],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": ['black','orange'],
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz,     
    "scatter" : True,
    "msize" : 0.1,
    "endpoint" : [True,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

if (lpcirc):
    kwargs1 = {
        "tile1": "traj. relative PB / PB_ad" + "\n",
        "tile_save": "traj2d_pionbS",
        "ind": [indni_pb,indi_pb1,indi_pb2,indi_pb3],
        "colx": "uxpbrela",
        "coly": "uypbrela",
        "rep_save": repsect1,
        "label1": ['no contact','contact pin 1','contact pin 2','contact pin 3'],
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "color1": ['black','red','green','blue'],
        "rcirc" : ray_circ,
        "excent" : excent,
        "spinz" : spinz,     
        "scatter" : True,
        "msize" : 0.1,
        "endpoint" : [True,False,False,False],
        "markers" : ['s','s','s','s'],
        "arcwidth" : sect_pion_deg,
        "clmax" : cmax,
        "xymax" : maxdeplPB,
    }
    traj.pltraj2d_pion(df, **kwargs1)

    kwargs1 = {
        "tile1": "traj. relative PB / PB_ad" + "\n",
        "tile_save": "traj2d_pionbS_contact_only",
        "ind": [indi_pb1,indi_pb2,indi_pb3],
        "colx": "uxpbrela",
        "coly": "uypbrela",
        "rep_save": repsect1,
        "label1": ['contact pin 1','contact pin 2','contact pin 3'],
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "color1": ['red','green','blue'],
        "rcirc" : ray_circ,
        "excent" : excent,
        "spinz" : spinz,     
        "scatter" : True,
        "msize" : 0.1,
        "endpoint" : [False,False,False],
        "markers" : ['s','s','s'],
        "arcwidth" : sect_pion_deg,
        "clmax" : cmax,
        "xymax" : maxdeplPB,
    }
    traj.pltraj2d_pion(df, **kwargs1)

    kwargs1 = {
        "tile1": "traj. relative PH / PH_ad" + "\n",
        "tile_save": "traj2d_pionhS_contact_only",
        "ind": [indi_ph1,indi_ph2,indi_ph3],
        "colx": "uxphrela",
        "coly": "uyphrela",
        "rep_save": repsect1,
        "label1": ['contact pin 1','contact pin 2','contact pin 3'],
        # "label1": ['no contact','contact '+r"$P_1^u$",'contact '+r"$P_2^u$", 'contact '+r"$P_3^u$"],
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "color1": ['red','green','blue'],
        "rcirc" : ray_circ,
        "excent" : excent,
        "spinz" : spinz,     
        "scatter" : True,
        "msize" : 0.1,
        "endpoint" : [False,False,False],
        "markers" : ['s','s','s'],
        "arcwidth" : sect_pion_deg,
        "clmax" : cmax,
        "xymax" : maxdeplPB,
    }
    traj.pltraj2d_pion(df, **kwargs1)

    kwargs1 = {
        "tile1": "traj. relative PH / PH_ad" + "\n",
        "tile_save": "traj2d_pionhS",
        "ind": [indni_ph,indi_ph1,indi_ph2,indi_ph3],
        "colx": "uxphrela",
        "coly": "uyphrela",
        "rep_save": repsect1,
        "label1": ['no contact','contact pin 1','contact pin 2','contact pin 3'],
        # "label1": ['no contact','contact '+r"$P_1^u$",'contact '+r"$P_2^u$", 'contact '+r"$P_3^u$"],
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "color1": ['black','red','green','blue'],
        "rcirc" : ray_circ,
        "excent" : excent,
        "spinz" : spinz,     
        "scatter" : True,
        "msize" : 0.1,
        "endpoint" : [True,False,False,False],
        "markers" : ['s','s','s','s'],
        "arcwidth" : sect_pion_deg,
        "clmax" : cmax,
        "xymax" : maxdeplPB,
    }
    traj.pltraj2d_pion(df, **kwargs1)

    if (transition):
        kwargs1 = {
            "tile1": "traj. relative PB / PB_ad" + "\n",
            "tile_save": "traj2d_transitions",
            "ind": ltrans,
            "colx": "uxpbrela",
            "coly": "uypbrela",
            "rep_save": repsect1,
            "label1": [f'trans{i+1}' for i,li in enumerate(ltrans)],
            "labelx": r"$X \quad (m)$",
            "labely": r"$Y \quad (m)$",
            "color1": ['red','green','blue']*int(np.floor(len(ltrans)/3)+1),
            "rcirc" : ray_circ,
            "excent" : excent,
            "spinz" : spinz,     
            "scatter" : True,
            "msize" : 0.1,
            "endpoint" : [True]*len(ltrans),
            "markers" : ['s']*len(ltrans),
            "arcwidth" : sect_pion_deg,
            "clmax" : cmax,
            "xymax" : maxdeplPB,
        }
        traj.pltraj2d_pion(df, **kwargs1)
#%%
kwargs1 = {
    "tile1": "traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_pionh_colimpact",
    "ind": [indni_pb,indi_pb],
    "colx": "uxphrela",
    "coly": "uyphrela",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": ['no contact','contact'],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": ['black','orange'],
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : True,
    "endpoint" : [True,False],
    "equalaxis" : True,
    # "legendfontsize" : 10.,
}
traj.pltraj2d_ind(df, **kwargs1)

#%% pion bas : 
kwargs1 = {
    "tile1": "traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj2d_pionb",
    "colx": "uxpbrela",
    "coly": "uypbrela",
    "rep_save": repsect1,
    "label1": r"$P_{pin}^l$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": 'black',
    "rcirc" : ray_circ,
    "excent" : excent,     
    "spinz" : spinz     
}
traj.pltraj2d_circs(df, **kwargs1)

#%%############################################
#           PLOTS : trajectoires relatives :
###############################################
repsect1 = f"{rep_save}traj_relatives_3d/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%%
kwargs1 = {
    "tile1": "traj. relative PCcirc / PCad" + "\n",
    "tile_save": "traj3d_Ccirc",
    "colx": "uxcerela",
    "coly": "uycerela",
    "colz": "uzcerela",
    "rep_save": repsect1,
    "label1": r"$C_{circ}$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1[1],
    "view": view,
}
traj.pltraj3d(df, **kwargs1)

#%%
kwargs1 = {
    "tile1": f"CDM traj. 3d PCcirc / PCad,colorimpact" + "\n",
    "tile_save": f"traj3d_Ccirc_colimpact",
    "ind": [indni_ccone,indi_ccone],
    "colx": "uxcerela",
    "coly": "uycerela",
    "colz": "uzcerela",
    "rep_save": repsect1,
    "label1": ['no contact','contact'],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": ["black","orange"],
    "view": view,
    "scatter": True,
    "endpoint": [True,False],
    "msize": 0.15,
    "loc_leg": (0.75,0.82),
    "markers": ['s','s'],
}
traj.pltraj3d_ind(df, **kwargs1)

#%%
kwargs1 = {
    "tile1": f"CDM traj. 3d PCcirc / PCad,colorimpact" + "\n",
    "tile_save": f"traj3d_Ccirc_colimpact",
    "ind": [indni_ccone,indi_ccone],
    "colx": "uxcerela",
    "coly": "uycerela",
    "colz": "uzcerela",
    "rep_save": repsect1,
    "label1": ['no contact','contact'],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": ["black","orange"],
    "view": view,
    "scatter": True,
    "msize": 0.1,
    "endpoint": [True,False],
    "markers": ['s','s'],
}
traj.pltraj3d_ccirc(df, **kwargs1)

#%%
kwargs1 = {
    "tile1": "traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj3d_ppionh",
    "colx": "uxphrela",
    "coly": "uyphrela",
    "colz": "uzphrela",
    "rep_save": repsect1,
    "label1": r"$P_{pin}^u$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1[0],
    "view": view,
}

traj.pltraj3d(df, **kwargs1)

kwargs1 = {
    "tile1": "traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj3d_ppionb",
    "colx": "uxpbrela",
    "coly": "uypbrela",
    "colz": "uzpbrela",
    "rep_save": repsect1,
    "label1": r"$P_{pin}^l$",
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "labelz": r"$Z \quad (m)$",
    "color1": color1[1],
    "view": view,
}

traj.pltraj3d(df, **kwargs1)

if lplow:
    kwargs1 = {
        "tile1": "traj. relative Plow / Plow_ad" + "\n",
        "tile_save": "traj3d_plow",
        "colx": "uxplowrela",
        "coly": "uyplowrela",
        "colz": "uzplowrela",
        "rep_save": repsect1,
        "label1": r"$P_{pin}^l$",
        "labelx": r"$X \quad (m)$",
        "labely": r"$Y \quad (m)$",
        "labelz": r"$Z \quad (m)$",
        "color1": color1[2],
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
if (lpcirc):
    # ph 1 : 
    kwargs1 = {
        "tile1": "FN ph1 = f(t)" + "\n",
        "tile_save": "fn_ph1",
        "colx": "t",
        "coly": "FN_pcirch1",
        "rep_save": repsect1,
        "label1": None,
        "labelx": r"$t \quad (s)$",
        "labely": r"$F_{n}^u \quad (N)$",
        "color1": color1[0],
    }
    traj.pltraj2d(df, **kwargs1)

    # pb 1 : 
    kwargs1 = {
        "tile1": "FN pb1 = f(t)" + "\n",
        "tile_save": "fn_pb1",
        "colx": "t",
        "coly": "FN_pcircb1",
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
        "tile_save": "fn_pb2",
        "colx": "t",
        "coly": "FN_pcircb2",
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
        "coly": "FN_pcircb3",
        "rep_save": repsect1,
        "label1": None,
        "labelx": r"$t \quad (s)$",
        "labely": r"$F_{n}^l \quad (N)$",
        "color1": color1[0],
    }
    traj.pltraj2d(df, **kwargs1)
if (lpion):
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
        "tile_save": "fn_pb2",
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
if (lpion):
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
if (lchoc):
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
        "msize" : 1,
    }
    traj.scat3d_pchoc(df.loc[indchoc], **kwargs1)

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
        "msize" : 1,
    }
    traj.scat3d_pchoc(df.loc[indchoc], **kwargs1)
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
        "colcol" : 'pctg_glis_ad',
        "ampl" : 200.,
        # "title_colbar" : r"$log_{10}(1+$"+"Angular width ("+ r"$\degree$"+")"+r"$)$",
        "title_colbar" : "Glide/Adhesion " + r"$Glide/Adhesion %$",
        "leg" : False,
        "logcol" : False
    }
    traj.scat2d_df_colorbar(df.loc[indchoc], **kwargs1)
