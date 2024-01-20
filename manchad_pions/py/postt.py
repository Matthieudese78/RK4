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
lraidtimo = False
lplam = False
lplow = False
# lconefixe = True
# %% Scripts :
    # which slice ?
slice = 1
# cas la lache de la manchette avec juste le poids :
# namerep = "manchadela_weight"
# namerep = "manchadela_RSG"
# namerep = "manchadela_RSG_conefixe"


Fext = 100.
vlimoden = 1.e-4
spinini = 0.
vlostr = int(-np.log10(vlimoden))
namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}'
repload = f'./pickle/{namerep}/'
# namerep = f"manchadela_pions_{slice}"
# repload = f"./pickle/{namerep}/"

# %%
rep_save = f"./fig/{namerep}/"

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")

#%% contact time interval :
df['tag'] = df['FN_pb1'] < 0
fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
prb1 = [(i,j) for i,j in zip(fst,lst)]
# %% On change de repere pour le tracer des trajectoires :
exb = np.array([0.0, -1.0, 0.0])
eyb = np.array([0.0, 0.0, 1.0])
ezb = np.array([-1.0, 0.0, 0.0])
base2 = [exb, eyb, ezb]

# centre du cercle et son vis a vis :
name_cols = ["uxp2", "uyp2", "uzp2"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["uxce_ad", "uyce_ad", "uzce_ad"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

if (lplam):
    name_cols = ["uxplam_h", "uyplam_h", "uzplam_h"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["uxplam_b", "uyplam_b", "uzplam_b"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

name_cols = ["uxg_tot_ad", "uyg_tot_ad", "uzg_tot_ad"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

# name_cols = ["uxscone_tot", "uyscone_tot", "uzscone_tot"]
# kwargs1 = {"base2": base2, "name_cols": name_cols}
# rc.repchgdf(df, **kwargs1)

name_cols = ["uxph_ad", "uyph_ad", "uzph_ad"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["uxpb_ad", "uypb_ad", "uzpb_ad"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

if lplow:
    name_cols = ["uxplow", "uyplow", "uzplow"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)
    name_cols = ["uxplow_ad", "uyplow_ad", "uzplow_ad"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

if lraidtimo:
    name_cols = ["pix_ad", "piy_ad", "piz_ad"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["wx_ad", "wy_ad", "wz_ad"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["ax_ad", "ay_ad", "az_ad"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

# grandeur manchette :
name_cols = ["WX", "WY", "WZ"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["AX", "AY", "AZ"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["PIX", "PIY", "PIZ"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["VX_PINCID", "VY_PINCID", "VZ_PINCID"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

# on ne passe pas les depl. du point d'incidence dans le repere utilisateur
# rapport au traitement qu'il va subir par le suite
# name_cols = ["UXpincid", "UYpincid", "UZpincid"]
# kwargs1 = {"base2": base2, "name_cols": name_cols}
# rc.repchgdf(df, **kwargs1)

name_cols = ["uxg_m", "uyg_m", "uzg_m"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["uxph", "uyph", "uzph"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["uxpb", "uypb", "uzpb"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)


# grandeur ccone :
name_cols = ["FX_CCONE", "FY_CCONE", "FZ_CCONE"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["MX_CCONE", "MY_CCONE", "MZ_CCONE"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["VX_PINCID", "VY_PINCID", "VZ_PINCID"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

# %% trajectoires relatives
### calculs
lk = []
lk.append({"col1": "uxp2", "col2": "uxce_ad", "col3": "uxcerela"})
lk.append({"col1": "uyp2", "col2": "uyce_ad", "col3": "uycerela"})
lk.append({"col1": "uzp2", "col2": "uzce_ad", "col3": "uzcerela"})
lk.append({"col1": "uxpb", "col2": "uxpb_ad", "col3": "uxpbrela"})
lk.append({"col1": "uypb", "col2": "uypb_ad", "col3": "uypbrela"})
lk.append({"col1": "uzpb", "col2": "uzpb_ad", "col3": "uzpbrela"})
lk.append({"col1": "uxph", "col2": "uxph_ad", "col3": "uxphrela"})
lk.append({"col1": "uyph", "col2": "uyph_ad", "col3": "uyphrela"})
lk.append({"col1": "uzph", "col2": "uzph_ad", "col3": "uzphrela"})
if lplow:
    lk.append({"col1": "uxplow", "col2": "uxplow_ad", "col3": "uxplowrela"})
    lk.append({"col1": "uyplow", "col2": "uyplow_ad", "col3": "uyplowrela"})
    lk.append({"col1": "uzplow", "col2": "uzplow_ad", "col3": "uzplowrela"})
[traj.rela(df, **ki) for i, ki in enumerate(lk)]

# %% traitement du point de contact :

# %% matrice de rotation de la manchette :
kq = {"colname": "mrot", "q1": "quat1", "q2": "quat2", "q3": "quat3", "q4": "quat4"}
rota.q2mdf(df, **kq)

# %% extraction du spin :
    #%% matrice de rotation de la manchette dans le repere utilisateur :
name_cols = ["mrotu"]
kwargs1 = {"base2": base2, "mat": "mrot", "name_cols": name_cols}
rc.repchgdf_mat(df, **kwargs1)
    #%% extraction du spin : 
name_cols = ["spin"] 
kwargs1 = {"mat": "mrotu", "colnames": name_cols}
rota.spinextrdf(df,**kwargs1)
    # on drop la column "mrotu" qui prend de la place : 
df.drop(['mrotu'],inplace=True,axis=1)

# %% on fait xb - xcdr :
lk = []
lk.append({"col1": "UXpincid", "col2": "UXcdr", "col3": "uxpicircB"})
lk.append({"col1": "UYpincid", "col2": "UYcdr", "col3": "uypicircB"})
lk.append({"col1": "UZpincid", "col2": "UZcdr", "col3": "uzpicircB"})
[traj.rela(df, **ki) for i, ki in enumerate(lk)]
# on a lieu le contact ?
# indchoc = df[((np.sqrt(df['uxpicircB']**2+df['uypicircB']**2+df['uzpicircB']**2))>1.e-3)].index
indchoc = df[
    (
        (np.sqrt(df["UXpincid"] ** 2 + df["UYpincid"] ** 2 + df["UZpincid"] ** 2))
        > 1.0e-3
    )
].index
# %% on fait R^T * (xb - xcdr) :
ktr = {
    "colnames": ["uxpicircA", "uypicircA", "uzpicircA"],
    "mrot": "mrot",
    "x": "uxpicircB",
    "y": "uypicircB",
    "z": "uzpicircB",
}
# ktr = {'colnames' : ['uxpicircA', 'uypicircA', 'uzpicircA'] , 'mrot' : 'mrot', 'x' : 'UXpincid', 'y' : 'UYpincid', 'z' : 'UZpincid'}
rota.b2a(df, **ktr)
# on passe dans le repere utilisateur :
name_cols = ["uxpicircA", "uypicircA", "uzpicircA"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)


# %% recombinaison des depl. du sommet du cone :
lk = []
lk.append({"col1": "UXpincid", "col2": "uxscone_tot", "col3": "uxpisc"})
lk.append({"col1": "UYpincid", "col2": "uyscone_tot", "col3": "uypisc"})
lk.append({"col1": "UZpincid", "col2": "uzscone_tot", "col3": "uzpisc"})
[traj.rela(df, **ki) for i, ki in enumerate(lk)]

# %% on peut passer le point d'incidence dans la base utilisateur :
name_cols = ["UXpincid", "UYpincid", "UZpincid"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

name_cols = ["uxpisc", "uypisc", "uzpisc"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

# %%          COLORATION : 
    # en fonction de l'usure :
kcol = {'colx' : 'pusure_ccone', 'ampl' : 200., 'logcol' : False}
dfcolpus = traj.color_from_value(df.loc[indchoc],**kcol)

# kcol = {'colx' : "PCTG_GLIS_ADH", 'ampl' : 200., 'logcol' : False}
# dfcolglisad = traj.color_from_value(df.loc[indchoc],**kcol)

kcol = {'colx' : "FN_CCONE", 'color_normal' : 'black', 'color_impact' : 'orange'}
dfcolimpact = traj.color_impact(df,**kcol)
# alternative :
icrit = 1.e-12
# ccone :
indi_ccone = df[np.abs(df["FN_CCONE"])>icrit].index
indni_ccone = df.drop(indi_ccone).index
# pion haut :
indi_ph = df[(np.abs(df["FN_ph1"])>icrit) | 
             (np.abs(df["FN_ph2"])>icrit) |
             (np.abs(df["FN_ph3"])>icrit)].index
# indni_ph = df[(np.abs(df["FN_ph1"])<=icrit) | 
#              (np.abs(df["FN_ph2"])<=icrit) |
#              (np.abs(df["FN_ph3"])<=icrit)].index
indni_ph = df.drop(indi_ph).index

indi_pb = df[(np.abs(df["FN_pb1"])>icrit) | 
             (np.abs(df["FN_pb2"])>icrit) |
             (np.abs(df["FN_pb3"])>icrit)].index
indni_pb = df.drop(indi_pb).index

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
    "tile1": "Fext = f(t)" + "\n",
    "tile_save": "Fext_ft",
    "colx": "t",
    "coly": "Fext",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$F_{ext} \quad (N)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

# vitesses de rotations body frame :
kwargs1 = {
    "tile1": "Wx = f(t)" + "\n",
    "tile_save": "Wx_t",
    "colx": "t",
    "coly": "WX",
    "rep_save": repsect1,
    "label1": r"$W_{X}$",
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_X \quad (rad/s)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Wy = f(t)" + "\n",
    "tile_save": "Wy_t",
    "colx": "t",
    "coly": "WY",
    "rep_save": repsect1,
    "label1": r"$W_{Y}$",
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_Y \quad (rad/s)$",
    "color1": color1[1],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Wz = f(t)" + "\n",
    "tile_save": "Wz_t",
    "colx": "t",
    "coly": "WZ",
    "rep_save": repsect1,
    "label1": r"$W_{Z}$",
    "labelx": r"$t \quad (s)$",
    "labely": r"$W_Z \quad (rad/s)$",
    "color1": color1[2],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "Spin = f(t)" + "\n",
    "tile_save": "psi_t",
    "colx": "t",
    "coly": "spin",
    "rep_save": repsect1,
    "label1": r"$\psi$",
    "labelx": r"$t \quad (s)$",
    "labely": r"$\psi \quad (rad)$",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(G) adapter = f(t)" + "\n",
    "tile_save": "uygad_t",
    "colx": "t",
    "coly": "uyg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_y(G_ad)$"+" (m)",
    "color1": color1[0],
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

# usefull parameters :
dint_a = 70.e-3 
d_ext = 63.5*1.e-3 
d_pion = 68.83*1.e-3 
h_pion = ((d_pion - d_ext)/2.) 
ray_circ = ((dint_a/2.) - (d_ext/2.))
spinz = 0.
# excent = (0., h_pion)
excent = (0. ,(h_pion))
# secteur angulaire pris par un pion : 
sect_pion = (19. / (68.83 * np.pi ) ) * 360. / 2.
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
    "tile_save": "traj2d_pionh_colimpact",
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
}
traj.pltraj2d_pion(df, **kwargs1)
#%%
kwargs1 = {
    "tile1": "traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj2d_pionb_colimpact",
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
    "msize": 0.1,
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
