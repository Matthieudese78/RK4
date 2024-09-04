#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import os

# %% usefull parameters :
color1 = ["blue", "red", "green", "orange", "purple", "pink"]
view = [20, -50]
# %% quel type de modele ?
lraidtimo = False
lconefixe = True
lrsg      = False
lchoc45   = False
lchute    = True
# %% Scripts :
# cas la lache de la manchette avec juste le poids :
# script1 = "manchadela_weight"
script1 = "manchadela"
if lconefixe:
    script1 = f'{script1}_conefixe'
if lrsg:
    script1 = f'{script1}_RSG'
if lchoc45:
    script1 = f'{script1}_choc45'
if lchute:
    script1 = f'{script1}_chute'
# script1 = "manchadela_RSG_conefixe"
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

# %% On change de repere pour le tracer des trajectoires :
exb = np.array([0.0, -1.0, 0.0])
eyb = np.array([0.0, 0.0, 1.0])
ezb = np.array([-1.0, 0.0, 0.0])
base2 = [exb, eyb, ezb]

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

name_cols = ["uxplow", "uyplow", "uzplow"]
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
lk.append({"col1": "uxpb", "col2": "uxpb_ad", "col3": "uxpbrela"})
lk.append({"col1": "uypb", "col2": "uypb_ad", "col3": "uypbrela"})
lk.append({"col1": "uzpb", "col2": "uzpb_ad", "col3": "uzpbrela"})
lk.append({"col1": "uxph", "col2": "uxph_ad", "col3": "uxphrela"})
lk.append({"col1": "uyph", "col2": "uyph_ad", "col3": "uyphrela"})
lk.append({"col1": "uzph", "col2": "uzph_ad", "col3": "uzphrela"})
lk.append({"col1": "uxplow", "col2": "uxplow_ad", "col3": "uxplowrela"})
lk.append({"col1": "uyplow", "col2": "uyplow_ad", "col3": "uyplowrela"})
lk.append({"col1": "uzplow", "col2": "uzplow_ad", "col3": "uzplowrela"})
[traj.rela(df, **ki) for i, ki in enumerate(lk)]

# %% traitement du point de contact :

# %% matrice de rotation de la manchette :
kq = {"colname": "mrot", "q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4"}
rota.q2mdf(df, **kq)
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

# %%          COLORATION SYNTAXIQUE : 
kcol = {'colx' : 'PUSURE', 'ampl' : 200., 'logcol' : False}
dfcolpus = traj.color_from_value(df.loc[indchoc],**kcol)
# %%          PLOTS :
repsect1 = f"{rep_save}traj_relatives/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
kwargs1 = {
    "tile1": "traj. relative PCcirc / PCad" + "\n",
    "tile_save": "traj3d_Ccirc",
    "colx": "uxphrela",
    "coly": "uyphrela",
    "colz": "uzphrela",
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
# %%
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
    "labelx": r"$X$",
    "labely": r"$Y$",
    "color1": color1[1],
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
    "labelx": r"$X$",
    "labely": r"$Y$",
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
    "labelx": r"$X$"+" m",
    "labely": r"$Y$"+" m",
    "mtype" : 'o',
    "msize" : 4,
    "colcol" : 'PUSURE',
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
    "labelx": r"$X$"+" m",
    "labely": r"$Y$"+" m",
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
