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
# excent = (0., h_pion)
excent = (0. ,(h_pion))
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
lraidiss = True
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

linert = False
lamode = True

lpion = False
lpcirc = True

# Fext = 193.
Fext = 2.*79.44
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 5.e-6
h_lam = 50.e-3
b_lam = 9.e-3
lspring = 45.e-2

hlstr = int(h_lam*1.e3)
blstr = int(b_lam*1.e3)
lspringstr = int(lspring*1.e2)
vlostr = int(-np.log10(vlimoden))
# dtstr = int(-np.log10(dte))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_bl_{blstr}_lspr_{lspringstr}'
amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))
if lamode:
    namerep = f'{namerep}_amodem_{amodemstr}_amodead_{amodeadstr}'
if (linert):
    namerep = f'{namerep}_inert'
if (lraidtimo):
    namerep = f'{namerep}_raidtimo'
if (lraidiss):
    namerep = f'{namerep}_raidiss'

repload = f'./pickle/{namerep}/'
rep_save = f"./fig/{namerep}/split/"

# %%

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}2048/result.pickle")
    # on trie et on reindexe :
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
# %% fenetrage en temps : 
t1 = 0.
if (not linert):
    t1 = 0.1 
t2 = 128.
# df = df[(df['t']>t1) & (df['t']<=t2)]
# df.reset_index(drop=True,inplace=True)

# %% 100 points par seconde 
    # nsort = 10 et x4 dans dopickle_slices :
if (not linert):
    nsort = 40
if (linert):
    nsort = 30
    # on veut un point ttes les :
discr = 1.e-3
dtsort = df.iloc[1]['t'] - df.iloc[0]['t']
ndiscr = int(discr/(dtsort))
df = df.iloc[::ndiscr]
# rows2keep = df.index % ndiscr == 0 
# df = df[rows2keep]
df.reset_index(drop=True,inplace=True)
#%%
f1 = 2.
f2 = 20.
ttot = df.iloc[-1]['t']
print(f"tmin = {df.iloc[0]['t']}")
print(f"ttot = {ttot}")
df['freq'] = f1 + ((f2-f1)/ttot)*df['t'] 
print(f"fmin = {df.iloc[0]['freq']}")
print(f"fmax = {df.iloc[-1]['freq']}")
#%%
nt = int(np.floor(np.log(len(df['t']))/np.log(2.)))
indexpsd = df[df.index < 2**nt].index
#%% contact time interval :
if (lpion):
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
    name_cols = ["PIX_AD", "PIY_AD", "PIZ_AD"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["WX_AD", "WY_AD", "WZ_AD"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

    name_cols = ["AX_AD", "AY_AD", "AZ_AD"]
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

# %% maximum penetration :
penePB = cmax - df['uypbrela'].abs().max()
penePH = cmax - df['uyphrela'].abs().max()
maxdeplPB = df['uypbrela'].abs().max()
maxdeplPH = df['uyphrela'].abs().max()
maxdeplCC = np.sqrt(df['uxcerela'].abs().max()**2 + df['uycerela'].abs().max()**2)

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
if (lpion):
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

if (lpcirc):
    # pion haut :
    indi_ph = df[(np.abs(df["FN_pcirch1"])>icrit) | 
                 (np.abs(df["FN_pcirch2"])>icrit) |
                 (np.abs(df["FN_pcirch3"])>icrit)].index
    indni_ph = df.drop(indi_ph).index

    # pion bas :
    indi_pb = df[(np.abs(df["FN_pcircb1"])>icrit) | 
                 (np.abs(df["FN_pcircb2"])>icrit) |
                 (np.abs(df["FN_pcircb3"])>icrit)].index
    indni_pb = df.drop(indi_pb).index

    indi_pb1 = df[(np.abs(df["FN_pcircb1"])>icrit)].index
    indi_pb2 = df[(np.abs(df["FN_pcircb2"])>icrit)].index
    indi_pb3 = df[(np.abs(df["FN_pcircb3"])>icrit)].index

    indi_ph1 = df[(np.abs(df["FN_pcirch1"])>icrit)].index
    indi_ph2 = df[(np.abs(df["FN_pcirch2"])>icrit)].index
    indi_ph3 = df[(np.abs(df["FN_pcirch3"])>icrit)].index
# split en temps :
if lraidtimo:
    t1 = 70.
    t2 = 110.
else:
    t1 = 50.
    t2 = 80.

indreso = df[(df['t']>=t1) & (df['t']<=t2)].index
indp1   = df[(df['t']<=t1)].index
indp2   = df[(df['t']>=t2)].index
    # ccone :
indnicc1     = indni_ccone.intersection(indp1)
indniccreso  = indni_ccone.intersection(indreso)
indnicc2     = indni_ccone.intersection(indp2)
indicc1      = indi_ccone.intersection(indp1)
indiccreso   = indi_ccone.intersection(indreso)
indicc2      = indi_ccone.intersection(indp2)
    # pion haut :
indnipb1    = indni_pb.intersection(indp1)
indnipbreso = indni_pb.intersection(indreso)
indnipb2    = indni_pb.intersection(indp2)
indipb1     = indi_pb.intersection(indp1)
indipbreso  = indi_pb.intersection(indreso)
indipb2     = indi_pb.intersection(indp2)
    # pion bas :
indniph1    = indni_ph.intersection(indp1)
indniphreso = indni_ph.intersection(indreso)
indniph2    = indni_ph.intersection(indp2)
indiph1     = indi_ph.intersection(indp1)
indiphreso  = indi_ph.intersection(indreso)
indiph2     = indi_ph.intersection(indp2)
#%% fenetrage en temps :
transition = False
if transition:
    tslice = 0.5
    ntrans = 3
    ltrans = []
    # itrans = range(1,ntrans)
    itrans = [55,57,60,62,74,86]
    for i in itrans:
        t1 = i*tslice - 1.e-2 
        t2 = i*tslice + 1.e-2 
        ltrans.append(df[(df['t']>t1) & (df['t']<t2)].index)

#%% energies :
M = 11.46
Mad = 25.643267377499903 
df['epot'] = M*df['uzg_m'] 
df['epotad'] = Mad*df['uzg_tot_ad'] 
df['etot'] = df['epot'] + df['edef'] + df['EC']
df['etotad'] = df['epotad'] + df['edefad'] + df['ecad']
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

#%%############################################
#           PLOTS : trajectoires relatives :
###############################################

repsect1 = f"{rep_save}variables_ft/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

kwargs1 = {
    "tile1": "uz(G) adapter = f(t)" + "\n",
    "tile_save": "uzccirc_t",
    "colx": "t",
    "coly": "uzcerela",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_z(C_{circ})$"+" (m)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 2,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uz(G) adapter = f(f)" + "\n",
    "tile_save": "uzccirc_f",
    "colx": "freq",
    "coly": "uzcerela",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency (Hz)",
    "labely": r"$u_z(C_{circ})$" + " (m)" ,
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 2,
}
traj.pltraj2d(df, **kwargs1)

repsect1 = f"{rep_save}traj_relatives_2d/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

kwargs1 = {
    "tile1": "Phase 1 : traj. relative Ccirc / PB_ad" + "\n",
    "tile_save": "traj2d_ccirc_circ_phase1",
    "ind": [indnicc1,indicc1],
    # "ind": [indpPB],
    "colx": "uxcerela",
    "coly": "uycerela",
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": "Resonance : traj. relative Ccirc / PB_ad" + "\n",
    "tile_save": "traj2d_ccirc_circ_reso",
    "ind": [indniccreso,indiccreso],
    # "ind": [indpPB],
    "colx": "uxcerela",
    "coly": "uycerela",
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": "Phase 2 : traj. relative Ccirc / PB_ad" + "\n",
    "tile_save": "traj2d_ccirc_circ_phase2",
    "ind": [indnicc2,indicc2],
    # "ind": [indpPB],
    "colx": "uxcerela",
    "coly": "uycerela",
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": "Phase 1 : traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_ph_circ_phase1",
    "ind": [indniph1,indiph1],
    # "ind": [indpPB],
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": "Resonance : traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_ph_circ_reso",
    "ind": [indniphreso,indiphreso],
    # "ind": [indpPB],
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": "Phase 2 : traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_ph_circ_phase2",
    "ind": [indniph2,indiph2],
    # "ind": [indpPB],
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)
    # pions bas
kwargs1 = {
    "tile1": "Phase 1 : traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj2d_pb_circ_phase1",
    "ind": [indnipb1,indipb1],
    # "ind": [indpPB],
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": "Resonance : traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj2d_pb_circ_reso",
    "ind": [indnipbreso,indipbreso],
    # "ind": [indpPB],
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": "Phase 2 : traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj2d_pb_circ_phase2",
    "ind": [indnipb2,indipb2],
    # "ind": [indpPB],
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
    "endpoint" : [False,False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    # "xymax" : maxdeplCC,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)

kwargs1 = {
    "tile1": f"CDM traj. 3d PCcirc / PCad,colorimpact" + "\n",
    "tile_save": f"traj3d_Ccirc_colimpact_p1",
    "ind": [indnicc1,indicc1],
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

kwargs1 = {
    "tile1": f"CDM traj. 3d PCcirc / PCad,colorimpact" + "\n",
    "tile_save": f"traj3d_Ccirc_colimpact_reso",
    "ind": [indniccreso,indiccreso],
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

kwargs1 = {
    "tile1": f"CDM traj. 3d PCcirc / PCad,colorimpact" + "\n",
    "tile_save": f"traj3d_Ccirc_colimpact_p2",
    "ind": [indnicc2,indicc2],
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

sys.exit()
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
