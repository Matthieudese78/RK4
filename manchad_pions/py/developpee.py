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
import matplotlib as mpl
from matplotlib import gridspec
import mplcursors
import sys
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
#%% chatgpt : code qui marche :
# Sample data: replace this with your DataFrame
data = {
    'X': np.random.rand(100),  # X coordinates of points
    'Y': np.random.rand(100),  # Y coordinates of points
    'Value': np.random.rand(100),  # Attribute value
}

df = pd.DataFrame(data)

# Define the desired number of bins for X and Y
num_bins = 10

# Assign each point to a specific cell
df['X_bin'] = pd.cut(df['X'], bins=np.linspace(0, 1, num_bins + 1), labels=False)
df['Y_bin'] = pd.cut(df['Y'], bins=np.linspace(0, 1, num_bins + 1), labels=False)

# Create a new column 'Cell' to represent the cell for each point
df['Cell'] = list(zip(df['X_bin'], df['Y_bin']))

# Calculate the mean value for each cell
heatmap_data = df.groupby('Cell')['Value'].mean().reset_index()

# Create a MultiIndex for the heatmap
heatmap_data.set_index('Cell', inplace=True)

# Create a grid with all possible cell combinations
all_cells = pd.MultiIndex.from_product([range(num_bins), range(num_bins)], names=['X_bin', 'Y_bin'])
#
complete_grid = pd.DataFrame(index=all_cells)

# Merge the complete grid with the original heatmap_data
heatmap_data.index = pd.MultiIndex.from_tuples(heatmap_data.index, names=['X_bin', 'Y_bin'])

# Merge the complete grid with the original heatmap_data
heatmap_data_complete = pd.merge(complete_grid, heatmap_data, how='left', left_index=True, right_index=True)

# Reshape the heatmap_matrix into a 2D array
heatmap_matrix_2d = heatmap_data_complete['Value'].values.reshape(num_bins, num_bins)

# Plot the heatmap with rectangles
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_matrix_2d, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Mean Value'})
plt.title('Rectangle Heatmap with 10x10 Cells')
plt.xlabel('X-axis Cells')
plt.ylabel('Y-axis Cells')
# plt.show()

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

L_tete = 28.5*1.e-3 

# %% quel type de modele ?
lraidtimo = False
lraidiss = True
lsplit = True
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
lkxp = False
lpion = False
lpcirc = True
Fext = 2.*79.44
mu = 0.6
xi = 0.05

amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 5.e-6
vlostr = int(-np.log10(vlimoden))
# dtstr = int(-np.log10(dte))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
h_lam = 50.e-3
b_lam = 9.e-3
lspring = 45.e-2
hlstr = int(h_lam*1.e3)
blstr = int(b_lam*1.e3)
lspringstr = int(lspring*1.e2)

namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_bl_{blstr}_lspr_{lspringstr}'

amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))

if lamode:
    namerep = f'{namerep}_amodem_{amodemstr}_amodead_{amodeadstr}'
if (lkxp):
  namerep = f'{namerep}_kxp'
if (linert):
    namerep = f'{namerep}_inert'
if (lraidtimo):
    namerep = f'{namerep}_raidtimo'
if (lraidiss):
    namerep = f'{namerep}_raidiss'

repload = f'./pickle/{namerep}/'
rep_save = f"./fig/{namerep}/"

# %%
if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")
# %% fenetrage en temps : 
lfenetre = False
t1 = df['t'].iloc[0]
t2 = df['t'].iloc[-1]
if lfenetre:
    t1 = 5.
    t2 = 6.
    i1 = df[(df['t']>=t1) & (df['t']<=t2)].index
    t12 = 95.
    t22 = 96.
    i2 = df[(df['t']>=t12) & (df['t']<=t22)].index
    iwindow = i1.union(i2)
    df = df.iloc[iwindow]
    df.sort_values(by='t',inplace=True)
    df.reset_index(drop=True,inplace=True)
#%% on ne prend que les chocs ccone :
indchoc = df[df['FN_CCONE'].abs()>1.e-5].index
df = df.iloc[indchoc]
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)

# ndiscr = 2
# df = df.iloc[::ndiscr]
# df.sort_values(by='t',inplace=True)
# df.reset_index(drop=True,inplace=True)

# #%% frequence en fonction du temps :
# f1 = 2.
# f2 = 20.
# ttot = df.iloc[-1]['t']
# df['freq'] = f1 + ((f2-f1)/ttot)*df['t'] 

#%%
col2keep = ['t','FN_CCONE','THMAX','pusure_ccone','pctg_glis_ad','DIMP','RCINC',
            'quat1',
            'quat2',
            'quat3',
            'quat4',
            'uxscone_tot',
            'uyscone_tot',
            'uzscone_tot',
            'UXcdr',
            'UYcdr',
            'UZcdr',
            'UXpincid',
            'UYpincid',
            'UZpincid',
            'uxg_tot_ad',
            'uyg_tot_ad',
            'uzg_tot_ad']
if lraidtimo:
  col2keep.append(['quat1_ad','quat2_ad','quat3_ad','quat4_ad'])

df = df[col2keep]

# # %% 100 points par seconde 
#     # nsort = 10 et x4 dans dopickle_slices :
# if (not linert):
#     nsort = 40
# if (linert):
#     nsort = 30
#     # on veut un point ttes les :
# discr = 1.e-3
# ndiscr = int(discr/(dte*nsort))
# df = df.iloc[::ndiscr]
# # rows2keep = df.index % ndiscr == 0 
# # df = df[rows2keep]
# df.reset_index(drop=True,inplace=True)
# %% On change de repere pour le tracer des trajectoires :
exb = np.array([0.0, -1.0, 0.0])
eyb = np.array([0.0, 0.0, 1.0])
ezb = np.array([-1.0, 0.0, 0.0])
base2 = [exb, eyb, ezb]

# centre du cercle et son vis a vis :
# name_cols = ["uxp2", "uyp2", "uzp2"]
# kwargs1 = {"base2": base2, "name_cols": name_cols}
# rc.repchgdf(df, **kwargs1)

# name_cols = ["uxce_ad", "uyce_ad", "uzce_ad"]
# kwargs1 = {"base2": base2, "name_cols": name_cols}
# rc.repchgdf(df, **kwargs1)

# name_cols = ["VX_PINCID", "VY_PINCID", "VZ_PINCID"]
# kwargs1 = {"base2": base2, "name_cols": name_cols}
# rc.repchgdf(df, **kwargs1)

# %% matrice de rotation de la manchette :
kq = {"colname": "mrot", "q1": "quat1", "q2": "quat2", "q3": "quat3", "q4": "quat4"}
rota.q2mdf(df, **kq)
if lraidtimo:
    kq = {"colname": "mrot_ad", "q1": "quat1_ad", "q2": "quat2_ad", "q3": "quat3_ad", "q4": "quat4_ad"}
    rota.q2mdf(df, **kq)
# %% extraction du spin :
#     #%% matrice de rotation de la manchette dans le repere utilisateur :
# name_cols = ["mrotu"]
# kwargs1 = {"base2": base2, "mat": "mrot", "name_cols": name_cols}
# rc.repchgdf_mat(df, **kwargs1)

#     #%% extraction du spin : 
# name_cols = ["spin"] 
# kwargs1 = {"mat": "mrotu", "colnames": name_cols}
# rota.spinextrdf(df,**kwargs1)
#     # on drop la column "mrotu" qui prend de la place : 
# df.drop(['mrotu'],inplace=True,axis=1)

# %% on fait xb - xcdr :
lk = []
lk.append({"col1": "UXpincid", "col2": "UXcdr", "col3": "uxpicircB"})
lk.append({"col1": "UYpincid", "col2": "UYcdr", "col3": "uypicircB"})
lk.append({"col1": "UZpincid", "col2": "UZcdr", "col3": "uzpicircB"})
[traj.rela(df, **ki) for i, ki in enumerate(lk)]

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

    # on test : 
if lraidtimo:
    ktr = {
        "colnames": ["uxpiscA", "uypiscA", "uzpiscA"],
        "mrot": "mrot_ad",
        "x": "uxpisc",
        "y": "uypisc",
        "z": "uzpisc",
    }
    rota.b2a(df, **ktr)

    name_cols = ["uxpiscA", "uypiscA", "uzpiscA"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)
    # on passe dans le repere d'origine le centre du cercle a t = 0
    df['uzpiscA'] += L_tete

name_cols = ["uxpisc", "uypisc", "uzpisc"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
# on passe dans le repere d'origine le centre du cercle a t = 0
df['uzpisc'] += L_tete

#%% Ccirc_ad --> Pincid exprime dans le repere de l'adaptateur :
    # RQ : donne la meme chose que uxpiscA :
ltest = False
if (lraidtimo & ltest):
        # GadPi = Pincid - (XGad + uGad) (base B)
            # - uGad
    lk = []
    lk.append({"col1": "UXpincid", "col2": "uxg_tot_ad", "col3": "xgadpiB"})
    lk.append({"col1": "UYpincid", "col2": "uyg_tot_ad", "col3": "ygadpiB"})
    lk.append({"col1": "UZpincid", "col2": "uzg_tot_ad", "col3": "zgadpiB"})
    [traj.rela(df, **ki) for i, ki in enumerate(lk)]
            # - XGad
    xgad = 0.35245
    df['xgadpiB'] += -xgad 

        # GadPi = R_ad^T(Pincid - Gad)
    ktr = {
        "colnames": ["xgadpiA", "ygadpiA", "zgadpiA"],
        "mrot": "mrot_ad",
        "x": "xgadpiB",
        "y": "ygadpiB",
        "z": "zgadpiB",
    }
    rota.b2a(df, **ktr)
        # CcircadPi = CircGad + GadPi (base A)
            # altitude G altitude ccircad = l_tete
    ccircgad = xgad - L_tete
    df['xgadpiA'] += ccircgad
    df.rename(columns={'xgadpiA' : 'xcadpiA', 'ygadpiA' : 'ycadpiA', 'zgadpiA' : 'zcadpiA'},inplace=True)
        # on passe dans le repere utilisateur :
    name_cols = ["xcadpiA", "ycadpiA", "zcadpiA"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

#%% selection des colonnes de depl. : les 3 sont equivalents
colx = 'uxpisc'
coly = 'uypisc'
colz = 'uzpisc'

    # pour l'adaptateur : foireux
# colx = 'xcadpiA'
# coly = 'xcadpiA'
# colz = 'xcadpiA'
 
if lraidtimo:
    colx = 'uxpiscA'
    coly = 'uypiscA'
    colz = 'uzpiscA'

#%%############################################
#           preproc 
###############################################
zlim = 2.e-3
# zlim = 1.e-4
# df = df[(df[colz] <= np.mean(df[colz])+zlim) & (df[colz] >= np.mean(df[colz])-zlim)]
df = df[(df[colz] <= zlim) & (df[colz] >= -zlim)]
    # on trie et on reindexe :
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)

# df['th'] = np.arctan2(df['uypisc'],df['uxpisc']) * 180. / np.pi
df['th'] = np.arctan2(df[coly],df[colx]) * 180. / np.pi
df['thm'] = np.arctan2(df['uypicircA'],df['uxpicircA']) * 180. / np.pi

nbsect = 72
nbsect_ad = 144
nbz_ad = 80

# on prend les valeurs positives de deplacments impose :
dtsort = df['t'].iloc[1] - df['t'].iloc[0] 
df['DIMP'] = - df['DIMP']
df['Wener'] = df['pusure_ccone']*dtsort

#%%############################################
# SPLIT :
###############################################
if lsplit:
    if lraidtimo:
        ts = 70.
        te = 110.
        # tr1 = ts + (te-ts)/3.
        # tr2 = ts + 2.*(te-ts)/3.
        tr1 = ts 
        tr2 = te 
        indreso = df[(df['t']>=tr1) & (df['t']<=tr2)].index
        indp1   = df[(df['t']<=tr1)].index
        indp2   = df[(df['t']>=tr2)].index
        # indp1 = df[(df['t']>=ts) & (df['t']<=te)].index
        # indp2 = df[(df['t']>=ts2) & (df['t']<=te2)].index
        # ireso = i1.union(i2)
    else:
        ts = 20.
        te = 45.
        tr1 = ts 
        tr2 = te 
        indreso = df[(df['t']>=tr1) & (df['t']<=tr2)].index
        indp1   = df[(df['t']<=tr1)].index
        indp2   = df[(df['t']>=tr2)].index

# %% on peut passer le point d'incidence dans la base utilisateur :
plotpchoc = False
if plotpchoc:
    name_cols = ["UXpincid", "UYpincid", "UZpincid"]
    kwargs1 = {"base2": base2, "name_cols": name_cols}
    rc.repchgdf(df, **kwargs1)

#%% pour les heatmaps :
ymin = 1.01*df[colz].min()
ymax = 1.01*df[colz].max()
# %%          TEST : 
if lfenetre:
    i1 = df[(df['t']>=t1) & (df['t']<=t2)].index
    i2 = df[(df['t']>=t12) & (df['t']<=t22)].index
    iwindow = i1.union(i2)

    # ymin = 1.01*df.loc[iwindow,colz].min()
    # ymax = 1.01*df.loc[iwindow,colz].max()

    ymin = 1.01*df.loc[iwindow,colz].min()
    ymax = 1.01*df.loc[iwindow,colz].max()

    df1 = df.loc[i1]

    xticks = np.arange(ymin,ymax, step=20)
    yticks = np.arange(ymin,ymax, step=1.e-5)

    num_x = nbsect_ad 
    num_y = nbz_ad
    agreg = 'sum'
    colval = 'Wener'

    df1['Xbin'] = pd.cut(df1['th'], bins=np.linspace(-180.,180., nbsect_ad + 1), labels=False)
    df1['Ybin'] = pd.cut(df1[colz], bins=np.linspace(ymin,ymax, nbz_ad + 1), labels=False)

    nanx = df1[df1['th'].isna()].index
    nanz = df1[df1[colz].isna()].index
    if ((any(nanx)) or (any(nanz))):
        print("il y a des nans dans colx ou colz !!")

    nanxbin = df1[df1['Xbin'].isna()].index
    nanybin = df1[df1['Ybin'].isna()].index
    if ((any(nanxbin)) or (any(nanybin))):
        print("il y a des nans dans Xbin ou Ybin !!")

    # Create a new column 'Cell' to represent the cell for each point
    df1['Cell'] = list(zip(df1['Xbin'], df1['Ybin']))
    if (agreg=='mean'):
        heatmap_data = df1.groupby('Cell')[colval].mean().reset_index()
    if (agreg=='sum'):
        heatmap_data = df1.groupby('Cell')[colval].sum().reset_index()

    # Create a MultiIndex for the heatmap
    heatmap_data.set_index('Cell', inplace=True)

    all_cells = pd.MultiIndex.from_product([range(num_x+1), range(num_y+1)], names=['Xbin', 'Ybin'])
    #
    complete_grid = pd.DataFrame(index=all_cells)
    # complete_grid.loc['Ybin',colval] = np.nan

    heatmap_data.index = pd.MultiIndex.from_tuples(heatmap_data.index, names=['Xbin', 'Ybin'])

    heatmap_data_complete = heatmap_data.combine_first(complete_grid)

    heatmap_data_complete = heatmap_data_complete.values.reshape(num_x+1,num_y+1)

    #### plt heatmap
    f = plt.figure(figsize=(8, 4),dpi=600)
    axes = f.gca()
    axes.imshow(np.transpose(heatmap_data_complete),
    # axes.imshow((heatmap_matrix_2d),
                cmap = plt.cm.inferno,
                interpolation='nearest', 
                aspect='auto', 
                origin='lower',
                extent=[-180.,180.,ymin,ymax],
               )
    # plt.show()
    plt.close('all')

#%%############################################
#           preproc 
###############################################
repsect1 = f"{rep_save}developpee/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%% wear power 
kw = {
        "colval" : "pusure_ccone",
        "angle"  : "thm", 
        "nbsect" : nbsect,
        "title1" : "WP_sleeve",
        "title_save" : "WP_sleeve",
        "rep_save" : repsect1,
        "title_colbar" : "Wear Power (W)",
}
traj.plt_heat_circ(df,**kw)

kw = {
        "colval" : "Wener",
        "angle"  : "thm", 
        "nbsect" : nbsect,
        "title1" : "wear_energy_sleeve_sum",
        "title_save" : "wearener_sleeve_sum",
        "rep_save" : repsect1,
        "title_colbar" : "Wear Energy (J)",
        "agreg" : "sum",
}
traj.plt_heat_circ(df,**kw)
#%% thmax : 
kw = {
        "colval" : "THMAX",
        "angle"  : "thm", 
        "nbsect" : nbsect,
        "title1" : "thmax_sleeve",
        "title_save" : "thmax_sleeve",
        "rep_save" : repsect1,
        "title_colbar" : "Contact Angular Width " + r"$\theta_{max} \degree$",
}
traj.plt_heat_circ(df,**kw)

#%% dimp : 
kw = {
        "colval" : "DIMP",
        "angle"  : "thm", 
        "nbsect" : nbsect,
        "title1" : "dimp_sleeve",
        "title_save" : "dimp_sleeve",
        "rep_save" : repsect1,
        "title_colbar" : "Penetration " + r"$\delta_{imp}$" + " (m)",
}
traj.plt_heat_circ(df,**kw)
#%% Fn sleeve : 
kw = {
        "colval" : "FN_CCONE",
        "angle"  : "thm", 
        "nbsect" : nbsect,
        "title1" : "FN_sleeve",
        "title_save" : "FN_sleeve",
        "rep_save" : repsect1,
        "title_colbar" :"Force Normal Component " + r"$F_{n}$" + " (N)",
}
traj.plt_heat_circ(df,**kw)

#%% heatmap wear power :
kw = {
       "numx"   : nbsect_ad,
       "numy"   : nbz_ad,
       "colx"   : "th",
       "coly"   : colz,
       "colval" : "pusure_ccone",
       "agreg"  : "mean",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
    #    "ylim"   : [-1.e-4,1.e-4],
       "zlim"   : 1.e-3,
    }
heatmap_WPad_mean = traj.heatmap(df,**kw)

kw = {
       "numx"   : nbsect_ad,
       "numy"   : nbz_ad,
       "colx"   : "th",
       "coly"   : colz,
       "colval" : "pusure_ccone",
       "agreg"  : "sum",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
    #    "ylim"   : [-1.e-4,1.e-4],
       "zlim"   : 1.e-3,
    }
heatmap_WPad_sum = traj.heatmap(df,**kw)
# plot wear power adapter :
kw = {
       "heatmap" : heatmap_WPad_sum, 
       "labelx"   : "Angular location " + r"$(\degree)$",
       "labely"   : "Elevation " + r"$z$" + " (m)",
       "title1"   : "heatmap_wearpower_sum",
       "title_save"   : "wearpower_ad_sum",
       "rep_save"   : repsect1,
       "colval" : "pusure_ccone",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
       "title_colbar"   : "Wear Power (W)",
       "cmap"   : "inferno",
    }
traj.plt_heat(df,**kw)

kw = {
       "numx"   : nbsect_ad,
       "numy"   : nbz_ad,
       "colx"   : "th",
       "coly"   : colz,
       "colval" : "Wener",
       "agreg"  : "sum",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
    #    "ylim"   : [-1.e-4,1.e-4],
       "zlim"   : 1.e-3,
    }
heatmap_Wener_sum = traj.heatmap(df,**kw)
# plot wear power adapter :
kw = {
       "heatmap" : heatmap_Wener_sum, 
       "labelx"   : "Angular location " + r"$(\degree)$",
       "labely"   : "Elevation " + r"$z$" + " (m)",
       "title1"   : "heatmap_wearenergy_sum",
       "title_save"   : "wearener_ad_sum",
       "rep_save"   : repsect1,
       "colval" : "Wener",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
       "title_colbar"   : "Wear Energy (J)",
       "cmap"   : "inferno",
    }
traj.plt_heat(df,**kw)
#%% heatmap thmax :
kw = {
       "numx"   : nbsect_ad,
       "numy"   : nbz_ad,
       "colx"   : "th",
       "coly"   : colz,
       "colval" : "THMAX",
       "agreg"  : "mean",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
    #    "ylim"   : [-1.e-4,1.e-4],
    }
heatmap_WPad = traj.heatmap(df,**kw)
# plot wear power adapter :
kw = {
       "heatmap" : heatmap_WPad, 
       "labelx"   : "Angular location " + r"$(\degree)$",
       "labely"   : "Elevation " + r"$z$" + " (m)",
       "title1"   : "heatmap_thmax",
       "title_save"   : "thmax_ad",
       "rep_save"   : repsect1,
       "colval" : "THMAX",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
       "title_colbar"   : "Contact Angular Width " + r"$\theta_{max} \degree$",
       "cmap"   : "inferno",
    }
traj.plt_heat(df,**kw)

#%% heatmap Fn :
kw = {
       "numx"   : nbsect_ad,
       "numy"   : nbz_ad,
       "colx"   : "th",
       "coly"   : colz,
       "colval" : "FN_CCONE",
       "agreg"  : "mean",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
    #    "ylim"   : [-1.e-4,1.e-4],
    }
heatmap_WPad = traj.heatmap(df,**kw)
# plot wear power adapter :
kw = {
       "heatmap" : heatmap_WPad, 
       "labelx"   : "Angular location " + r"$(\degree)$",
       "labely"   : "Elevation " + r"$z$" + " (m)",
       "title1"   : "heatmap_FN",
       "title_save"   : "FN_ad",
       "rep_save"   : repsect1,
       "colval" : "FN_CCONE",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
       "title_colbar"   : "Force Normal Component " + r"$F_{n}$" + " (N)",
       "cmap"   : "inferno",
    }
traj.plt_heat(df,**kw)

# plot dimp adapter :
kw = {
       "numx"   : nbsect_ad,
       "numy"   : nbz_ad,
       "colx"   : "th",
       "coly"   : colz,
       "colval" : "DIMP",
       "agreg"  : "mean",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
    #    "ylim"   : [-1.e-4,1.e-4],
    }
heatmap_WPad = traj.heatmap(df,**kw)
kw = {
       "heatmap" : heatmap_WPad, 
       "labelx"   : r"$\delta_{imp}$" + " (m)",
       "labely"   : "Elevation " + r"$z$" + " (m)",
       "title1"   : "heatmap_dimp",
       "title_save"   : "dimp_ad",
       "rep_save"   : repsect1,
       "colval" : "DIMP",
       "xlim"   : [-180.,180.],
       "ylim"   : [1.1*ymin,1.1*ymax],
       "title_colbar"   : "Penetration " + r"$\delta_{imp}$" + " (m)",
       "cmap"   : "inferno",
    }
traj.plt_heat(df,**kw)

#%% split : 
repsect1 = f"{rep_save}developpee/split/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%%
if lsplit:
    lind = [indp1,indreso,indp2]
    for i,ind in enumerate(lind):
      kw = {
              "colval" : "pusure_ccone",
              "angle"  : "thm", 
              "nbsect" : nbsect,
              "title1" : "WP_sleeve",
              "title_save" : f"WP_sleeve_p{i+1}",
              "rep_save" : repsect1,
              "title_colbar" : "Wear Power (W)",
      }
      traj.plt_heat_circ(df.iloc[ind],**kw)
      
      kw = {
              "colval" : "Wener",
              "angle"  : "thm", 
              "nbsect" : nbsect,
              "title1" : "wear_energy_sleeve_sum",
              "title_save" : f"wearener_sleeve_sum_p{i+1}",
              "rep_save" : repsect1,
              "title_colbar" : "Wear Energy (J)",
              "agreg" : "sum",
      }
      traj.plt_heat_circ(df.iloc[ind],**kw)

      # thmax : 
      kw = {
              "colval" : "THMAX",
              "angle"  : "thm", 
              "nbsect" : nbsect,
              "title1" : "thmax_sleeve",
              "title_save" : f"thmax_sleeve_p{i+1}",
              "rep_save" : repsect1,
              "title_colbar" : "Contact Angular Width " + r"$\theta_{max} \degree$",
      }
      traj.plt_heat_circ(df.iloc[ind],**kw)
      
      # dimp : 
      kw = {
              "colval" : "DIMP",
              "angle"  : "thm", 
              "nbsect" : nbsect,
              "title1" : "dimp_sleeve",
              "title_save" : f"dimp_sleeve_p{i+1}",
              "rep_save" : repsect1,
              "title_colbar" : "Penetration " + r"$\delta_{imp}$" + " (m)",
      }
      traj.plt_heat_circ(df.iloc[ind],**kw)
      # Fn sleeve : 
      kw = {
              "colval" : "FN_CCONE",
              "angle"  : "thm", 
              "nbsect" : nbsect,
              "title1" : "FN_sleeve",
              "title_save" : f"FN_sleeve_p{i+1}",
              "rep_save" : repsect1,
              "title_colbar" :"Force Normal Component " + r"$F_{n}$" + " (N)",
      }
      traj.plt_heat_circ(df.iloc[ind],**kw)
      
      # heatmap wear power :
      kw = {
             "numx"   : nbsect_ad,
             "numy"   : nbz_ad,
             "colx"   : "th",
             "coly"   : colz,
             "colval" : "pusure_ccone",
             "agreg"  : "mean",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
          #    "ylim"   : [-1.e-4,1.e-4],
             "zlim"   : 1.e-3,
          }
      heatmap_WPad_mean = traj.heatmap(df.iloc[ind],**kw)
      
      kw = {
             "numx"   : nbsect_ad,
             "numy"   : nbz_ad,
             "colx"   : "th",
             "coly"   : colz,
             "colval" : "pusure_ccone",
             "agreg"  : "sum",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
          #    "ylim"   : [-1.e-4,1.e-4],
             "zlim"   : 1.e-3,
          }
      heatmap_WPad_sum = traj.heatmap(df.iloc[ind],**kw)
      # plot wear power adapter :
      kw = {
             "heatmap" : heatmap_WPad_sum, 
             "labelx"   : "Angular location " + r"$(\degree)$",
             "labely"   : "Elevation " + r"$z$" + " (m)",
             "title1"   : "heatmap_wearpower_sum",
             "title_save"   : f"wearpower_ad_sum_p{i+1}",
             "rep_save" : repsect1,
             "colval" : "pusure_ccone",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
             "title_colbar"   : "Wear Power (W)",
             "cmap"   : "inferno",
          }
      traj.plt_heat(df.iloc[ind],**kw)
      
      kw = {
             "numx"   : nbsect_ad,
             "numy"   : nbz_ad,
             "colx"   : "th",
             "coly"   : colz,
             "colval" : "Wener",
             "agreg"  : "sum",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
          #    "ylim"   : [-1.e-4,1.e-4],
             "zlim"   : 1.e-3,
          }
      heatmap_Wener_sum = traj.heatmap(df.iloc[ind],**kw)
      # plot wear power adapter :
      kw = {
             "heatmap" : heatmap_Wener_sum, 
             "labelx"   : "Angular location " + r"$(\degree)$",
             "labely"   : "Elevation " + r"$z$" + " (m)",
             "title1"   : "heatmap_wearenergy_sum",
             "title_save" : f"wearener_ad_sum_p{i+1}",
             "rep_save"   : repsect1,
             "colval" : "Wener",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
             "title_colbar"   : "Wear Energy (J)",
             "cmap"   : "inferno",
          }
      traj.plt_heat(df.iloc[ind],**kw)
      # heatmap thmax :
      kw = {
             "numx"   : nbsect_ad,
             "numy"   : nbz_ad,
             "colx"   : "th",
             "coly"   : colz,
             "colval" : "THMAX",
             "agreg"  : "mean",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
          #    "ylim"   : [-1.e-4,1.e-4],
          }
      heatmap_WPad = traj.heatmap(df.iloc[ind],**kw)
      # plot wear power adapter :
      kw = {
             "heatmap" : heatmap_WPad, 
             "labelx"   : "Angular location " + r"$(\degree)$",
             "labely"   : "Elevation " + r"$z$" + " (m)",
             "title1"   : "heatmap_thmax",
             "title_save" : f"thmax_ad_p{i+1}",
             "rep_save"   : repsect1,
             "colval" : "THMAX",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
             "title_colbar"   : "Contact Angular Width " + r"$\theta_{max} \degree$",
             "cmap"   : "inferno",
          }
      traj.plt_heat(df.iloc[ind],**kw)
      
      # heatmap Fn :
      kw = {
             "numx"   : nbsect_ad,
             "numy"   : nbz_ad,
             "colx"   : "th",
             "coly"   : colz,
             "colval" : "FN_CCONE",
             "agreg"  : "mean",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
          #    "ylim"   : [-1.e-4,1.e-4],
          }
      heatmap_WPad = traj.heatmap(df.iloc[ind],**kw)
      # plot wear power adapter :
      kw = {
             "heatmap" : heatmap_WPad, 
             "labelx"   : "Angular location " + r"$(\degree)$",
             "labely"   : "Elevation " + r"$z$" + " (m)",
             "title1"   : "heatmap_FN",
             "title_save" : f"FN_ad_p{i+1}",
             "rep_save"   : repsect1,
             "colval" : "FN_CCONE",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
             "title_colbar"   : "Force Normal Component " + r"$F_{n}$" + " (N)",
             "cmap"   : "inferno",
          }
      traj.plt_heat(df.iloc[ind],**kw)
      
      # plot dimp adapter :
      kw = {
             "numx"   : nbsect_ad,
             "numy"   : nbz_ad,
             "colx"   : "th",
             "coly"   : colz,
             "colval" : "DIMP",
             "agreg"  : "mean",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
          #    "ylim"   : [-1.e-4,1.e-4],
          }
      heatmap_WPad = traj.heatmap(df.iloc[ind],**kw)
      kw = {
             "heatmap" : heatmap_WPad, 
             "labelx"   : r"$\delta_{imp}$" + " (m)",
             "labely"   : "Elevation " + r"$z$" + " (m)",
             "title1"   : "heatmap_dimp",
             "title_save" : f"dimp_ad_p{i+1}",
             "rep_save"   : repsect1,
             "colval" : "DIMP",
             "xlim"   : [-180.,180.],
             "ylim"   : [1.1*ymin,1.1*ymax],
             "title_colbar"   : "Penetration " + r"$\delta_{imp}$" + " (m)",
             "cmap"   : "inferno",
          }
      traj.plt_heat(df.iloc[ind],**kw)

#%%
#%%
if plotpchoc:
    repsect1 = f"{rep_save}developpee/pchoc/"
    if not os.path.exists(repsect1):
      os.makedirs(repsect1)
      print(f"FOLDER : {repsect1} created.")
    else:
      print(f"FOLDER : {repsect1} already exists.")

    dtsort = df['t'].iloc[1] - df['t'].iloc[0]
    discr = 1.e-3
    ndiscr = int(discr/dtsort)
    df = df.iloc[::ndiscr]
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
        "msize" : 1,
    }
    traj.scat3d_pchoc(df.loc[indchoc], **kwargs1)
    if lraidtimo:
        kwargs1 = {
            "tile1": "point d'impact" + "\n",
            "tile_save": "ceadpiA",
            "colx": colx,
            "coly": coly,
            "colz": colz,
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
#%%
sys.exit()

#%% truc qui marchent transformes en fonction : 
# Define the desired number of bins for X and Y
num_th = 72
num_z = 40

# Assign each point to a specific cell
df1['th_bin'] = pd.cut(df1['th'], bins=np.linspace(-180., 180., num_th + 1), labels=False)
df1['Z_bin'] = pd.cut(df1['uzpisc'], bins=np.linspace(-1.e-4, 1.e-4, num_z + 1), labels=False)

# Create a new column 'Cell' to represent the cell for each point
df1['Cell'] = list(zip(df1['th_bin'], df1['Z_bin']))

# Calculate the mean value for each cell
heatmap_data = df1.groupby('Cell')['pusure_ccone'].mean().reset_index()

# Create a MultiIndex for the heatmap
heatmap_data.set_index('Cell', inplace=True)

# Create a grid with all possible cell combinations
all_cells = pd.MultiIndex.from_product([range(num_th), range(num_z)], names=['th_bin', 'Z_bin'])
#
complete_grid = pd.DataFrame(index=all_cells)

# Merge the complete grid with the original heatmap_data
heatmap_data.index = pd.MultiIndex.from_tuples(heatmap_data.index, names=['th_bin', 'Z_bin'])

# Merge the complete grid with the original heatmap_data
heatmap_data_complete = pd.merge(complete_grid, heatmap_data, how='left', left_index=True, right_index=True)

# Reshape the heatmap_matrix into a 2D array

heatmap_matrix_2d = heatmap_data_complete['pusure_ccone'].values.reshape(num_th, num_z)
#%%
title1 = "heatmap_wearpower \n"
labelx = "Angular location " + r"$(\degree)$"
labely = "Elevation " + r"$z$" + " (m)"
# newxticks = np.linspace(-180.,180.,int(num_th)+1)
# newyticks = -np.linspace(-1.e-4,1.e-4,int(num_z)+1)
xticks = np.arange(-180., 200., step=20)
yticks = np.arange(-1.e-4, 1.1e-4, step=1.e-5)
# Plot the heatmap with rectangles
f = plt.figure(figsize=(10, 8),dpi=600)
gs = gridspec.GridSpec(1,2, width_ratios=[10,0.5])
# 1st subplot : 
plt.subplot(gs[0])
axes = f.gca()
axes.set_title(title1)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))
axes.xaxis.set_major_formatter(formatter)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
axes.yaxis.set_major_formatter(formatter)

axes.set_xlabel(labelx,fontsize=12)
axes.set_ylabel(labely,fontsize=12)
# sns.heatmap(np.transpose(heatmap_matrix_2d), cmap='viridis', annot=False, fmt='.2f', cbar_kws={'label': 'Mean Value'})
# sns.heatmap(np.transpose(heatmap_matrix_2d), 
#             cmap='inferno', annot=False, 
#             fmt='.2f', 
#             cbar_kws={'label': 'Mean Wear Power (W)'},
#             xticklabels = xticks,
#             yticklabels = yticks,
#             )
# plt.xticks(rotation=0, ha='right')
# plt.title(title1)
# plt.xlabel('X-axis Cells')
# plt.ylabel('Y-axis Cells')

# axes.pcolormesh(np.transpose(heatmap_matrix_2d), cmap='inferno',shading='auto')
cmapp = 'inferno'
# cmapp = 'viridis',
axes.imshow(np.transpose(heatmap_matrix_2d),
# axes.imshow((heatmap_matrix_2d),
            cmap = cmapp,
            interpolation='nearest', 
            aspect='auto', 
            extent=[1.*min(xticks),
                    1.*max(xticks),
                    1.*min(yticks),
                    1.*max(yticks)])
axes.set_xticks(xticks)

# axes.set_aspect('equal')
# axes.set_xlim(min(xticks), max(xticks))
# axes.set_ylim(min(yticks), max(yticks))
# axes.set_xticks(xticks)
# axes.set_yticks(yticks)

# heatmap_data = df1.pivot(index='uzpisc', columns='th', values='pusure_ccone')
# heatmap = axes.pcolormesh(heatmap_data.columns, heatmap_data.index, heatmap_data, cmap="inferno", shading='auto')
# axes.set_yticks(yticks)

plt.subplot(gs[1])
plt.ticklabel_format(useOffset=False, style='plain', axis='both')
axes=f.gca()
axes.ticklabel_format(axis='y', style='plain', useOffset=False)
varcol = df['pusure_ccone'].drop_duplicates().array.astype(float)
norm = mpl.colors.Normalize(vmin=np.min(varcol), vmax=np.max(varcol))
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmapp),
             cax=axes, orientation='vertical', label="Wear Power (W)") 
cbar.set_label(label="Wear Power (W)",size=12) 
cbar.formatter.set_useOffset(False)
# cbar.set_label(fontsize=12)

f.tight_layout(pad=0.5)

plt.show()

#%% colored angular sectors :

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)
ax.grid(False)
# Plot each sector and fill with mean value color
for sector, mean_value in mean_values.iteritems():
    start_angle = (sector * 2)
    end_angle = ((sector + 1) * 2)
    # wedge = Wedge((0, 0), df['pusure_ccone'].max(), start_angle, end_angle, facecolor=plt.cm.inferno(mean_value))
    wedge = Wedge((0, 0), np.max(mean_values), start_angle, end_angle, facecolor=plt.cm.inferno(mean_value/np.max(mean_values)))
    # wedge = Wedge((0, 0), np.max(mean_values), start_angle, end_angle, facecolor=plt.cm.inferno(mean_value/np.max(mean_values)))
    ax.add_patch(wedge)

# Remove legend inside the circle
ax.legend().set_visible(False)
ax.set_yticklabels([])

plt.show()

#%% plot thermal sleeve works

# Bin the angles into sectors (every five degrees)
df['Sector'] = pd.cut(df['thm'], bins=np.arange(-180, 185, 2), labels=False, right=False)

# Calculate the mean value for each sector
mean_values = df.groupby('Sector')['pusure_ccone'].mean()

#
theta = np.radians(np.arange(-180, 185, 2))

all_cells = pd.Index(range(len(theta)), names=['Sector'])
complete_grid = pd.DataFrame(index=all_cells)

mean_values_complete = pd.merge(complete_grid, mean_values, how='left', left_index=True, right_index=True)

# Create circular plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Plot the heatmap
circular_plot = ax.scatter(theta, mean_values_complete, marker='o', linestyle='-',s=2)

ax.legend().set_visible(False)
ax.set_yticklabels([])

# Add labels, title, etc., as needed
ax.set_rlabel_position(90)
ax.set_title('Circular Heatmap of Mean Values', pad=20)

# Show the plot
plt.show()

#%%
title_col = "Wear Power (W)"
title_save = "WP_sleeve"
rep_save = repsect1
# # Convert Cartesian coordinates to polar coordinates
# df['angle'] = np.arctan2(df['y'], df['x']) * (180 / np.pi)  # Convert radians to degrees
df['thm'] = (df['thm'] + 360) % 360

# Create a polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},dpi=600)

# Define the number of sectors
num_sectors = 72  # 360 degrees / 5 degrees per sector

# Calculate mean values for each sector
mean_values = []
for i in range(num_sectors):
    lower_bound = i * (360 / num_sectors)
    upper_bound = (i + 1) * (360 / num_sectors)
    sector_points = df[(df['thm'] >= lower_bound) & (df['thm'] < upper_bound)]
    if not sector_points.empty:
        mean_value = sector_points['pusure_ccone'].mean()
        mean_values.append(mean_value)
    else:
        mean_values.append(np.nan)

# Plot the colored sectors
theta = np.linspace(0, 2*np.pi, num_sectors, endpoint=False)
colors = plt.cm.inferno(mean_values/np.max(mean_values))  # You can use other colormaps
bars = ax.bar(theta, mean_values, width=(2*np.pi)/num_sectors, align="center", color=colors)

# Remove legend inside the circle
ax.legend().set_visible(False)
ax.set_yticklabels([])

# Set the aspect ratio of the plot to be equal
ax.set_aspect("equal")

# sm = ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=np.nanmin(mean_values), vmax=np.nanmax(mean_values)))
sm = ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label(label=title_col,size=12) 
cbar.formatter.set_useOffset(False)

fig.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")

plt.show()