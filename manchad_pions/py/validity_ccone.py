
#!/bin/python3
#%%
import numpy as np
# import numpy.linalg as LA
import pandas as pd
import trajectories as traj
# import rotation as rota
# import repchange as rc
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import mplcursors
# import sys
# import seaborn as sns
from matplotlib import ticker
# from matplotlib.colors import Normalize
# from matplotlib.patches import Wedge
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.cm import ScalarMappable
#%%
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
# options de tracer :
cursor = False
line = False
zoom = False
zoomplus = False
# %%
if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{repload}result.pickle")
# df = pd.read_pickle(f"{repload}2048/result.pickle")
#%%
df = df[['t','dt','DIMP','FN_CCONE','RCINC','NUT','pusure_ccone']]
#%%
indchoc = df[df['FN_CCONE'].abs()>0.].index
df = df.iloc[indchoc]
#%%
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
dt = df.iloc[1]['t'] - df.iloc[0]['t'] 
df['DIMP'] = -df['DIMP']*1.e6
df['energy'] = df['pusure_ccone']*dt
# df['RCINC'] = (df['RCINC']-1.)*1.e4
#%% zoom
if zoom:
    rzoom = 1.0004
    indzoom = df[(df['RCINC']<=rzoom)].index
    df = df.iloc[indzoom]
if zoomplus:
    rzmax = 1.00015
    rzmin = 0.99993
    dimax = 3.
    indzoom = df[(df['RCINC']>=rzmin) & (df['RCINC']<=rzmax) & (df['DIMP']<=dimax)].index
    df = df.iloc[indzoom]
#%% line
dint_a = 70.e-3 
d_ext = 63.5*1.e-3 
ray_circ = ((dint_a/2.) - (d_ext/2.))
# sh = 6.666*10e-2
# sm = 2.50*10e-2
# st = 1.111*10e-2
sh = 0.06980525619210386 
sm = 0.02708324165749807 
st = 0.012533086738156748 
# rc = df.sort_values('RCINC')['RCINC'].drop_duplicates().values
rc = np.linspace(1.,np.max(df['RCINC']),100)
x = (rc) - 1.
ym = sm*x
yt = st*x
yh = sh*x
#%%
repsect1 = f"{rep_save}validity_params/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
nbsect_dimp = 500
nbsect_rcinc = 500

xlim = [(1.-1.e-5)*np.min(df['RCINC']),1.00001*np.max(df['RCINC'])]
# xlim = [0.9999*np.min(df['RCINC']),rzoom],
ylim = [0.9999*np.min(df['DIMP']),1.00001*np.max(df['DIMP'])]

kw = {
       "numx"   : nbsect_rcinc,
       "numy"   : nbsect_dimp,
       "colx"   : "RCINC",
       "coly"   : "DIMP",
       "colval" : "energy",
       "agreg"  : "sum",
       "xlim"   : xlim,
       "ylim"   : ylim,
    #    "ylim"   : [-1.e-4,1.e-4],
      #  "zlim"   : 1.e-3,
    }
#%%
num_x = kw['numx']
num_y = kw['numy']
colx = kw['colx'] 
coly = kw['coly'] 
colval = kw['colval'] 
agreg = kw['agreg']
# zlim = kw['zlim']

# df = df[(df[coly] <= np.mean(df[coly])+zlim) & (df[coly] >= np.mean(df[coly])-zlim)]
# Assign each point to a specific cell
df['Xbin'] = pd.cut(df[colx], bins=np.linspace(kw['xlim'][0],kw['xlim'][1], num_x + 1), labels=False)
df['Ybin'] = pd.cut(df[coly], bins=np.linspace(kw['ylim'][0],kw['ylim'][1], num_y + 1), labels=False)
# df['Xbin'] = pd.cut(df[colx], bins=np.linspace(kw['xlim'][0],kw['xlim'][1], num_x ), labels=False)
# df['Ybin'] = pd.cut(df[coly], bins=np.linspace(kw['ylim'][0],kw['ylim'][1], num_y ), labels=False)
    
# Create a new column 'Cell' to represent the cell for each point
df['Cell'] = list(zip(df['Xbin'], df['Ybin']))
#%%    
# Calculate the mean value for each cell
if (agreg=='mean'):
    heatmap_data = df.groupby('Cell')[colval].mean().reset_index()
if (agreg=='sum'):
    heatmap_data = df.groupby('Cell')[colval].sum().reset_index()
# Create a MultiIndex for the heatmap
heatmap_data.set_index('Cell', inplace=True)
# Create a grid with all possible cell combinations
all_cells = pd.MultiIndex.from_product([range(num_x+1), range(num_y+1)], names=['Xbin', 'Ybin'])
#
complete_grid = pd.DataFrame(index=all_cells)
# Merge the complete grid with the original heatmap_data
heatmap_data.index = pd.MultiIndex.from_tuples(heatmap_data.index, names=['Xbin', 'Ybin'])
    
heatmap_data_complete = heatmap_data.combine_first(complete_grid)

heatmap_data_complete = heatmap_data_complete.values.reshape(num_x+1,num_y+1)
#%% energie max et min :
enermax = np.max(heatmap_data['energy'])
enermin = np.min(heatmap_data['energy'])
#%%
titlesave = "wearenergy_params"
if line:
    titlesave = titlesave + "_line"
if zoom:
    titlesave = titlesave + "_zoom"
if zoomplus:
    titlesave = titlesave + "_zoomplus"
kw = {
       "heatmap" : heatmap_data_complete, 
       "labelx"   : r"$\frac{R_{curv}}{R_{circ}}$",
       "labely"   : r"$\delta_{imp} \quad (\mu m)$",
       "title1"   : "heatmap_wearpower_params",
       "title_save"   : titlesave,
       "rep_save"   : repsect1,
       "colval" : "energy",
       "xlim"   : xlim,
       "ylim"   : ylim,
       "title_colbar"   : "Wear Energy (J)",
       "cmap"   : "inferno",
    }


title1 = kw["title1"]
title_save = kw["title_save"]
repsave = kw["rep_save"]
labelx = kw["labelx"]
labely = kw["labely"]
colval = kw["colval"]
title_col = kw["title_colbar"]

# xticks = np.arange(kw['xlim'][0],kw['xlim'][1], step=nbsect_rcinc)
# yticks = np.arange(kw['ylim'][0],kw['ylim'][1], step=nbsect_dimp)
# Plot the heatmap with rectangles
f = plt.figure(figsize=(8, 4),dpi=600)
gs = gridspec.GridSpec(1,2, width_ratios=[10,0.5])
# 1st subplot : 
plt.subplot(gs[0])
axes = f.gca()
axes.set_title(title1 + "\n")
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))
axes.xaxis.set_major_formatter(formatter)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))
axes.yaxis.set_major_formatter(formatter)

axes.set_xlabel(labelx,fontsize=12)
axes.set_ylabel(labely,fontsize=12)
cmapp = kw["cmap"]
axes.imshow(np.transpose(kw['heatmap']),
# axes.imshow((kw['heatmap']),
# axes.imshow((heatmap_matrix_2d),
            cmap = cmapp,
            interpolation='nearest', 
            aspect='auto', 
            origin='lower', 
            extent=[(kw['xlim'][0]),
                    (kw['xlim'][1]),
                    (kw['ylim'][0]),
                    (kw['ylim'][1])])
# axes.set_xlim(xmin=kw['xlim'][0],xmax=kw['xlim'][1])
# axes.set_ylim(ymin=kw['ylim'][0],ymax=kw['ylim'][1])

# axes.set_xticks(xticks)

axes.ticklabel_format(axis='x', style='plain', useOffset=False)

if line:
    plt.plot(1.+x,yt*1e6,color='green',label='transition Hertz-Parabolic')
    plt.legend(loc="upper left")
if zoom:
    axes.set_xlim(xmax=rzoom)
if zoomplus:
    axes.set_xlim(xmax=rzmax)
# axes.plot(1.+x,y1*1.e6,color='black',linestyle='dashed')
# axes.plot(1.+x,y2*1.e6,color='black',linestyle='dashed')
# axes.plot(1.+x,yt*1.e6,color='black',linestyle='dashed')
plt.subplot(gs[1])

plt.ticklabel_format(useOffset=False, style='plain', axis='both')

axes=f.gca()

axes.ticklabel_format(axis='y', style='plain', useOffset=False)

# varcol = df[colval].drop_duplicates().array.astype(float)
varcol = df[colval].drop_duplicates().array.astype(float)
norm = mpl.colors.Normalize(vmin=np.min(np.min(heatmap_data[colval])), vmax=np.max(heatmap_data[colval]))
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmapp),
             cax=axes, orientation='vertical') 
cbar.set_label(label=title_col,size=12) 
cbar.formatter.set_useOffset(False)

f.tight_layout(pad=0.5)
# plt.show()
if cursor:
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.6f}, {sel.target[1]:.3f}"))
    plt.show()
f.savefig(repsave + title_save + ".png", bbox_inches="tight", format="png")

plt.close('all')
# traj.plt_heat(df,**kw)
# %%
