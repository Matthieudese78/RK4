#!/bin/python3
#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import os
#
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import tikzplotlib as tikz

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [20, -50]
resolution = 600
ncurve_default = 5
ndf_default = 3
defkwargs = {
    "title1": "plotraj3d_default_title",
    "title_save": "plotraj3d_default_title",
    "colx": "uxplam_h",
    "coly": "uyplam_h",
    "colz": "uzplam_h",
    "rep_save": "~/default_graphs/",
    "label1": [f"curve_{i}" for i in np.arange(ncurve_default)],
    "labelx": r"$\mathbf{X}$",
    "labely": r"$\mathbf{Y}$",
    "labelz": r"$\mathbf{Z}$",
    "color1": ["blue" for i in np.arange(ncurve_default)],
    "view": view,
    "linestyle1": [["solid", "dashdot"][i % 2] for i in np.arange(ndf_default)],
    "lslabel": [f"$defleg_{i}$" for i in np.arange(ndf_default)],
    "title_ls": "linestyle def_title :",
    "title_col": "colors def_title :",
    "title_both": "colors def_both :",
    "leg_col": False,
    "leg_ls": False,
    "leg_both": False,
    "loc_ls": "lower left",
    "loc_col": "lower right",
    "loc_both": "upper center",
    "lind": np.arange(1, 10, 1),
    "loc_leg": "upper right",
    "mtype": 'o',
    "msize": 4,
    "colcol": 'defcol',
    "ampl": 200.,
    "rcirc" : 1.,
    "rcircdot" : 0.5,
    "excent" : 0.25,
    "spinz" : 0. 
}
# %% Scripts :
icas1 = 4
lialgo = [1,2,3]
#%%
if (icas1==1):
  script1 = f"fast_top"
  x1 = 19 
  x2 = 15 
  x3 = 15 
  lnconv = [20,15,15]
if (icas1==2):
  script1 = f"slow_top"
  x1 = 17 
  x2 = 16 
  x3 = 8 
  lnconv = [17,16,8]
if (icas1==3):
  script1 = f"fsb"
  x1 = 6 
  x2 = 4 
  x3 = 5
  lnconv = [6,4,5]
if (icas1==4):
  script1 = f"bt"
  x1 = 14 
  x2 = 11 
  x3 = 10
  lnconv = [14,11,10]

if (icas1==5):
  script1 = f"cb"
  x1 = 14 
  x2 = 10 
  x3 = 12
  lnconv = [14,7,7]
#   h1 = df[]
lnconvtot = [[20,15,15],[17,16,8],[6,4,5],[14,11,10],[14,7,7]]
# lindalgo1 = [ df[(df['ialgo']==ialgi) & (df['icas']==icas1)].index for i,ialgi in enumerate(lialgo) ] 

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

#%% poutr un algo : 
lindalgo1 = [ df[(df['ialgo']==ialgi) & (df['icas']==icas1)].index for i,ialgi in enumerate(lialgo) ] 
lind1 = [ df.iloc[indi]['n'].drop_duplicates().index for i,indi in enumerate(lindalgo1)]

ln1  = [ df.iloc[indi]['n'] for i,indi in enumerate(lind1) ]
lct1 = [ df.iloc[indi]['ctime'] for i,indi in enumerate(lind1) ]

# #%%
# lcas = [1,2,3,4,5]
# lindconv = [[ df[(df['ialgo']==ialgi) & (df['icas']==icasj) & (df['n']==lnconvtot[j][i])].index for i,ialgi in enumerate(lialgo) ] for j,icasj in enumerate(lcas) ]  
# #%%
# lctconv = [ [ df.iloc[indi]['ctime'].drop_duplicates() for i,indi in enumerate(lindconv[i]) ] for i,icasi in enumerate(lcas) ] 
lctconv = [ df[(df['icas']==icas1) & (df['ialgo']==ialgi) & (df['n']==lnconv[i])]['ctime'].drop_duplicates() for i,ialgi in enumerate(lialgo) ]
#%%############################################
#           PLOTS : temps de calcul en fonction d :
###############################################
repsect1 = f"{rep_save}ctime_falgo/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#
kwargs1 = {
    "tile1": f"{script1} ctime = f(n), for all algos" + "\n",
    "tile_save": "ctime_log2",
    "ind" : lind1,
    "colx": "n",
    "coly": "ctime",
    "rep_save": repsect1,
    "label1": ["SW","NMB","RK4"],
    "labelx": r"$\log_2(h)$",
    "labely": "Computation time (ms)",
    "color1": color1,
    "msize": 8,
    "loc_leg": "lower right",
}

kwargs1 = defkwargs | kwargs1

title1 = kwargs1["tile1"]
title_save = kwargs1["tile_save"]
ind = kwargs1["ind"]
colx = kwargs1["colx"]
coly = kwargs1["coly"]
rep_save = kwargs1["rep_save"]
label1 = kwargs1["label1"]
labelx = kwargs1["labelx"]
labely = kwargs1["labely"]
color1 = kwargs1["color1"]
loc_leg = kwargs1["loc_leg"]
mp = kwargs1["mtype"]
sp = kwargs1["msize"]

f = plt.figure(figsize=(8,6), dpi=resolution)
axes = f.gca()

axes.set_title(title1)
if type(ind) == list:
    x_data = [df.iloc[indi][colx] for indi in ind]
    y_data = [df.iloc[indi][coly] for indi in ind]
    # h1 = [df.iloc[indi][df.iloc[indi][colx]==x1][coly] for indi in ind]
    # h2 = [df.iloc[indi][df.iloc[indi][colx]==x2][coly] for indi in ind]
    # h3 = [df.iloc[indi][df.iloc[indi][colx]==x3][coly] for indi in ind]
    
else:
    x_data = df[ind][colx]
    y_data = df[ind][coly]
# plt.xscale('log',base=2)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
axes.xaxis.set_major_formatter(formatter)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
axes.yaxis.set_major_formatter(formatter)
# axes.axis("equal")
plt.yscale('log',base=2)
if type(ind) != list:
    axes.scatter(x_data, y_data, label=label1, marker='o', s=sp, color=color1)
else:
    [
        axes.scatter(
            xi, y_data[i], label=label1[i], marker='o', s=sp, color=color1[i]
        )
        for i, xi in enumerate(x_data)
    ]
    [ plt.axhline(y=lctconv[i].iloc[0], color=color1[i], linestyle='--',label=f'{label1[i]} converged') for i,ialgi in enumerate(lialgo)]

axes.set_facecolor("white")
axes.grid(True)

plt.legend(loc=loc_leg,facecolor='white')

axes.set_xlabel(labelx)
axes.set_ylabel(labely)
f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
# f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
# tikz.save(rep_save + title_save + ".tex")
plt.close("all")
# traj.scat2d_semilogx_ind(df, **kwargs1)
# %%
