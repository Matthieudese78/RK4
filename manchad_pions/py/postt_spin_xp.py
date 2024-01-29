#!/bin/python3
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
from matplotlib import ticker
#%%
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def unwrap(signal,ecart) :
    buffer=np.zeros(1000)+signal[0]
    vafter = np.zeros(len(signal))
    for i in range(len(signal)) :
        e=signal[i]
        m=np.mean(buffer)
        if abs(e-m)<ecart :
            vafter[i]=e
        elif abs(e+18-m)<ecart :
            vafter[i]=e+18
        elif abs(e-18-m)<ecart :
            vafter[i]=e-18
        else :
            vafter[i]=m
        if i>1000 :
            buffer=vafter[i-1000:i]
    return vafter

cmap3 = truncate_colormap(mpl.cm.gist_ncar, 0., 0.9, n=100)
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
lplam = False
lplow = False
#%% repload :
repload = '/home/matthieu/Documents/EDF/mesures/data/donneesLaser/'
namerep = f'XP_spins'
# orientation verifiee :
orientation = [0.,30.,60.,90.,120.,150.,180.,-150.,-120.,-90 -60.,-30.,0.]

lcasneuf = [
 '20201127_1350',
 '20201127_1406',
 '20201127_1409',
 '20201127_1413',
 '20201127_1416',
 '20201127_1419',
 '20201127_1422',
 '20201127_1427',
 '20201127_1429',
 '20201127_1436',
 '20201127_1448',
 '20201127_1454',
 '20201127_1457']
# O1 : pion en face du pot :
icas = 0
essai = lcasneuf[icas]
filename = f'{repload}{essai}_laser.pickle'
rep_save = f"./fig/{namerep}/"
if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
df = pd.read_pickle(f"{filename}")
df = pd.DataFrame(df)
df = df[['tL','L4']]
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)
O10 = 84. * np.pi/180.
O1 = O10
if icas < 1 :
    O1=O10
elif icas < 7 :
    # O1=O10
    O1=O10-180
elif icas < 14 :
    O1=O10
#%% unwrap :
df['L4'] = unwrap(df['L4'],1)
#%%
df['spin'] = df['L4']*10. * np.pi/180. - O1 
# equivalent :
# df['spin'] = df['spin'][0] - (df['spin'] - df['spin'][0])
df['spindeg'] = df['spin'] * 180. / np.pi
plt.scatter(df['tL'],df['spindeg'],s=0.1) 
# plt.show()
plt.close('all')
#%%

repsect1 = rep_save
kwargs1 = {
    "tile1": "Spins = f(t)" + "\n",
    "tile_save": "psis_ft",
    "rep_save": repsect1,
    "labelx": r"$t \quad (s)$",
    "labely": r"$\Psi \quad (\degree)$",
}

l0 = []
lcol = []
lcolor = []
l1 = lcasneuf

lw = 0.8  # linewidth
f = plt.figure(figsize=(8, 6), dpi=600)
axes = f.gca()

axes.set_title(kwargs1["tile1"])
axes.set_facecolor("white")
axes.grid(False)
axes.set_xlabel(kwargs1['labelx'],fontsize=12)
axes.set_ylabel(kwargs1['labely'],fontsize=12)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))
axes.xaxis.set_major_formatter(formatter)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))
axes.yaxis.set_major_formatter(formatter)

for icc,ic2 in enumerate(l1): 
    O10 = 84.
    O1 = O10
    if icc < 1 :
        O1=O10
    elif icc < 7 :
        # O1=O10
        O1=O10-180.
    elif icc < 14 :
        O1=O10
    filename = f'{repload}{ic2}_laser.pickle'
    print(f"filename = {filename}")
    df1 = pd.read_pickle(f"{filename}")
    df1 = pd.DataFrame(df1)
    df1 = df1[['tL','L4']]
    df1['L4'] = unwrap(df1['L4'],1)
    df1.sort_values(by='tL',inplace=True)
    df1.reset_index(drop=True,inplace=True)
    colspin = f'spin_O{icc+1}'
    df[colspin] = df1['L4']*10. * np.pi/180. 
    # df[colspin] = df[colspin][0] - (df[colspin] - df[colspin][0])
    # df[f'spin_O{icc+1}'] = df1['L4']*10. * np.pi/180.  
    df[f'{colspin}_deg'] = df[colspin] * 180. / np.pi - O1
    l0.append(df[f"{colspin}_deg"].iloc[0]+180.)
    lcol.append(f"{colspin}_deg")

for icc,ic2 in enumerate(l1): 
    # clr = cmap3(l0[icc]/np.max(l0))
    clr = cmap3(l0[icc]/np.max(l0))
    lcolor.append(clr) 
    plt.scatter(df['tL'],df[f'spin_O{icc+1}_deg'],color=clr,s=0.1)

# plt.show() 
f.savefig(rep_save + kwargs1['tile_save'] + ".png", bbox_inches="tight", format="png")
plt.close('all')

#%%
# repsect1 = rep_save
# kwargs1 = {
#     "tile1": "Spins = f(t)" + "\n",
#     "tile_save": "psis_ft",
#     "colx": ["tL"] * len(colspin),
#     "coly": lcol,
#     "rep_save": repsect1,
#     "label1": [None]*len(colspin),
#     "labelx": r"$t \quad (s)$",
#     "labely": r"$\Psi \quad \degree$",
#     "color1": lcolor,
# }
# traj.pltraj2d(df, **kwargs1)
# %%
