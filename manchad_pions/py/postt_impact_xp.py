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
import scipy
from matplotlib.patches import PathPatch
#%%

def butter_bandstop(center,width, fs, order=5):
    nyq = 0.5 * fs
    normal_leftcutoff = (center-width/2) / nyq
    normal_rightcutoff = (center+width/2) / nyq
    b, a = scipy.signal.butter(order, (normal_leftcutoff,normal_rightcutoff), btype='bandstop', analog=False)
    return b, a
def butter_bandstop_filter(data, center, width, fs, order=5):
    b, a = butter_bandstop(center, width, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y
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
#%% repload :
repload = '/home/matthieu/Documents/EDF/mesures/data/donneesLaser/'
namerep = f'XP_impacts'
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
df = df[['tL','TTL','Force']]
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)
# df = df[df['tL']<=128.]
trigger_level = 2
ltags=['TTL','Force','tL']
for i in range(len(df['TTL'])) :
    if df['TTL'][i] > trigger_level :
        imin = i
        break
df = df[df.index>=imin]
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)

#%%
df['tL'] = df['tL'] - df['tL'][0]
fs = 1./(df.iloc[1]['tL']-df.iloc[0]['tL'])
#%% filter force
forceFiltered1 = butter_bandstop_filter(df['Force'],50,2,fs)
#%%
plt.scatter(df['tL'],df['Force'],s=0.1) 
plt.show()
plt.close('all')
#%%
jeuPB = 2. # pion bas
facteurProminence = 5
prominenceThrPB = jeuPB/facteurProminence
nbseg = 8
indpeaks = []
lwidth = []
A0 = []
force_split = np.array_split(df['Force'],nbseg)
t_split = np.array_split(df['tL'],nbseg)
hpeaks = [50.].append([80.]*(nbseg-1))
hpeaks = 100.
seuil = 10.
for i,ti in enumerate(t_split):
    fi = force_split[i]

    # peakspos, _ = scipy.signal.find_peaks(fi,prominence=(prominenceThrPB,None),threshold=seuil)

    # peaksneg, _ = scipy.signal.find_peaks(-fi,prominence=(prominenceThrPB,None),threshold=seuil)

    # peakspos, _ = scipy.signal.find_peaks(fi,prominence=(prominenceThrPB,None),height=hpeaks)

    # peaksneg, _ = scipy.signal.find_peaks(-fi,prominence=(prominenceThrPB,None),height=hpeaks)

    peakspos, _ = scipy.signal.find_peaks(fi,prominence=(prominenceThrPB,None),height=hpeaks,threshold=seuil)

    peaksneg, _ = scipy.signal.find_peaks(-fi,prominence=(prominenceThrPB,None),height=hpeaks,threshold=seuil)

    peaks = np.union1d(peakspos,peaksneg)

    A0.append(np.concatenate(fi[peakspos],-fi[peaksneg]))

    # widths, _, _, _ = scipy.signal.peak_widths(fi, peaks, rel_height=0.5)

    widthspos, _, _, _ = scipy.signal.peak_widths(fi, peakspos, rel_height=0.5)
    widthsneg, _, _, _ = scipy.signal.peak_widths(-fi, peaksneg, rel_height=0.5)

    widths = np.concatenate((widthspos,widthsneg))

    plt.plot(ti.iloc[peaks],fi.iloc[peaks]) 

    plt.hlines(y=fi.iloc[peakspos], xmin=ti.iloc[peakspos] - (widthspos/fs) / 2, xmax=ti.iloc[peakspos] + (widthspos/fs) / 2, color="red", label='Widths')
    plt.hlines(y=fi.iloc[peaksneg], xmin=ti.iloc[peaksneg] - (widthsneg/fs) / 2, xmax=ti.iloc[peaksneg] + (widthsneg/fs) / 2, color="red", label='Widths')

    # plt.scatter(ti.iloc[peaks], widths/fs, s=0.1)

    indpeaks.append(peaks)
    lwidth.append(widths/fs)
    # print(f"tps de choc mini : {np.min(widths/fs)}")
    plt.show()
    plt.close('all')

#%% moyenne 
lnchoc = []
lmoy = []
lsigma = []
[ lnchoc.append(len(lwidth[i])) for i,pi in enumerate(indpeaks) ]
[ lmoy.append(np.mean(lwidth[i])*1.e3) for i,pi in enumerate(indpeaks) ]
[ lsigma.append(np.std(lwidth[i])*1.e3) for i,pi in enumerate(indpeaks) ]

lmoyA0 = []
lsigmaA0 = []
[ lnchoc.append(len(lwidth[i])) for i,pi in enumerate(indpeaks) ]
[ lmoy.append(np.mean(lwidth[i])*1.e3) for i,pi in enumerate(indpeaks) ]
[ lsigma.append(np.std(lwidth[i])*1.e3) for i,pi in enumerate(indpeaks) ]

time_interval = df.iloc[-1]['tL']/(nbseg)
x = np.linspace(0.,df.iloc[-1]['tL']-time_interval,nbseg)


#%% number of impacts
fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
# ax.set_xlim(xmin=0.,xmax=df.iloc[-1]['tL'])
bars = plt.bar(x,lnchoc,width=time_interval,align='edge',color='red',alpha=0.8)
# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
    # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

ax.set_ylabel("Impact Number",fontsize=12)
ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
plt.xticks(rotation=0, ha='right')
plt.show()
title=f"{rep_save}number_impact"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
#%% mean impact time :
fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
bars = plt.bar(x,lmoy,width=time_interval,align='edge',color='green')
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
ax.set_ylabel("Mean Impact Time (ms)",fontsize=12)
ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
plt.xticks(rotation=0, ha='right')

plt.show()

title=f"{rep_save}mean_choc"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 

#%% std deviation :
fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
# ax.scatter(x,lsigma,s=6)
bars = plt.bar(x,lsigma,width=time_interval,align='edge',color='blue')
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
ax.set_ylabel("Impact Time Standard Deviation (ms)",fontsize=12)
ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
plt.xticks(rotation=0, ha='right')

plt.show()

title=f"{rep_save}sigma"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 

#%% amplitude of impacts
fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
# ax.set_xlim(xmin=0.,xmax=df.iloc[-1]['tL'])
bars = plt.bar(x,lnchoc,width=time_interval,align='edge',color='red',alpha=0.8)
# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
    # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

ax.set_ylabel("Impact Number",fontsize=12)
ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
plt.xticks(rotation=0, ha='right')
plt.show()
title=f"{rep_save}number_impact"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
#%%
plt.bar(np.arange(len(indpeaks)),lmoy)
plt.show()
plt.close('all')
plt.bar(np.arange(len(indpeaks)),lsigma)
plt.show()
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
axes.set_xlabel(kwargs1['labelx'])
axes.set_ylabel(kwargs1['labely'])

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
