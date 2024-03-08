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
import mplcursors
import matplotlib.cm as cm

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [20, -50]
# %% quel type de modele ?
lraidtimo = True
lraidiss = True
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
lxp = False
linert = True
# manchette bloquee ? 
lbloq = False
# si oui quelle epaisseur pour les ressorts a lame ?
lbfin = False
lbepai = True

lamode = True
lpion = False
lpcirc = True
# Fext = 79.44
Fext = 35
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
mu = 0.6
xi = 0.05
spinini = 0.
dte = 5.e-6
h_lam = 50.e-3
lspring = 45.e-2

hlstr = int(h_lam*1.e3)
lspringstr = int(lspring*1.e2)
vlostr = int(-np.log10(vlimoden))
# dtstr = int(-np.log10(dte))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_lspr_{lspringstr}'

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
rep_save = f"./fig/{namerep}/"

if (lbloq):
    # manchad bloquee bfin :
    if lbfin:
        namerep = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/bfin/manchadela_pions_1/'
        rep_save = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/fig/manchadela_bloquee/bfin/'

    # manchad bloquee bepai :
    if lbepai:
        namerep = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/manchadela_pions_1/'
        rep_save = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/fig/manchadela_bloquee/'

    repload = f'{namerep}'

if lxp:
    repload = '/home/matthieu/Documents/EDF/mesures/data/donneesLaser/'
    rep_save = f'/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/fig/XP_fext_{int(Fext)}_spin_{int(spinini)}/'
# manchad bloquee bepai :
# namerep = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/manchadela_pions_1/'

#%%

# namerep = f"manchadela_pions_{slice}"
# repload = f"./pickle/{namerep}/"

# %%

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

# %% lecture du dataframe :
if lxp:
    df = pd.read_pickle(f"{repload}20201127_1350_laser.pickle")
else:
    df = pd.read_pickle(f"{repload}result.pickle")

df = df[['t','uzg_tot_ad','uzpb','uzph','Fext']]
    # on trie et on reindexe :
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
# %% fenetrage en temps : 
t1 = 0.
if (not linert):
    t1 = 0.1 
# t2 = 80. --> donne de bon resultats
t2 =128.
# if lbloq & lbfin:
#     t2 = 40.
# df = df[(df['t']>t1) & (df['t']<=t2)]
# df.reset_index(drop=True,inplace=True)
# %% 100 points par seconde 
    # nsort = 10 et x4 dans dopickle_slices :
# if (not linert):
#     nsort = 40
# if (linert):
#     dte = 1.e-6
#     nsort = 30
#     discr = 3.e-5
# if (lbloq):
#     nsort = 10
#     dte = 1.e-5
discr = 1.e-4
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
ttot = 128.
print(f"tmin = {df.iloc[0]['t']}")
print(f"ttot = {ttot}")
df['freq'] = f1 + ((f2-f1)/ttot)*df['t'] 
print(f"fmin = {df.iloc[0]['freq']}")
print(f"fmax = {df.iloc[-1]['freq']}")
#%%
nt = int(np.floor(np.log(len(df['t']))/np.log(2.)))
indexpsd = df[df.index <= 2**nt].index
print(f"nt = {nt}")

#%%############################################
#           PLOTS : zoom :
###############################################
repsect1 = f"{rep_save}variables_ft/zoom/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

ind1 = df[(df['t']>=19.) & (df['t']<=20.)].index
f = df['freq'].iloc[ind1[0]]
print(f"f = {f}")
kwargs1 = {
    "tile1": f"uygad fload = {f}" + "\n",
    "tile_save": f"zoom_traj2d_uygad_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$u_y(G_{ad})$" + " (m)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

kwargs1 = {
    "tile1": f"fext fload = {f}" + "\n",
    "tile_save": f"zoom_fext_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "Fext",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$F_{ext}$" + " (N)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

#%%
ind1 = df[(df['t']>=39.) & (df['t']<=40.)].index
f = df['freq'].iloc[ind1[0]]
print(f"f = {f}")
kwargs1 = {
    "tile1": f"uygad fload = {f}" + "\n",
    "tile_save": f"zoom_traj2d_uygad_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$u_y(G_{ad})$" + " (m)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

kwargs1 = {
    "tile1": f"fext fload = {f}" + "\n",
    "tile_save": f"zoom_fext_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "Fext",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$F_{ext}$" + " (N)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

#%%
ind1 = df[(df['t']>=55.) & (df['t']<=56.)].index
f = df['freq'].iloc[ind1[0]]
print(f"f = {f}")
kwargs1 = {
    "tile1": f"uygad fload = {f}" + "\n",
    "tile_save": f"zoom_traj2d_uygad_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$u_y(G_{ad})$" + " (m)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

kwargs1 = {
    "tile1": f"fext fload = {f}" + "\n",
    "tile_save": f"zoom_fext_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "Fext",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$F_{ext}$" + " (N)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

#%%
ind1 = df[(df['t']>=99.) & (df['t']<=100.)].index
f = df['freq'].iloc[ind1[0]]
print(f"f = {f}")
kwargs1 = {
    "tile1": f"uygad fload = {f}" + "\n",
    "tile_save": f"zoom_traj2d_uygad_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$u_y(G_{ad})$" + " (m)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

kwargs1 = {
    "tile1": f"fext fload = {f}" + "\n",
    "tile_save": f"zoom_fext_fload{int(f)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "Fext",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$F_{ext}$" + " (N)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)
#%%
ind1 = df[(df['t']>=99.) & (df['t']<=105.)].index
fstart = df['freq'].iloc[ind1[0]]
fe = df['freq'].iloc[ind1[-1]]
print(f"fstart = {fstart}")
print(f"fend = {fe}")
kwargs1 = {
    "tile1": f"long uygad fload = {fstart} - {fe} Hz" + "\n",
    "tile_save": f"longzoom_traj2d_uygad_fs{int(fstart)}_fe{int(fe)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "uzg_tot_ad",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$u_y(G_{ad})$" + " (m)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)

kwargs1 = {
    "tile1": f"long fext = {fstart} - {fe} Hz" + "\n",
    "tile_save": f"longzoom_fext_fs{int(fstart)}_fe{int(fe)}",
    "ind": [ind1],
    "colx": "t",
    "coly": "Fext",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": r"$F_{ext}$" + " (N)",
    "color1": ['black','orange'],
    # "rcirc" : ray_circ,
    # "excent" : excent,
    # "spinz" : spinz,     
    "msize" : 0.1,
    "scatter" : [True,True],
    "endpoint" : [False,False],
    # "arcwidth" : sect_pion_deg,
    # "clmax" : cmax,
}
traj.pltraj2d_ind(df, **kwargs1)
#%%############################################
#           PLOTS : PSD :
###############################################
repsect1 = f"{rep_save}variables_ft/PSD/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
    # pas de nooverlap :
if (lbloq): 
    nfft = 2**(nt)
    # nfft = 2**(nt-5)
    nblocks = (len(df)) / (nfft)
    print(f"nblocks = {nblocks}")
    power, freq = plt.psd(1.e5*df['uzg_tot_ad'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., color=color1[2])

    powerPB, freqPB = plt.psd(1.e5*df['uzpb'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., detrend='linear',color=color1[0])

    powerPH, freqPH = plt.psd(1.e5*df['uzph'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., detrend='linear',color=color1[0])


if (not lbloq):
    # belle allure entre 0 et 30 hz : mais nfft fait la longueur du domaine
    # nfft = 2**(nt-7)
    nfft = 2**(nt)
    # nvrlp = 0.75*nfft
    nvrlp = 0.
    nblocks = (len(df) - nvrlp) / (nfft - nvrlp)
    print(f"nblocks = {nblocks}")
    power, freq = plt.psd((1.e5)*df['uzg_tot_ad'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., detrend='linear',color=color1[0])

    powerPB, freqPB = plt.psd((1.e5)*df['uzpb'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., detrend='linear',color=color1[0])

    powerPH, freqPH = plt.psd((1.e5)*df['uzph'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., detrend='linear',color=color1[0])
    # power, freq = plt.psd(1.e5*df['uzg_tot_ad'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., detrend='linear',color=color1[0],noverlap=nvrlp)

    # power = 10. * np.log10(power)
    # powerPH = 10. * np.log10(powerPH)
    # powerPB = 10. * np.log10(powerPB)

if (lxp):
    # belle allure entre 0 et 30 hz : mais nfft fait la longueur du domaine
    nfft = 2**(nt)
    nvrlp = 0.
    nblocks = (len(df) - nvrlp) / (nfft - nvrlp)
    print(f"nblocks = {nblocks}")
    power, freq = plt.psd(-1.e6*df['uzg_tot_ad'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0.,color=color1[0],noverlap=nvrlp)

lpower = []
lfreq = []
lpowerph = []
lfreqph = []
lpowerpb = []
lfreqpb = []
lind = np.array_split(df.index,20)
for indi in lind: 
    dfi = df.iloc[indi]
    dfi.sort_values(by='t',inplace=True)
    dfi.reset_index(drop=True,inplace=True)
    nt1 = int(np.floor(np.log(len(dfi['t']))/np.log(2.)))
    indexpsd1 = dfi[dfi.index <= 2**nt1].index
    print(f"nt1 = {nt1}")
    nfft1 = 2**(nt1)
    # nfft = 2**(nt-5)
    nblocks = (len(dfi)) / (nfft1)
    print(f"nblocks = {nblocks}")
    poweri, freqi = plt.psd(1.e5*dfi['uzg_tot_ad'].iloc[indexpsd1], NFFT=nfft1, Fs=fs, scale_by_freq=0., color=color1[2])
    powerih, freqih = plt.psd(1.e5*dfi['uzph'].iloc[indexpsd1], NFFT=nfft1, Fs=fs, scale_by_freq=0., color=color1[2])
    powerib, freqib = plt.psd(1.e5*dfi['uzpb'].iloc[indexpsd1], NFFT=nfft1, Fs=fs, scale_by_freq=0., color=color1[2])
    lpower.append(poweri)
    lfreq.append(freqi)
    lpowerph.append(powerih)
    lfreqph.append(freqih)
    lpowerpb.append(powerib)
    lfreqpb.append(freqib)

# power_density = 10. * np.log10(power)
power_density   = power
power_densityPB = powerPB
power_densityPH = powerPH

linteractif = True
if (linteractif):
    fig, ax = plt.subplots()
    ax.plot(freq, power_density, label='Data')
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))
    plt.xlim(0,30)
    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('uygad : Interactive Plot with mplcursors')
    # Show the plot
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(freqPH, power_densityPH, label='Data')
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))
    plt.xlim(0,30)
    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('uyph : Interactive Plot with mplcursors')
    # Show the plot
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(freqPB, power_densityPB, label='Data')
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))
    plt.xlim(0,30)
    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('uypb : Interactive Plot with mplcursors')
    # Show the plot
    plt.show()
#%%
if (linert):
    # pour : 
        # linert = True
        # lamode = False
        # lpion = False
        # lpcirc = True
        # Fext = 193.
        # vlimoden = 1.e-4
        # mu = 0.6
        # xi = 0.01
        # spinini = 0.
        # dte = 1.e-6 
    lanot = [(0.88,80.93),(2.70,33.30),(4.49,23.37),(8.11,8.61),(13.33,8.25)]
    # pour : 
        # linert = True
        # lamode = True
        # amode_m = 0.02
        # amode_ad = 0.02
        # lpion = False
        # lpcirc = True
        # Fext = 193.
        # vlimoden = 1.e-5
        # mu = 0.6
        # xi = 0.05
        # spinini = 0.
        # dte = 1.e-6 
    lanot = [(0.88,67.03),(1.95,36.95),(2.64,24.86),(3.03,19.12),(3.76,13.27),(5.57,6.63)]
if (not linert):
        # linert = False
        # lamode = False
        # lpion = False
        # lpcirc = True
        # Fext = 193.
        # vlimoden = 1.e-4
        # mu = 0.6
        # xi = 0.01
        # spinini = 0.
        # dte = 1.e-6 
    lanot = [(1.07,77.61),(3.12,33.18),(5.27,19.85),(9.52,10.67),(15.58,9.41)]
if (lbloq | lbepai): 
    lanot = [(10.28,41.47),(13.24,30.70)]
    xmax = 30.
    ymax = 60.
if (not lbloq):
    lanot = [(12.54,44.54)]

#%% 
cmap2 = traj.truncate_colormap(cm.inferno,0.,0.85,100)
lmax = [ np.max(powi) for powi in lpower ]
lampl = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpower ]
lmaxph = [ np.max(powi) for powi in lpowerph ]
lamplph = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpowerph ]
lmaxpb = [ np.max(powi) for powi in lpowerpb ]
lamplpb = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpowerpb ]
ymax = 1.1*np.max([np.max(power),np.max(powerPB),np.max(powerPH)])

lanotg = [(14.87,19.69)]
lanotph = [(4.58,6.69),(5.42,9.00),(14.87,13.)]
lanotpb = [(3.07,16.30),(4.58,43.42),(5.42,60.13),(14.87,2.14)]
#%% uygad :
# xmax = 20.
# ymax = 60.
# ymin = -150.
xmax = 30.
ymax = 1.1*np.max(power_density)
ymin = 0.
kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad",
    "x": freq,
    "y": power_density,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[2],
    "annotations": lanotg,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)


cmap2 = traj.truncate_colormap(cm.inferno,0.,0.85,100)
lmax = [ np.max(powi) for powi in lpower ]
lampl = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpower ]
xmax = 30.
ymaxsplit = 1.1*np.max(lmax)
# ymin = -500.
kwargs1 = {
    "tile1": "splir PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad_split",
    "x": lfreq,
    "y": lpower,
    "rep_save": repsect1,
    "label1": [None]*len(lpower),
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": lampl,
    "annotations": [],
    "xmax": xmax,
    "ymax": ymaxsplit,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
#%% PB :
# xmax = 20.
# ymax = 60.
# ymin = -150.
xmax = 30.
ymax = 1.1*np.max(powerPB)
# ymin = -10.
kwargs1 = {
    "tile1": " PSD uy(PB) sleeve = f(freq)" + "\n",
    "tile_save": "PSD_uypb",
    "x": freqPB,
    "y": powerPB,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[2],
    "annotations": lanotpb,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

cmap2 = traj.truncate_colormap(cm.inferno,0.,0.85,100)
lmax = [ np.max(powi) for powi in lpowerpb ]
lampl = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpowerpb ]

xmax = 30.
ymaxsplit = 1.1*np.max(lmaxpb)
# ymin = -500.
kwargs1 = {
    "tile1": "split PSD uy(PB) sleeve = f(freq)" + "\n",
    "tile_save": "PSD_uypb_split",
    "x": lfreqpb,
    "y": lpowerpb,
    "rep_save": repsect1,
    "label1": [None]*len(lpower),
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": lampl,
    "annotations": [],
    "xmax": xmax,
    "ymax": ymaxsplit,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
#%% PH :
# xmax = 20.
# ymax = 60.
# ymin = -150.
xmax = 30.
ymax = 1.1*np.max(powerPH)
# ymin = -10.
kwargs1 = {
    "tile1": " PSD uy(PH) sleeve = f(freq)" + "\n",
    "tile_save": "PSD_uyph",
    "x": freqPH,
    "y": powerPH,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[2],
    "annotations": lanotph,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

cmap2 = traj.truncate_colormap(cm.inferno,0.,0.85,100)
lmax = [ np.max(powi) for powi in lpowerph ]
lampl = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpowerph ]

xmax = 30.
ymaxsplit = 1.1*np.max(lmaxph)
# ymin = -500.
kwargs1 = {
    "tile1": "split PSD uy(PH) sleeve = f(freq)" + "\n",
    "tile_save": "PSD_uyph_split",
    "x": lfreqph,
    "y": lpowerph,
    "rep_save": repsect1,
    "label1": [None]*len(lpower),
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": lampl,
    "annotations": [],
    "xmax": xmax,
    "ymax": ymaxsplit,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
# %%
