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
import sys

#%% usefull parameters :
color1 = ["red", "green", "blue", "orange", "purple", "pink"]
view = [20, -50]
# %% Scripts :
# cas la lache de la manchette avec juste le poids :
# namerep = "manchadela_weight"
# namerep = "manchadela_RSG"
# namerep = "manchadela_RSG_conefixe"
linert = True
# manchette bloquee ? 
lbloq = True
lfree = True
lxp   = True
# si oui quelle epaisseur pour les ressorts a lame ?
lbfin = False
lbepai = True
lamode = True
lpion = False
lpcirc = True
Fext = 193.
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
mu = 0.6
xi = 0.05
spinini = 0.
dte = 4.e-6
h_lam = 40.e-3
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

repload_free = f'./pickle/{namerep}/'

    # repsave for all :
repsave = f"./fig/{namerep}/"

    # manchad bloquee bepai :
if lbloq:
    namerep = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/manchadela_pions_1/'
    # repsave_bloq = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/fig/manchadela_bloquee/'
    repload_bloq = f'{namerep}'
    # xp : 
repload_xp = '/home/matthieu/Documents/EDF/mesures/data/donneesLaser/'
# repsave_xp = f'/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/fig/XP_fext_{int(Fext)}_spin_{int(spinini)}/'

# manchad bloquee bepai :
# namerep = '/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/manchadela_pions_1/'

#%%

# namerep = f"manchadela_pions_{slice}"
# repload = f"./pickle/{namerep}/"

# %%

if not os.path.exists(repsave):
    os.makedirs(repsave)
    print(f"FOLDER : {repsave} created.")
else:
    print(f"FOLDER : {repsave} already exists.")

#%% extraction xp :
if lxp:
    df_xp = pd.read_pickle(f"{repload_xp}20201127_1350_laser.pickle")
    df_xp = pd.DataFrame(df_xp)
    ltags=['L1','L2','L3','L5','L6','L7','L8','L9','L10']
    for tag in ltags :
        df_xp[tag] = df_xp[tag] - np.mean(df_xp[tag])

    pointH  = np.array([0,0,878])
    pointPH = np.array([0,0,697])
    pointPB = np.array([0,0,304])
    pointB  = np.array([0,0,189])
    pL1 = np.array([0,-15,715])
    pL2 = np.array([0,-100,224])
    pL3 = np.array([0,100,224])
    pL4 = np.array([0,0,0])
    pL5 = np.array([0,0,986])
    pL6 = np.array([0,0,985])
    pL7 = np.array([0,0,-551])
    pL8 = np.array([0,0,-547])
    pL9 = np.array([0,0,139])
    pL10= np.array([0,0,135])

    df_xp['xA']      = (pL1[2]*(df_xp['L2']+df_xp['L3'])-2*pL2[2]*df_xp['L1'])/(2*pL1[2]-2*pL2[2])
    df_xp['tyA']     = (df_xp['L1']-df_xp['xA'])/pL1[2]
    df_xp['tzA']     = (df_xp['L2']-df_xp['L3'])/(2*pL3[1])
    df_xp['xM']      = (-pL10[2]*df_xp['L5']+pL5[2]*df_xp['L9'])/(pL5[2]-pL9[2])
    df_xp['yM']      = (-pL9[2]*df_xp['L6']+pL6[2]*df_xp['L10'])/(pL6[2]-pL10[2])
    df_xp['tyM']     = (df_xp['L5']-df_xp['xM'])/pL5[2]
    df_xp['txM']     = (df_xp['L10']-df_xp['yM'])/pL10[2]
    df_xp['posHx']   = (df_xp['xM']-df_xp['xA'])+(df_xp['tyM']-df_xp['tyA'])*pointH[2]
    df_xp['posHy']   = df_xp['yM'] + df_xp['txM'] * pointH[2]
    df_xp['posPHx']  = (df_xp['xM']-df_xp['xA'])+(df_xp['tyM']-df_xp['tyA'])*pointPH[2]
    df_xp['posPHy']  = df_xp['yM'] + df_xp['txM'] * pointPH[2]
    df_xp['posPBx']  = (df_xp['xM']-df_xp['xA'])+(df_xp['tyM']-df_xp['tyA'])*pointPB[2]
    df_xp['posPBy']  = df_xp['yM'] + df_xp['txM'] * pointPB[2]

    df_xp.sort_values(by='tL',inplace=True)
    df_xp.reset_index(drop=True,inplace=True)

    trigger_level = 2
    for i in range(len(df_xp['TTL'])) :
        if df_xp['TTL'][i] > trigger_level :
            imin = i
            break
    df_xp = df_xp[df_xp.index>=imin]
    df_xp.reset_index(drop=True,inplace=True)

    df_xp['tL'] = df_xp['tL'] - df_xp['tL'][0]
    df_xp.reset_index(drop=True,inplace=True)

    df_xp = df_xp[['tL','xA','posPHx','posPBx']]

    nt_xp = int(np.floor(np.log(len(df_xp['tL']))/np.log(2.)))
    indexpsd_xp = df_xp[df_xp.index <= 2**nt_xp].index

    dt_xp = df_xp['tL'].iloc[1] - df_xp['tL'].iloc[0] 
    fs_xp = 1/dt_xp
    print(f"dt_xp = {dt_xp}")
    print(f"fs_xp = {fs_xp}")

# %% lecture du dataframe :
if lbloq:
    df_bloq = pd.read_pickle(f"{repload_bloq}result.pickle")
    df_bloq = df_bloq[['t','uzg_tot_ad','uzph','uzpb']]
    df_bloq.sort_values(by='t',inplace=True)
    df_bloq.reset_index(drop=True,inplace=True)
if lfree:
    df_free = pd.read_pickle(f"{repload_free}result.pickle")
    df_free = df_free[['t','uzg_tot_ad','uzph','uzpb']]
    df_free.sort_values(by='t',inplace=True)
    df_free.reset_index(drop=True,inplace=True)

# %% fenetrage en temps : 
    # xp : 
    # num : 
if lbloq:
    discr = 1.e-4
    dtsort_bloq = df_bloq.iloc[1]['t'] - df_bloq.iloc[0]['t']
    ndiscr_bloq = int(discr/(dtsort_bloq))
    df_bloq = df_bloq.iloc[::ndiscr_bloq]
    df_bloq.reset_index(drop=True,inplace=True)
    dt_bloq = df_bloq['t'].iloc[1] - df_bloq['t'].iloc[0] 
    fs_bloq = 1/dt_bloq
    nt_bloq = int(np.floor(np.log(len(df_bloq['t']))/np.log(2.)))
    indexpsd_bloq = df_bloq[df_bloq.index <= 2**nt_bloq].index
    print(f"dt_bloq = {dt_bloq}")
    print(f"fs_bloq = {fs_bloq}")

if lfree:
    discr = 1.2e-4
    dtsort_free = df_free.iloc[1]['t'] - df_free.iloc[0]['t']
    ndiscr_free = int(discr/(dtsort_free))
    df_free = df_free.iloc[::ndiscr_free]
    df_free.reset_index(drop=True,inplace=True)

    dt_free = df_free['t'].iloc[1] - df_free['t'].iloc[0] 
    fs_free = 1/dt_free

    nt_free = int(np.floor(np.log(len(df_free['t']))/np.log(2.)))
    indexpsd_free = df_free[df_free.index <= 2**nt_free].index
    print(f"dt_free = {dt_free}")
    print(f"fs_free = {fs_free}")
# %% frequency = f(t) : 
f1 = 2.
f2 = 20.
ttot = 128.

#%%############################################
#           PLOTS : PSD :
###############################################
repsect1 = f"{repsave}PSD_all/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
    # pas de nooverlap :
if (lbloq): 
    nfft = 2**(nt_bloq)
    # nfft = 2**(nt-5)
    nblocks = (len(df_bloq)) / (nfft)
    print(f"bloquee : nblocks = {nblocks}")
    power_bloq, freq_bloq = plt.psd(-1.e6*df_bloq['uzg_tot_ad'].iloc[indexpsd_bloq], NFFT=nfft, Fs=fs_bloq, scale_by_freq=0., color=color1[1])
    power_density_bloq = 10. * np.log10(power_bloq)

if (lfree):
    # belle allure entre 0 et 30 hz : mais nfft fait la longueur du domaine
    # nfft = 2**(nt-7)
    nfft = 2**(nt_free)
    nblocks = (len(df_free)) / (nfft)
    print(f"free : nblocks = {nblocks}")
    # nvrlp = 0.75*nfft
    nvrlp = 0.
    nblocks = (len(df_free) - nvrlp) / (nfft - nvrlp)
    print(f"nblocks = {nblocks}")
    power_free, freq_free = plt.psd(-1.e6*df_free['uzg_tot_ad'].iloc[indexpsd_free], NFFT=nfft, Fs=fs_free, scale_by_freq=0., detrend='linear',color=color1[2],noverlap=nvrlp)
    power_density_free = 10. * np.log10(power_free)

if (lxp):
    # belle allure entre 0 et 30 hz : mais nfft fait la longueur du domaine
    nfft = 2**(nt_xp)
    nvrlp = 0.
    nblocks = (len(df_xp) - nvrlp) / (nfft - nvrlp)
    print(f"nblocks = {nblocks}")
    power_xp, freq_xp = plt.psd(1.e6*df_xp['xA'].iloc[indexpsd_xp], NFFT=nfft, Fs=fs_xp, scale_by_freq=0.,color=color1[0],noverlap=nvrlp)
    power_density_xp = 10. * np.log10(power_xp)

plt.close('all')

linteractif = True
if (linteractif):
    fig, ax = plt.subplots()
    if lbloq:
        ax.plot(freq_bloq, power_density_bloq, label='bloq')
    if lfree:
        ax.plot(freq_free, power_density_free, label='free')
    if lxp:
        ax.plot(freq_xp, power_density_xp, label='xp')
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))
    plt.xlim(0,30)
    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Interactive Plot with mplcursors')
    
    # Show the plot
    plt.show()

sys.exit()
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
    "annotations": lanot,
    "xmax": xmax,
    "ymax": ymax,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
# %%
