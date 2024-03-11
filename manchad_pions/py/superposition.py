#!/bin/python3
#%%
import scipy 
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
import rotation as rota
import repchange as rc
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
import mplcursors
import sys
#%%
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap3 = truncate_colormap(mpl.cm.inferno, 0., 0.9, n=100)
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

#%% manchette neuve
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
# 
# %% quel type de modele ?
lraidtimo = False
lplam = False
lplow = False
# lconefixe = True
# %% Scripts :
# cas la lache de la manchette avec juste le poids :
# namerep = "manchadela_weight"
# namerep = "manchadela_RSG"
# namerep = "manchadela_RSG_conefixe"
lpion = False
lpcirc = True

Fext = 79.44
spinini = 0.
# manchette bloquee sinus balaye niveau n1 :
lbloq = True

namerep = f'XP_fext_{int(Fext)}_spin_{int(spinini)}'

if lbloq:
    namerep = f"{namerep}_bloq"

repload = '/home/matthieu/Documents/EDF/mesures/data/donneesLaser/'
orientation = [0.,30.,60.,90.,120.,150.,180.,-30.,-60.,-90 -120.,-150.]
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
 '20201127_1454']
# O1 : pion en face du pot :
icas = 0

O10 = 84. * np.pi/180.
O1 = O10
if icas < 1 :
    O1=O10
elif icas < 7 :
    # O1=O10
    O1=O10-180
elif icas < 14 :
    O1=O10

essai = lcasneuf[icas]

# manchette bloquee dans l'adaptateur :
if (lbloq):
    essai = '20201126_1036'
# O2 : decale de 30 degres :
# essai = '20201127_1406'

filename = f'{repload}{essai}_laser.pickle'

# %% lecture du dataframe :
df = pd.read_pickle(f"{filename}")
df = pd.DataFrame(df)
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)
# %% fenetrage en temps : 
# if (icas==0):
#     t1 = 11. 
# t2 = df['tL'].iloc[-1]
# df = df[(df['tL']>=t1) & (df['tL']<=t2)]

trigger_level = 2
for i in range(len(df['TTL'])) :
    if df['TTL'][i] > trigger_level :
        imin = i
        break
df = df[df.index>=imin]
#%%
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)
#%% on remet l'origine du temps a 0 :
df['tL'] = df['tL'] - df['tL'][0]
nt = int(np.floor(np.log(len(df['tL']))/np.log(2.)))


#%% frequence en fonction du temps :
f1 = 2.
f2 = 20.
ttot = df.iloc[-1]['tL']
df['freq'] = f1 + ((f2-f1)/ttot)*df['tL'] 
#%% frequence d'echantillonnage :
fs = 1./(df['tL'][1]-df['tL'][0])
# %% moyennage :
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

df['L4'] = unwrap(df['L4'],1)

#%% on soustrait la moyenne
ltags=['L1','L2','L3','L5','L6','L7','L8','L9','L10']
for tag in ltags :
    df[tag] = df[tag] - np.mean(df[tag])

#%% SPIN :
# df['spin'] = df['L4']*10. * np.pi/180. - O1 + np.pi
df['spin'] = df['L4']*10. * np.pi/180. - O1 
# equivalent :
df['spin'] = df['spin'][0] - (df['spin'] - df['spin'][0])

df['spindeg'] = df['spin'] * 180. / np.pi

# O2 : decale de 30 degres :
# essai = '20201127_1406'

# def chspindir(df,**kwargs):
#     # return "done"    
#     xval = df[kwargs['col1']]
#     xini = kwargs['valini']
#     return xini - (xval - xini)

# def chspindirdf(df,**kwargs):
#     dict1 = {kwargs['col1'] : df.apply(chspindir,**kwargs,axis=1)}
#     df1 = pd.DataFrame(dict1)
#     print (df1.index)
#     df[kwargs["col1"]] = df1

# k1 = {'col1' : 'spin', 'valini' : df['spin'][0] } 
# chspindirdf(df,**k1) 
#%% deplacements adapter : 
df['xA'] = (pL1[2]*(df['L2']+df['L3'])-2*pL2[2]*df['L1'])/(2*pL1[2]-2*pL2[2])
df['tyA'] = (df['L1']-df['xA'])/pL1[2]
df['tzA'] = (df['L2']-df['L3'])/(2*pL3[1])
# df['rxad'] = (df['L1']-df['uygad'])/pL1[2]
# df['rzad'] = (df['L2']-df['L3'])/(2*pL3[1])

        # depl a partir de haut et medium utilise pour les mesures :
df['xM']  = (-pL10[2]*df['L5']+pL5[2]*df['L9'])/(pL5[2]-pL9[2])
df['yM']  = (-pL9[2]*df['L6']+pL6[2]*df['L10'])/(pL6[2]-pL10[2])
df['tyM'] = (df['L5']-df['xM'])/pL5[2]
df['txM'] = (df['L10']-df['yM'])/pL10[2]

df['posHx']  = (df['xM']-df['xA'])+(df['tyM']-df['tyA'])*pointH[2]
df['posHy']  = df['yM'] + df['txM'] * pointH[2]
df['posPHx']  = (df['xM']-df['xA'])+(df['tyM']-df['tyA'])*pointPH[2]
df['posPHy']  = df['yM'] + df['txM'] * pointPH[2]
# posBx  = (xM-xA)+(tyM-tyA)*pointB[2]
# posBy  = yM + txM * pointB[2]
df['posPBx']  = (df['xM']-df['xA'])+(df['tyM']-df['tyA'])*pointPB[2]
df['posPBy']  = df['yM'] + df['txM'] * pointPB[2]
# position non relative des pions haut et bas :
df['PHx'] = df['posPHx']
df['PBx'] = df['posPBx']
if lbloq:
    df['PHx'] = (df['xM'])+(df['tyM']-df['tyA'])*pointPH[2]
    df['PBx'] = (df['xM'])+(df['tyM']-df['tyA'])*pointPB[2]

df['PHy']  = df['posPHy'] 
df['PBy']  = df['posPBy'] 
#%% on rajoute les coord bidons :
df['yA'] = 0.*df['xA']
df['zA'] = 0.*df['xA']
df['zM'] = 0.*df['xA']
df['posHz'] = 0.*df['xA']
df['posPHz'] = 0.*df['xA']
df['posPBz'] = 0.*df['xA']
df['PHz'] = 0.*df['xA']
df['PBz'] = 0.*df['xA']

#%% on change de repere :
exb = np.array([0.0, -1.0, 0.0])
eyb = np.array([1.0, 0.0, 0.0])
ezb = np.array([0., 0.0, 1.0])
base2 = [exb, eyb, ezb]
# centre du cercle et son vis a vis :
name_cols = ["xA", "yA", "zA"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
name_cols = ["xM", "yM", "zM"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
name_cols = ["posHx", "posHy", "posHz"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
name_cols = ["posPHx", "posPHy", "posPHz"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
name_cols = ["posPBx", "posPBy", "posPBz"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
name_cols = ["PHx", "PHy", "PHz"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
name_cols = ["PBx", "PBy", "PBz"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)

#%% old laser selectionne :
df['yA']      =  1.e-3 * df['yA']
df['posPHxm'] =  1.e-3 * df['posPHx']
df['posPHym'] =  1.e-3 * df['posPHy']
df['posPBxm'] =  1.e-3 * df['posPBx']
df['posPBym'] =  1.e-3 * df['posPBy']
df['posHxm']  =  1.e-3 * df['posHx']
df['posHym']  =  1.e-3 * df['posHy']
df['PHxm']  =  1.e-3 * df['PHx']
df['PHym']  =  1.e-3 * df['PHy']
df['PBxm']  =  1.e-3 * df['PBx']
df['PBym']  =  1.e-3 * df['PBy']
# df['posHrm']  =  1.e-3 * df['posHr']

# df['yA']      =  1. * df['xA']
# df['posPHym'] =  1. * df['posPHx']
# df['posPBym'] =  1. * df['posPBx']
# %% maximum penetration :
penePB = cmax - df['posPBxm'].abs().max()
penePH = cmax - df['posPHxm'].abs().max()
maxdeplPB = df['posPBxm'].abs().max()
maxdeplPH = df['posPHxm'].abs().max()

df = df[['tL','freq','yA','posPHxm','posPHym','posPBxm','posPBym','PHxm','PHym','PBxm','PBym']]
# df = df[['tL','freq','yA','posPHym','posPBym']]
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)

#%% NUM
repload = "/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/manchadela_bloq/"

dfnum = pd.read_pickle(f"{repload}result.pickle")
dfnum = dfnum[['t','uzg_tot_ad','uzpb','uzph','Fext']]
  # on trie et on reindexe :
dfnum.sort_values(by='t',inplace=True)
dfnum.reset_index(drop=True,inplace=True)

  # on ajuste la frequence d'echantillonnage a l'XP :
discr  = df.iloc[1]['tL'] - df.iloc[0]['tL']
dtsort = dfnum.iloc[1]['t'] - dfnum.iloc[0]['t']
ndiscr = max(1,int(discr/(dtsort)))
print(f"ndiscr = {ndiscr}")
dfnum = dfnum.iloc[::ndiscr]
dfnum.reset_index(drop=True,inplace=True)
    # frequence d'echantillonnage de la coimputation :
fsnum = 1./(dfnum['t'][1]-dfnum['t'][0])
ntnum = int(np.floor(np.log(len(dfnum['t']))/np.log(2.)))

f1 = 2.
f2 = 20.
ttot = dfnum.iloc[-1]['t']
dfnum['freqnum'] = f1 + ((f2-f1)/ttot)*dfnum['t'] 
# dfxp = df

df = pd.concat([df,dfnum],axis=1)
    # on vire dfnum
# del dfnum

indexpsd = df[df.index <= 2**nt].index
indexpsdnum = df[df.index <= 2**ntnum].index
# %% numeric
# %% lecture du dataframe :
rep_save = f"./fig/superposition/"

repsect1 = f"{rep_save}variables_ft/"

if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%% courant tension :
    # bruts :
#%%
kwargs1 = {
    "tile1": "uy(G) adapter = f(t)" + "\n",
    "tile_save": "uygad_t_xpnum",
    "colx": ["tL","t"],
    "coly": ["yA","uzg_tot_ad"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_y(G_ad)$"+" (m)",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(G) adapter = f(f)" + "\n",
    "tile_save": "uygad_f_xpnum",
    "colx": ["freq","freqnum"],
    "coly": ["yA","uzg_tot_ad"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    # "labelx": r"$t \quad (s)$",
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$u_y(G_ad)$"+" (m)",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PH) adapter = f(t)" + "\n",
    "tile_save": "uyph_t_xpnum",
    "colx": ["tL","t"],
    "coly": ["PHym","uzph"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_y(P_{pin}^u)$"+" (m)",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PH) adapter = f(f)" + "\n",
    "tile_save": "uyph_f_xpnum",
    "colx": ["freq","freqnum"],
    "coly": ["PHym","uzph"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$u_y(P_{pin}^u)$"+" (m)",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PB) adapter = f(t)" + "\n",
    "tile_save": "uypb_t_xpnum",
    "colx": ["tL","t"],
    "coly": ["PBym","uzpb"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    "labelx": r"$t \quad (s)$",
    # "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$u_y(P_{pin}^l)$"+" (m)",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PB) adapter = f(f)" + "\n",
    "tile_save": "uypb_f_xpnum",
    "colx": ["freq","freqnum"],
    "coly": ["PBym","uzpb"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$u_y(P_{pin}^l)$"+" (m)",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

#%%
repsect1 = f"{rep_save}variables_ft/PSD/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

# nt = np.min(nt,ntnum)
nfft = 2**nt
nfftnum = 2**ntnum
# # fs = np.min(fs,fsnum)
# indexpsd = df[df.index <= 2**nt].index

nvrlp = 64
# nfftnum = 128
# nfft = 128
# nvrlp = 32
# nblocks = (len(df.iloc[indexpsdnum]) - nvrlp) / (nfftnum - nvrlp)
# nblocksxp = (len(df.iloc[indexpsd]) - nvrlp) / (nfftxp - nvrlp)

power, freq = plt.psd((1.e8)*df['uzg_tot_ad'].iloc[indexpsdnum], NFFT=nfftnum, Fs=fsnum,   color=color1[2])

powerPB, freqPB = plt.psd((1.e8)*df['uzpb'].iloc[indexpsdnum], NFFT=nfftnum, Fs=fsnum,   color=color1[2])

powerPH, freqPH = plt.psd((1.e8)*df['uzph'].iloc[indexpsdnum], NFFT=nfftnum, Fs=fsnum,   color=color1[2])

powerxp, freqxp = plt.psd((1.e8)*df['yA'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2],noverlap=nvrlp)

powerPBxp, freqPBxp = plt.psd((1.e8)*df['PBym'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

powerPHxp, freqPHxp = plt.psd((1.e8)*df['PHym'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

xmax = 30.
imax = np.where(freq>=xmax)[0][0] 

psd_xp  = 10.*np.log10(powerxp / fs)
psd_num = 10.*np.log10(power / fsnum)

psdph_xp  = 10.*np.log10(powerPHxp / fs)
psdph_num = 10.*np.log10(powerPH / fsnum)

psdpb_xp  = 10.*np.log10(powerPBxp / fs)
psdpb_num = 10.*np.log10(powerPB / fsnum)

ymin = np.min(psd_xp[:imax])
ymax = 1.1*np.max([np.max(psd_xp[:imax]),np.max(psd_num[:imax])])

yminph = np.min(psdph_xp[:imax])
ymaxph = 1.1*np.max([np.max(psdph_xp[:imax]),np.max(psdph_num[:imax])])

yminpb = np.min(psdpb_xp[:imax])
ymaxpb = 1.1*np.max([np.max(psdpb_xp[:imax]),np.max(psdpb_num[:imax])])

kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad_xp",
    "x": freqxp,
    "y": psd_xp,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[0],
    "annotations": [],
    "xmax": xmax,
    "ymax": ymax,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

# ymin = np.min(psd_num)
# ymax = 1.1*np.max(psd_num)
kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad_num",
    "x": freq,
    "y": psd_num,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[2],
    "annotations": [],
    "xmax": xmax,
    "ymax": ymax,
    "ymin": ymin,
    "ypower": 3,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

#%%
imax = np.where(freq>=xmax)[0][0] 

kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad_xpnum",
    "x": [freqxp,freq],
    "y": [psd_xp,psd_num],
    "rep_save": repsect1,
    # "label1": [None,None],
    "label1": ["Experiment","Computation"],
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1,
    "annotations": None,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": ymin,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

    # pion haut :
kwargs1 = {
    "tile1": " PSD uy(PH) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uyph_xpnum",
    "x": [freqPHxp,freqPH],
    "y": [psdph_xp,psdph_num],
    "rep_save": repsect1,
    # "label1": [None,None],
    "label1": ["Experiment","Computation"],
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1,
    "annotations": None,
    "xmax": xmax,
    "ymax": ymaxph,
    "ymin": yminph,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

    # pion bas :
kwargs1 = {
    "tile1": " PSD uy(PB) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uypb_xpnum",
    "x": [freqPBxp,freqPB],
    "y": [psdpb_xp,psdpb_num],
    "rep_save": repsect1,
    # "label1": [None,None],
    "label1": ["Experiment","Computation"],
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1,
    "annotations": None,
    "xmax": xmax,
    "ymax": ymaxpb,
    "ymin": yminpb,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
#%%
repsect1 = f"{rep_save}variables_ft/specgram/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%% uygad 
# uygad num  :
kwargs1 = {
    "title1": " spectogram uy(G) adapter = f(t)" + "\n",
    "title_save": "specgram_uygad_num",
    "x": "uzg_tot_ad",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": 128,
    "noverlap": 64,
    "fs": fsnum,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

# uygad xp :
kwargs1 = {
    "title1": " spectrogram uy(G) adapter = f(t)" + "\n",
    "title_save": "specgram_uygad_xp",
    "x": "yA",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": 128,
    "noverlap": 64,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

#%% pions haut et bas :
# uyph num :
kwargs1 = {
    "title1": " spectrogram uy(PH) adapter = f(t)" + "\n",
    "title_save": "specgram_uyph_num",
    "x": "uzph",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": 128,
    "noverlap": 64,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

# uyph xp :
kwargs1 = {
    "title1": " spectrogram uy(PH) adapter = f(t)" + "\n",
    "title_save": "specgram_uyph_xp",
    "x": "PHym",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": 128,
    "noverlap": 64,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

# uypb num :
kwargs1 = {
    "title1": " spectrogram uy(PB) adapter = f(t)" + "\n",
    "title_save": "specgram_uypb_num",
    "x": "uzpb",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": 128,
    "noverlap": 64,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

# uypb xp :
kwargs1 = {
    "title1": " spectrogram uy(PB) adapter = f(t)" + "\n",
    "title_save": "specgram_uypb_xp",
    "x": "PBym",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": 128,
    "noverlap": 64,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

#%%
sys.exit()
#%%
kwargs1 = {
    "tile1": "uy(G) adapter = f(f)" + "\n",
    "tile_save": "uygad_f",
    "colx": "freq",
    "coly": "yA",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$u_y(G_ad)$"+" (m)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

ind1 = df[(df['tL']>=39.) & (df['tL']<=40.)].index
f = df['freq'].iloc[ind1[0]]
print(f"f = {f}")

kwargs1 = {
    "tile1": f"uygad fload = {f}" + "\n",
    "tile_save": f"zoom_traj2d_uygad_fload{int(f)}",
    "ind": [ind1],
    "colx": "tL",
    "coly": "yA",
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
    "tile1": f"tension fload = {f}" + "\n",
    "tile_save": f"zoom_tension_fload{int(f)}",
    "ind": [ind1],
    "colx": "tL",
    "coly": "Tension",
    "rep_save": repsect1,
    # "label1": r"$P_{pin}^u$",
    "label1": [None],
    "labelx": r"$t$"+" (s)",
    "labely": "Tension" + " (V)",
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
    "tile1": "uy(PH) adapter = f(t)" + "\n",
    "tile_save": "uyPH_t_xp",
    "colx": "tL",
    "coly": "posPHym",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_y(P_{pin}^u)$"+" (m)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PH) sleeve = f(f)" + "\n",
    "tile_save": "uyph_ft",
    "colx": "freq",
    "coly": "posPHym",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$u_y(P_{pin}^{u})$"+" (m)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PB) adapter = f(t)" + "\n",
    "tile_save": "uyPB_t_xp",
    "colx": "tL",
    "coly": "posPBym",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_y(P_{pin}^l)$"+" (m)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PB) sleeve = f(f)" + "\n",
    "tile_save": "uypb_ft",
    "colx": "freq",
    "coly": "posPBym",
    "rep_save": repsect1,
    "label1": None,
    # "labelx": r"$t \quad (s)$",
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$u_y(P_{pin}^{l})$"+" (m)",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "spin sleeve = f(t)" + "\n",
    "tile_save": "spin_t_xp",
    "colx": "tL",
    "coly": "spindeg",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$\Psi \quad (\degree)$",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "spinS sleeve = f(t)" + "\n",
    "tile_save": "spin_t_xp",
    "colx": "tL",
    "coly": "spindeg",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$\Psi \quad (\degree)$",
    "color1": color1[0],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
traj.pltraj2d(df, **kwargs1)

#%%
repsect1 = f"{rep_save}variables_ft/PSD/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")

#%% psd :
    # uygad :
# power, freq = plt.psd(1.e6*df['yA'], NFFT=2**(nt-6), Fs=fs/10, scale_by_freq=0., color=color1[2])

nfft = 2**(nt)
nblocks = (len(df)) / (nfft)
print(f"nblocks = {nblocks}")
# point d'application de la force :
power, freq = plt.psd(1.e6*df['yA'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., color=color1[2])

power_density = 10. * np.log10(power)

powerb, freqb = plt.psd(1.e6*df['posPBym'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., color=color1[2])

power_densityPB = 10. * np.log10(powerb)

powerh, freqh = plt.psd(1.e6*df['posPHym'].iloc[indexpsd], NFFT=nfft, Fs=fs, scale_by_freq=0., color=color1[2])
power_densityPH = 10. * np.log10(powerh)

#### get the ordinate of plt.psd :

linteractif = False
if (linteractif):
    fig, ax = plt.subplots()
    ax.plot(freq, power_density, label='Data')
    plt.xlim(0.,30.)
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('uyg : Interactive Plot with mplcursors')
    
    # Show the plot
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(freqh, power_densityPH, label='Data')
    plt.xlim(0.,30.)
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('uyph : Interactive Plot with mplcursors')
    
    # Show the plot
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(freqb, power_densityPB, label='Data')
    plt.xlim(0.,30.)
    # Enable cursor and display values
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('uypb : Interactive Plot with mplcursors')
    
    # Show the plot
    plt.show()

#%% split :
lpower = []
lfreq = []
lpowerph = []
lfreqph = []
lpowerpb = []
lfreqpb = []
lind = np.array_split(df.index,20)
for indi in lind: 
    dfi = df.iloc[indi]
    dfi.sort_values(by='tL',inplace=True)
    dfi.reset_index(drop=True,inplace=True)
    nt1 = int(np.floor(np.log(len(dfi['tL']))/np.log(2.)))
    indexpsd1 = dfi[dfi.index <= 2**nt1].index
    print(f"nt1 = {nt1}")
    nfft1 = 2**(nt1)
    # nfft = 2**(nt-5)
    nblocks = (len(dfi)) / (nfft1)
    print(f"nblocks = {nblocks}")
    poweri, freqi = plt.psd(1.e5*dfi['yA'].iloc[indexpsd1], NFFT=nfft1, Fs=fs, scale_by_freq=0., color=color1[2])
    powerih, freqih = plt.psd(1.e5*dfi['posPBym'].iloc[indexpsd1], NFFT=nfft1, Fs=fs, scale_by_freq=0., color=color1[2])
    powerib, freqib = plt.psd(1.e5*dfi['posPHym'].iloc[indexpsd1], NFFT=nfft1, Fs=fs, scale_by_freq=0., color=color1[2])
    # 
    poweri  = 10. * np.log10(poweri)
    powerih = 10. * np.log10(powerih)
    powerib = 10. * np.log10(powerib)
    #
    lpower.append(poweri)
    lfreq.append(freqi)
    lpowerph.append(powerih)
    lfreqph.append(freqih)
    lpowerpb.append(powerib)
    lfreqpb.append(freqib)

#%%
lanot = [(2.77,59.52),(9.65,25.63),(21.64,17.46)]
lanotph = [(2.77,21.13),(12.05,22.73),(21.12,10.81)]
lanotpb = [(2.77,28.35),(12.05,30.76)]

xmax = 30.
ymax = 65.
kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad_xp",
    "x": freq,
    "y": power_density,
    # "y": power,
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Frequency (Hz)",
    "labely": "Power (dB)",
    "color1": color1[0],
    "annotations": lanot,
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": -5.,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

kwargs1 = {
    "tile1": " PSD uy(PB) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uyPB_xp",
    "x": freqb,
    "y": power_densityPB,
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Frequency (Hz)",
    "labely": "Power (dB)",
    "color1": color1[0],
    "annotations": lanotpb,
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": -5.,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

kwargs1 = {
    "tile1": " PSD uy(PH) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uyPH_xp",
    "x": freqb,
    "y": power_densityPH,
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Frequency (Hz)",
    "labely": "Power (dB)",
    "color1": color1[0],
    "annotations": lanotph,
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": -5.,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
#%%
ymax = 1.1*np.max(power)
kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad_xp_nolog",
    "x": freq,
    # "y": power_density,
    "y": power,
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Frequency (Hz)",
    "labely": "Power (dB)",
    "color1": color1[0],
    "annotations": [],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": -5.,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

ymax = 1.1*np.max(powerh)
kwargs1 = {
    "tile1": " PSD uy(PH) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uyph_xp_nolog",
    "x": freqh,
    # "y": power_density,
    "y": powerh,
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Frequency (Hz)",
    "labely": "Power (dB)",
    "color1": color1[0],
    "annotations": [],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": -5.,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)

ymax = 1.1*np.max(powerb)
kwargs1 = {
    "tile1": " PSD uy(PB) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uypb_xp_nolog",
    "x": freqb,
    # "y": power_density,
    "y": powerb,
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Frequency (Hz)",
    "labely": "Power (dB)",
    "color1": color1[0],
    "annotations": [],
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
    "xmax": xmax,
    "ymax": ymax,
    "ymin": -5.,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
    #%% uypb :
power, freq = plt.psd(1.e6*df['posPBym'], NFFT=2**(nt-6), Fs=fs/10, scale_by_freq=0., color=color1[2])
plt.close('all')
# get the ordinate of plt.psd :
power_density = 10. * np.log10(power)

linteractif = False
if (linteractif):
  fig,ax = plt.subplots()
  ax.plot(freq, power_density, label='Data')
  # Enable cursor and display values
  mplcursors.cursor(hover=True).connect(
      "add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))

  # Adding labels and title
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('Interactive Plot with mplcursors')
    
  # Show the plot
  plt.show()
lanot = [(1.95,43.43),(4.66,24.26),(8.85,13.38),(23.45,7.16),(38.50,8.47)]
kwargs1 = {
    "tile1": " PSD uy(PB) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uypb_xp",
    "x": freq,
    "y": power_density,
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Frequency (Hz)",
    "labely": "Power (dB)",
    "color1": color1[0],
    "annotations": lanot,
    "endpoint": False,
    "xpower": 5,
    "ypower": 5,
}
# traj.PSD(df, **kwargs1)
traj.pltraj2d_list(**kwargs1)
#%% psd split :
cmap2 = traj.truncate_colormap(cm.inferno,0.,0.85,100)
lmax = [ np.max(powi) for powi in lpower ]
lampl = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpower ]
lmaxph = [ np.max(powi) for powi in lpowerph ]
lamplph = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpowerph ]
lmaxpb = [ np.max(powi) for powi in lpowerpb ]
lamplpb = [ cmap2(np.max(powi)/np.max(lmax)) for powi in lpowerpb ]
ymax = 1.1*np.max([np.max(power_density),np.max(power_densityPB),np.max(power_densityPH)])

xmax = 30.
ymin = 0.

ymaxsplit = 1.1*np.max(lmax)
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
traj.pltraj2d_list(**kwargs1)

ymaxsplit = 1.1*np.max(lmaxpb)
kwargs1 = {
    "tile1": "split PSD uy(PB) sleeve = f(freq)" + "\n",
    "tile_save": "PSD_uypb_split",
    "x": lfreqpb,
    "y": lpowerpb,
    "rep_save": repsect1,
    "label1": [None]*len(lpower),
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": lamplpb,
    "annotations": [],
    "xmax": xmax,
    "ymax": ymaxsplit,
    "ymin": ymin,
    "ypower": 3,
}
traj.pltraj2d_list(**kwargs1)


ymaxsplit = 1.1*np.max(lmaxph)
kwargs1 = {
    "tile1": "split PSD uy(PH) sleeve = f(freq)" + "\n",
    "tile_save": "PSD_uyph_split",
    "x": lfreqph,
    "y": lpowerph,
    "rep_save": repsect1,
    "label1": [None]*len(lpower),
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": lamplph,
    "annotations": [],
    "xmax": xmax,
    "ymax": ymaxsplit,
    "ymin": ymin,
    "ypower": 3,
}
traj.pltraj2d_list(**kwargs1)

# %%
repsect1 = f"{rep_save}traj_relatives_2d/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
kwargs1 = {
    "tile1": "traj. relative PH / PH_ad" + "\n",
    "tile_save": "traj2d_pionh_circ_colimpact_xp",
    "ind": [df.index],
    # "ind": [indpPH],
    "colx": "posPHxm",
    "coly": "posPHym",
    "rep_save": repsect1,
    "label1": [None],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": ['black'],
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz,     
    "scatter" : True,
    "msize" : 0.1,
    "endpoint" : [False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)
#%% pion bas :
kwargs1 = {
    "tile1": "traj. relative PB / PB_ad" + "\n",
    "tile_save": "traj2d_pionb_circ_colimpact_xp",
    "ind": [df.index],
    # "ind": [indpPB],
    "colx": "posPBxm",
    "coly": "posPBym",
    "rep_save": repsect1,
    "label1": [None],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": ['black'],
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz,     
    "scatter" : True,
    "msize" : 0.1,
    "endpoint" : [False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)
# %%
kwargs1 = {
    "tile1": "traj. relative Ccirc / PB_ad" + "\n",
    "tile_save": "traj2d_ccirc_circ",
    "ind": [df.index],
    # "ind": [indpPB],
    "colx": "posHxm",
    "coly": "posHym",
    "rep_save": repsect1,
    "label1": [None],
    "labelx": r"$X \quad (m)$",
    "labely": r"$Y \quad (m)$",
    "color1": ['black'],
    "rcirc" : ray_circ,
    "excent" : excent,
    "spinz" : spinz,     
    "scatter" : True,
    "msize" : 0.1,
    "endpoint" : [False],
    "markers" : ['s','s'],
    "arcwidth" : sect_pion_deg,
    "clmax" : cmax,
    "offsetangle" : 0.,
    "xymax" : maxdeplPB,
}
traj.pltraj2d_pion(df, **kwargs1)
