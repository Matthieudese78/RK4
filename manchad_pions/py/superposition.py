#!/bin/python3
#%%
# import scipy 
import numpy as np
import numpy.linalg as LA
import pandas as pd
import trajectories as traj
# import rotation as rota
import repchange as rc
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
# import mplcursors
import sys
#%%
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap3 = truncate_colormap(mpl.cm.inferno, 0., 0.9, n=100)
#%% usefull parameters :
color1 = ["red", "blue", "green", "orange", "purple", "pink"]
view = [20, -50]
  # 1 : xp, 2 : computation : 
alpha = [1.,0.6]
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
# matthieu
GPH = np.array([0.,0.,149.25])
pointG = pointPH - GPH
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

# manchette bloquee sinus balaye niveau n1 :
lbloq = False

# namerep = f'XP_fext_{int(Fext)}_spin_{int(spinini)}'

# if lbloq:
#     namerep = f"{namerep}_bloq"

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
fs = np.round(1./(df['tL'][1]-df['tL'][0]),1)
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

# df['xA'] = (df['xM'])+(df['tyM'])*pointG[2]
    # depl manchette 
df['PHx'] = (df['xM'])+(df['tyM'])*pointPH[2]
df['PBx'] = (df['xM'])+(df['tyM'])*pointPB[2]
df['bascul'] = (180./np.pi)*np.arctan(df['PBx'] - df['PHx'])/LA.norm(pointPB - pointPH)

    # depl adaptateur 
df['PHxA'] = (df['xA'])+(df['tyA'])*pointPH[2]
df['PBxA'] = (df['xA'])+(df['tyA'])*pointPB[2]
df['basculA'] = (180./np.pi)*np.arctan(df['PBxA'] - df['PHxA'])/LA.norm(pointPB - pointPH)

    # depl Gtot_ad_br : 
df['xA'] = (df['xA'])+(df['tyA'])*pointG[2]

if lbloq:
    ltest = False
    if ltest:
        df['PHxA'] = (df['xA'])+(df['tyA'])*pointPH[2]
        df['PBxA'] = (df['xA'])+(df['tyA'])*pointPB[2]
        df['PHxM'] = (df['xM'])+(df['tyM'])*pointPH[2]
        df['PBxM'] = (df['xM'])+(df['tyM'])*pointPB[2]
        l = 20000
        fig = plt.figure(figsize=(8,6),dpi=600)
        ax = fig.gca()
        ax.set_title("red : tya, blue : tym")
        ax.scatter(df['tL'][:l],df['tyA'][:l],s=0.1,c='r')
        ax.scatter(df['tL'][:l],df['tyM'][:l],s=0.1,c='b',alpha=0.5)
        plt.show()
        plt.close('all')
        fig = plt.figure(figsize=(8,6),dpi=600)
        ax = fig.gca()
        ax.set_title("red : xA, blue : xM")
        ax.scatter(df['tL'][:l],df['xA'][:l],s=0.1,c='r')
        ax.scatter(df['tL'][:l],df['xM'][:l],s=0.1,c='b',alpha=0.5)
        plt.show()
        plt.close('all')
        fig = plt.figure(figsize=(8,6),dpi=600)
        ax = fig.gca()
        ax.set_title("red : PBxA, blue : PBxM")
        ax.scatter(df['tL'][:l],df['PBxA'][:l],s=0.1,c='r')
        ax.scatter(df['tL'][:l],df['PBxM'][:l],s=0.1,c='b',alpha=0.5)
        plt.show()
        plt.close('all')
        fig = plt.figure(figsize=(8,6),dpi=600)
        ax = fig.gca()
        ax.set_title("red : PHxA, blue : PHxM")
        ax.scatter(df['tL'][:l],df['PHxA'][:l],s=0.1,c='r')
        ax.scatter(df['tL'][:l],df['PHxM'][:l],s=0.1,c='b',alpha=0.5)
        plt.show()
        plt.close('all')
        fig = plt.figure(figsize=(8,6),dpi=600)
        ax = fig.gca()
        ax.set_title("red : (PBxA - PHxA), blue : (PBxM - PHxM)")
        ax.scatter(df['tL'][:l],(df['PBxM'][:l] - df['PHxM'][:l]),s=0.1,c='b',alpha=0.5)
        ax.scatter(df['tL'][:l],(df['PBxA'][:l] - df['PHxA'][:l]),s=0.1,c='r',alpha=0.5)
        plt.show()
        plt.close('all')
        fig = plt.figure(figsize=(8,6),dpi=600)
        ax = fig.gca()
        ax.set_title("blue : basculement")
        ax.scatter(df['tL'][:l],df['bascul'][:l],s=0.1,c='b',alpha=0.5)
        plt.show()
        plt.close('all')

    # puis pour le tracer : on ne veut que les positions relatives :
if (not lbloq):
    # position non relative des pions haut et bas :
    df['PHx'] = df['posPHx']
    df['PBx'] = df['posPBx']

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

df['PHyA'] = 0.*df['xA']
df['PByA'] = 0.*df['xA']
df['PHzA'] = 0.*df['xA']
df['PBzA'] = 0.*df['xA']
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
name_cols = ["PHxA", "PHyA", "PHzA"]
kwargs1 = {"base2": base2, "name_cols": name_cols}
rc.repchgdf(df, **kwargs1)
name_cols = ["PBxA", "PByA", "PBzA"]
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

df = df[['tL','freq','yA','posPHxm','posPHym','posPBxm','posPBym','PHxm','PHym','PBxm','PBym','bascul','basculA']]
# df = df[['tL','freq','yA','posPHym','posPBym']]
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)

#%% NUM
linert = False
lraidtimo = False
raidiss = True
lamode = True
    # pour b_lam = 5.5
# Fext = 79.44
    # pour b_lam = 6.5
# Fext = 0.72*2.*79.44
    # pour b_lam = 7.5
# Fext = 0.83*2.*79.44
    # pour b_lam = 8
Fext = 2.*79.44
    # pour b_lam = 9
# Fext = 2.*79.44

mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 5.e-6

h_lam = 50.e-3
b_lam = 8.e-3
lspring = 45.e-2

vlostr = int(-np.log10(vlimoden))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
hlstr = int(h_lam*1.e3)
blstr = int(b_lam*1.e3)
lspringstr = int(lspring*1.e2)
amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))

namecase = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_bl_{blstr}_lspr_{lspringstr}'

if lamode:
  namecase = f'{namecase}_amodem_{amodemstr}_amodead_{amodeadstr}'
if (linert):
  namecase = f'{namecase}_inert'
if (lraidtimo):
  namecase = f'{namecase}_raidtimo'
if (raidiss):
  namecase = f'{namecase}_raidiss'

namecasebloq = 'manchadela_bloq'

repload = f'/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/{namecase}/'

if lbloq:
    repload = f'/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/RK4/manchad_pions/py/pickle/{namecasebloq}/'

rep_save = f"./fig/superposition/{namecase}/"

if lbloq:
    rep_save = f"{rep_save}bloq/"

if lbloq:
    dfnum = pd.read_pickle(f"{repload}result.pickle")
else:
    dfnum = pd.read_pickle(f"{repload}2048/result.pickle")

#%%
if lbloq:
    dfnum = dfnum[['t','uzg_tot_ad','uzpb','uzph','Fext']]

if not lbloq:
    dfnum = dfnum[['t','uzg_tot_ad','uzpb','uzph','Fext','uzph_ad','uzpb_ad']]

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

dfnum['basculnum'] = (180./np.pi)*(dfnum['uzpb'] - dfnum['uzph'])/0.3937 

if not lbloq:
  dfnum['basculnum_ad'] = (180./np.pi)*(dfnum['uzpb_ad'] - dfnum['uzph_ad'])/0.3937 

if not lbloq:
  lk = []
  lk.append({"col1": "uzpb", "col2": "uzpb_ad", "col3": "uzpbrela"})
  lk.append({"col1": "uzph", "col2": "uzph_ad", "col3": "uzphrela"})
  [traj.rela(dfnum, **ki) for i, ki in enumerate(lk)]
  dfnum['uzph'] = dfnum['uzphrela']
  dfnum['uzpb'] = dfnum['uzpbrela']
  dfnum.drop(columns=['uzphrela','uzpbrela'])

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
    "tile1": "basculement = f(t)" + "\n",
    "tile_save": "bascul_t_xpnum",
    "colx": ["tL","t"],
    "coly": ["bascul","basculnum"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    "labelx": r"$t \quad (s)$",
    "labely": "x-axis rotation "+r"$(\degree)$",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
    "alpha" : alpha,
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "basculement = f(f)" + "\n",
    "tile_save": "bascul_f_xpnum",
    "colx": ["freq","freqnum"],
    "coly": ["bascul","basculnum"],
    "rep_save": repsect1,
    "label1": ["Experiment","Computation"],
    # "labelx": r"$t \quad (s)$",
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": "x-axis rotation "+r"$(\degree)$",
    "color1": color1,
    "endpoint": [False,False],
    "xpower": 5,
    "ypower": 5,
    "alpha" : alpha,
}
traj.pltraj2d(df, **kwargs1)

if not lbloq:
    kwargs1 = {
        "tile1": "basculement adapter = f(t)" + "\n",
        "tile_save": "basculad_t_xpnum",
        "colx": ["tL","t"],
        "coly": ["basculA","basculnum_ad"],
        "rep_save": repsect1,
        "label1": ["Experiment","Computation"],
        "labelx": r"$t \quad (s)$",
        "labely": "x-axis rotation "+r"$(\degree)$",
        "color1": color1,
        "endpoint": [False,False],
        "xpower": 5,
        "ypower": 5,
        "alpha" : alpha,
    }
    traj.pltraj2d(df, **kwargs1)

    kwargs1 = {
        "tile1": "basculement adapter = f(f)" + "\n",
        "tile_save": "basculad_f_xpnum",
        "colx": ["freq","freqnum"],
        "coly": ["basculA","basculnum_ad"],
        "rep_save": repsect1,
        "label1": ["Experiment","Computation"],
        # "labelx": r"$t \quad (s)$",
        "labelx": "Loading Frequency" + " (Hz)",
        "labely": "x-axis rotation "+r"$(\degree)$",
        "color1": color1,
        "endpoint": [False,False],
        "xpower": 5,
        "ypower": 5,
        "alpha" : alpha,
    }
    traj.pltraj2d(df, **kwargs1)

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
    "alpha" : alpha,
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
    "alpha" : alpha,
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
    "alpha" : alpha,
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
    "alpha" : alpha,
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
    "alpha" : alpha,
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
    "alpha" : alpha,
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

# nfftnum = nfft
# nfft = nfft
# nvrlp = 32
# nblocks = (len(df.iloc[indexpsdnum]) - nvrlp) / (nfftnum - nvrlp)
# nblocksxp = (len(df.iloc[indexpsd]) - nvrlp) / (nfftxp - nvrlp)

power, freq = plt.psd((1.e8)*df['uzg_tot_ad'].iloc[indexpsdnum], NFFT=nfftnum, Fs=fsnum,   color=color1[2])

powerPB, freqPB = plt.psd((1.e8)*df['uzpb'].iloc[indexpsdnum], NFFT=nfftnum, Fs=fsnum,   color=color1[2])

powerPH, freqPH = plt.psd((1.e8)*df['uzph'].iloc[indexpsdnum], NFFT=nfftnum, Fs=fsnum,   color=color1[2])

powerxp, freqxp = plt.psd((1.e8)*df['yA'].iloc[indexpsd], NFFT=nfft, Fs=fs, color=color1[2])

powerPBxp, freqPBxp = plt.psd((1.e8)*df['PBym'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

powerPHxp, freqPHxp = plt.psd((1.e8)*df['PHym'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

powerbascxp, freqbascxp = plt.psd((1.e8)*df['bascul'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

powerbascxpA, freqbascxpA = plt.psd((1.e8)*df['basculA'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

powerbascnum, freqbascnum = plt.psd((1.e8)*df['basculnum'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

if not lbloq:
    powerbascnumA, freqbascnumA = plt.psd((1.e8)*df['basculnum_ad'].iloc[indexpsd], NFFT=nfft, Fs=fs,   color=color1[2])

xmax = 30.
imax = np.where(freq>=xmax)[0][0] 

psd_xp  = 10.*np.log10(powerxp / fs)
psd_num = 10.*np.log10(power / fsnum)

psdph_xp  = 10.*np.log10(powerPHxp / fs)
psdph_num = 10.*np.log10(powerPH / fsnum)

psdpb_xp  = 10.*np.log10(powerPBxp / fs)
psdpb_num = 10.*np.log10(powerPB / fsnum)

psdbasc_xp  = 10.*np.log10(powerbascxp / fs)
psdbasc_num = 10.*np.log10(powerbascnum / fsnum)

psdbasc_xp_ad  = 10.*np.log10(powerbascxpA / fs)
if not lbloq:
    psdbasc_num_ad = 10.*np.log10(powerbascnumA / fsnum)

ymin = np.min(psd_xp[:imax])
ymax = 1.1*np.max([np.max(psd_xp[:imax]),np.max(psd_num[:imax])])

yminph = np.min(psdph_xp[:imax])
ymaxph = 1.1*np.max([np.max(psdph_xp[:imax]),np.max(psdph_num[:imax])])

yminpb = np.min(psdpb_xp[:imax])
ymaxpb = 1.1*np.max([np.max(psdpb_xp[:imax]),np.max(psdpb_num[:imax])])

yminbasc = np.min(psdbasc_xp[:imax])
ymaxbasc = 1.1*np.max([np.max(psdbasc_xp[:imax]),np.max(psdbasc_num[:imax])])

if not lbloq:
    yminbasc_ad = np.min(psdbasc_xp_ad[:imax])
    ymaxbasc_ad = 1.1*np.max([np.max(psdbasc_xp_ad[:imax]),np.max(psdbasc_num_ad[:imax])])

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
    "tile1": " PSD basculement sleeve = f(freq)" + "\n",
    "tile_save": "PSD_bascul_sleeve_xpnum",
    "x": [freqbascxp,freqbascnum],
    "y": [psdbasc_xp,psdbasc_num],
    "rep_save": repsect1,
    # "label1": [None,None],
    "label1": ["Experiment","Computation"],
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1,
    "annotations": None,
    "xmax": xmax,
    "ymax": ymaxbasc,
    "ymin": yminbasc,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.pltraj2d_list(**kwargs1)

if not lbloq:
    kwargs1 = {
        "tile1": " PSD basculement adapter = f(freq)" + "\n",
        "tile_save": "PSD_bascul_adapter_xpnum",
        "x": [freqbascxpA,freqbascnumA],
        "y": [psdbasc_xp_ad,psdbasc_num_ad],
        "rep_save": repsect1,
        # "label1": [None,None],
        "label1": ["Experiment","Computation"],
        "labelx": r"$Frequency \quad (Hz)$",
        "labely": r"$Power \quad (dB)$",
        "color1": color1,
        "annotations": None,
        "xmax": xmax,
        "ymax": ymaxbasc_ad,
        "ymin": yminbasc_ad,
        "xpower": 3,
        "ypower": 3,
        "loc_leg": "upper right",
    }

    traj.pltraj2d_list(**kwargs1)

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

# nfftxp = 1024
nfft = 2048
noverlap = 512  
# noverlap = 1024

nfftxp = 1024
noverlapxp = 512
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
    "nfft": nfft,
    "noverlap": noverlap,
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
    "nfft": nfft,
    "noverlap": noverlap,
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
    "nfft": nfft,
    "noverlap": noverlap,
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
    "nfft": nfft,
    "noverlap": noverlap,
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
    "nfft": nfft,
    "noverlap": noverlap,
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
    "nfft": nfft,
    "noverlap": noverlap,
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
    "title1": " spectrogram bascul sleeve xp" + "\n",
    "title_save": "specgram_bascul_sleeve_xp",
    "x": "bascul",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": nfft,
    "noverlap": noverlap,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

kwargs1 = {
    "title1": " spectrogram bascul adapter xp" + "\n",
    "title_save": "specgram_bascul_adapter_xp",
    "x": "basculA",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": nfft,
    "noverlap": noverlap,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

kwargs1 = {
    "title1": " spectrogram bascul sleeve num" + "\n",
    "title_save": "specgram_bascul_sleeve_num",
    "x": "basculnum",
    "rep_save": repsect1,
    "label1": None,
    "labelx": "Loading Frequency" + " (Hz)",
    "labely": r"$Frequency \quad (Hz)$",
    "annotations": None,
    "nfft": nfft,
    "noverlap": noverlap,
    "fs": fs,
    "f1": f1,
    "f2": f2,
    "ymax": 80.,
    "xpower": 3,
    "ypower": 3,
    "loc_leg": "upper right",
}

traj.spectro(df,**kwargs1)

if not lbloq:
  kwargs1 = {
      "title1": " spectrogram bascul adapter num" + "\n",
      "title_save": "specgram_bascul_adapter_num",
      "x": "basculnum_ad",
      "rep_save": repsect1,
      "label1": None,
      "labelx": "Loading Frequency" + " (Hz)",
      "labely": r"$Frequency \quad (Hz)$",
      "annotations": None,
      "nfft": nfft,
      "noverlap": noverlap,
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