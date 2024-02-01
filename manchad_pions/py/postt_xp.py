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
    # which slice ?
slice = 1
# cas la lache de la manchette avec juste le poids :
# namerep = "manchadela_weight"
# namerep = "manchadela_RSG"
# namerep = "manchadela_RSG_conefixe"
lpion = False
lpcirc = True

Fext = 193.
spinini = 0.

namerep = f'XP_fext_{int(Fext)}_spin_{int(spinini)}'

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

# O2 : decale de 30 degres :
# essai = '20201127_1406'

filename = f'{repload}{essai}_laser.pickle'

# namerep = f"edf_xp"
# repload = f"./pickle/{namerep}/"

# %%
rep_save = f"./fig/{namerep}/"
if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")

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
df.sort_values(by='tL',inplace=True)
df.reset_index(drop=True,inplace=True)
#%% on remet l'origine du temps a 0 :
df['tL'] = df['tL'] - df['tL'][0]
nt = int(np.floor(np.log(len(df['tL']))/np.log(2.)))
indexpsd = df[df.index < 2**nt].index
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
#%% on rajoute les coord bidons :
df['yA'] = 0.*df['xA']
df['zA'] = 0.*df['xA']
df['zM'] = 0.*df['xA']
df['posHz'] = 0.*df['xA']
df['posPHz'] = 0.*df['xA']
df['posPBz'] = 0.*df['xA']

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

#%% on vire eventuellement les offsets dus aux appareils de mesure :
loffset = False
def rmoffset(df,**kwargs):
    # return "done"    
    xval = df[kwargs['col1']]
    xini = kwargs['valini']
    return xval - xini

def rmoffsetdf(df,**kwargs):
    dict1 = {kwargs['col1'] : df.apply(rmoffset,**kwargs,axis=1)}
    df1 = pd.DataFrame(dict1)
    print (df1.index)
    df[kwargs["col1"]] = df1
if (loffset):
    lk = []
    lk.append({'col1' : 'xA', 'valini' :     df['xA'][0]})
    lk.append({'col1' : 'yA', 'valini' :     df['yA'][0]})
    lk.append({'col1' : 'zA', 'valini' :     df['zA'][0]})
    lk.append({'col1' : 'xM', 'valini' :     df['xM'][0]})
    lk.append({'col1' : 'yM', 'valini' :     df['yM'][0]})
    lk.append({'col1' : 'zM', 'valini' :     df['zM'][0]})
    lk.append({'col1' : 'posHx', 'valini' :  df['posHx'][0] })  
    lk.append({'col1' : 'posHy', 'valini' :  df['posHy'][0] })  
    lk.append({'col1' : 'posHz', 'valini' :  df['posHz'][0] })  
    lk.append({'col1' : 'posPHx', 'valini' : df['posPHx'][0] }) 
    lk.append({'col1' : 'posPHy', 'valini' : df['posPHy'][0] }) 
    lk.append({'col1' : 'posPHz', 'valini' : df['posPHz'][0] }) 
    lk.append({'col1' : 'posPBx', 'valini' : df['posPBx'][0] }) 
    lk.append({'col1' : 'posPBy', 'valini' : df['posPBy'][0] }) 
    lk.append({'col1' : 'posPBz', 'valini' : df['posPBz'][0] }) 
    [ rmoffsetdf(df,**ki) for ki in lk]

#%% trajectoires rela :
lrela = False
if lrela:
    lk = []
    lk.append({"col1": "posHx", "col2": "xA", "col3": "uxcerela"})
    lk.append({"col1": "posHy", "col2": "yA", "col3": "uycerela"})
    lk.append({"col1": "posHz", "col2": "zA", "col3": "uzcerela"})

    lk.append({"col1": "posPHx", "col2": "xA", "col3": "uxphrela"})
    lk.append({"col1": "posPHy", "col2": "yA", "col3": "uyphrela"})
    lk.append({"col1": "posPHz", "col2": "zA", "col3": "uzphrela"})
    lk.append({"col1": "posPBx", "col2": "xA", "col3": "uxpbrela"})
    lk.append({"col1": "posPBy", "col2": "yA", "col3": "uypbrela"})
    lk.append({"col1": "posPBz", "col2": "zA", "col3": "uzpbrela"})

    [traj.rela(df, **ki) for i, ki in enumerate(lk)]

#%% find peaks deplacements :

jeuH = 1. # au niveau de la collerette ie  contact cône-cône?
jeuB = 6.5 # sortie d'adaptateur?
jeuPH = 2. # pion haut
jeuPB = 2. # pion bas
hpion = 2.4 # ??
facteurProminence = 5

df['posHr']  = np.sqrt(df['posHx']**2+df['posHy']**2)
df['posPHr'] = np.sqrt(df['posPHx']**2+df['posPHy']**2)
# df['posBr']  = np.sqrt(df['posBx']**2+df['posBy']**2)
df['posPBr'] = np.sqrt(df['posPBx']**2+df['posPBy']**2)
prominenceThrH  = jeuH/facteurProminence
prominenceThrB  = jeuB/facteurProminence
prominenceThrPB = jeuPB/facteurProminence
prominenceThrPH = jeuPH/facteurProminence

peaksH_laser = scipy.signal.find_peaks(df['posHr'].values,prominence=(prominenceThrH,None))[0]
peaksPH_laser = scipy.signal.find_peaks(df['posPHr'].values,prominence=(prominenceThrPH,None))[0]
# peaksB_laser = scipy.signal.find_peaks(df['posBr'].values,prominence=(prominenceThrB,None))[0]
peaksPB_laser = scipy.signal.find_peaks(df['posPBr'].values,prominence=(prominenceThrPB,None))[0]

indpH = pd.Index(peaksH_laser)
indpPH = pd.Index(peaksPH_laser)
indpPB = pd.Index(peaksPB_laser)

#%% old laser selectionne :
# df['Gad'] = 0.5*(df['L1'] + df['L2'])
laser = 'yA'
df['yA']      =  1.e-3 * df['yA']
df['posPHxm'] =  1.e-3 * df['posPHx']
df['posPHym'] =  1.e-3 * df['posPHy']
df['posPBxm'] =  1.e-3 * df['posPBx']
df['posPBym'] =  1.e-3 * df['posPBy']
df['posHxm'] =  1.e-3 * df['posHx']
df['posHym'] =  1.e-3 * df['posHy']
df['posHrm'] =  1.e-3 * df['posHr']

# %% maximum penetration :
penePB = cmax - df['posPBxm'].abs().max()
penePH = cmax - df['posPHxm'].abs().max()
maxdeplPB = df['posPBxm'].abs().max()
maxdeplPH = df['posPHxm'].abs().max()
maxdeplHr = df['posHrm'].abs().max()
# %% lecture du dataframe :
repsect1 = f"{rep_save}variables_ft/"
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%%
kwargs1 = {
    "tile1": "uy(G) adapter = f(t)" + "\n",
    "tile_save": "uygad_t_xp",
    "colx": "tL",
    "coly": "yA",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_y(G_ad)$"+" (m)",
    "color1": color1[0],
}
traj.pltraj2d(df, **kwargs1)

kwargs1 = {
    "tile1": "uy(PH) adapter = f(t)" + "\n",
    "tile_save": "uyPH_t_xp",
    "colx": "tL",
    "coly": "posPHym",
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$t \quad (s)$",
    "labely": r"$u_y(P_{pin}^u)$"+" (m)",
    "color1": color1[1],
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
    "color1": color1[2],
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
power, freq = plt.psd(1.e6*df['yA'], NFFT=2**(nt-6), Fs=fs/10, scale_by_freq=0., color=color1[2])
plt.close('all')
# get the ordinate of plt.psd :
power_density = 10. * np.log10(power)

linteractif = False
if (linteractif):
    fig, ax = plt.subplots()
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
lanot = [(0.30,59.86),(1.90,40.99),(4.60,27.33),(5.50,20.28)]
kwargs1 = {
    "tile1": " PSD uy(G) adapter = f(freq)" + "\n",
    "tile_save": "PSD_uygad_xp",
    "x": freq,
    "y": power_density,
    "rep_save": repsect1,
    "label1": None,
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[2],
    "annotations": lanot,
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
    fig, ax = plt.subplots()
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
    "labelx": r"$Frequency \quad (Hz)$",
    "labely": r"$Power \quad (dB)$",
    "color1": color1[2],
    "annotations": lanot,
}
# traj.PSD(df, **kwargs1)
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
