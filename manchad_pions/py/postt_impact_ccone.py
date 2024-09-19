#!/bin/python3
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.colors as pltcolors
from matplotlib import ticker
import scipy
# from matplotlib.patches import PathPatch
import sys
from rich.console import Console
from matplotlib.font_manager import FontProperties
figshow = False
#%%
class ColorMath:
    reset = "\033[0m"
    red = "\033[91m"
    green = "\033[92m"
    # darkgreen = "\033[32m"
    darkgreen = "\033[38;2;0;128;0m"
    blue = "\033[94m"

    @staticmethod
    def color_text(text, color_code):
        return f"{color_code}{text}{ColorMath.reset}"


# Example usage
equation = "E = mc^2"

# red_equation = ColorMath.color_text(equation, ColorMath.red)
# green_equation = ColorMath.color_text(equation, ColorMath.green)
# darkgreen_equation = ColorMath.color_text(equation, ColorMath.darkgreen)
# blue_equation = ColorMath.color_text(equation, ColorMath.blue)

# print("Original equation:", equation)
# print("red equation:", red_equation)
# print("green equation:", darkgreen_equation)
# print("Red equation:",

#%%
def statenergy(df,**kwargs):
    col1 = kwargs['colname']
    # col1 = kwargs['colimpact']
    dt = kwargs['dt']
    print(f"statenergy : dt = {dt}")
    df['tag'] = df.loc[:,kwargs['colimpact']].abs() > 0.
    fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
    lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
    # prb1 = [(i,j) for i,j in zip(fst,lst)]
    # dt = df.iloc[fst[0]+1]['t'] - df.iloc[fst[0]]['t']
    # on vire le dernier choc :
    fst = fst[:-1]
    lst = lst[:-1]
    # tchoc = [ dt*(j-i) for i,j in zip(fst,lst) ]
    # meanpower = [ np.mean([ df.iloc[i][col1] for i in np.arange(fsti,lsti+1) ]) for fsti,lsti in zip(fst,lst) ]
    intener = [ np.sum([ dt*(0.5*(df.iloc[i][col1]+df.iloc[i+1][col1])) for i in np.arange(fsti,lsti) ]) for fsti,lsti in zip(fst,lst) ]
    # print(f"intener = {intener}")
    meanener = np.mean(intener)
    enertot = np.sum(intener)
    stdener = np.std(intener)

    print(f"Colonne : {col1}")
    print(f"energie moyenne durant un choc = {meanener} J")
    print(f"                          std = {stdener} J")
    return meanener,stdener,enertot

def statchoc(df,**kwargs):
    col1 = kwargs['colname']
    dt = kwargs['dt']
    df['tag'] = df.loc[:,col1].abs() > 0.
    fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
    lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
    # prb1 = [(i,j) for i,j in zip(fst,lst)]
    dt = df.iloc[fst[0]+1]['t'] - df.iloc[fst[0]]['t']
    # on vire le dernier choc :
    fst = fst[:-1]
    lst = lst[:-1]
    tchoc = [ dt*(j-i) for i,j in zip(fst,lst) ]
    fchoc = [ np.mean([ df.iloc[i][col1] for i in np.arange(fsti,lsti+1) ]) for fsti,lsti in zip(fst,lst) ]
    meanfchoc = np.mean(fchoc)
    meantchoc = np.mean(tchoc)
    stdtchoc = np.std(tchoc)
    stdfchoc = np.std(fchoc)
    sumtchoc = np.sum(tchoc)
    instants_chocs = df.iloc[fst]['t']
    tlast = df.iloc[lst[-1]]['t']
    print(f"Colonne : {col1}")
    print(f"nbr de micro-impacts : {len(instants_chocs)}")
    print(f"instant du dernier choc = {tlast} s")
    print(f"temps de choc moyen = {meantchoc} s")
    print(f"ecart type tps de choc = {stdtchoc} s")
    print(f"temps de contact total = {sumtchoc} s")
    print(f"force de choc moyenne = {meanfchoc} N")
    return len(instants_chocs),meantchoc,stdtchoc,sumtchoc,meanfchoc,stdfchoc

def moyimpact(df,**kwargs):
    df['tag'] = df.loc[:,kwargs['colimpact']].abs() > 0.
    fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
    lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
    fst = fst[:-1]
    lst = lst[:-1]
    val = [ np.mean([ df.iloc[i][kwargs['colval']] for i in np.arange(fsti,lsti+1) ]) for fsti,lsti in zip(fst,lst) ]
    meanval = np.mean(val)
    stdval = np.std(val)
    print(f"Colonne value : {kwargs['colval']}")
    print(f"Colonne shock : {kwargs['colimpact']}")
    print(f"force de choc moyenne = {meanval} N")
    return meanval,stdval
#%%
globalstat = True
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
#%% options and directories :
linert = False
lamode = True
lraidtimo = False
lraidiss = True
lkxp = False
lpion = False
lpcirc = True
# Fext = 35.
Fext = 2.*79.44
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 5.e-6
h_lam = 50.e-3
b_lam = 9.e-3
lspring = 45.e-2

vlostr = int(-np.log10(vlimoden))
# dtstr = int(-np.log10(dte))
dtstr = int(1.e6*dte)
xistr = int(100.*xi)
hlstr = int(h_lam*1.e3)
blstr = int(b_lam*1.e3)
lspringstr = int(lspring*1.e2)

# namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}'
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

if not os.path.exists(rep_save):
    os.makedirs(rep_save)
    print(f"FOLDER : {rep_save} created.")
else:
    print(f"FOLDER : {rep_save} already exists.")
repsect1 = f"{rep_save}stats/"
    # repsect1 :
if not os.path.exists(repsect1):
    os.makedirs(repsect1)
    print(f"FOLDER : {repsect1} created.")
else:
    print(f"FOLDER : {repsect1} already exists.")
#%% lectue du DF :

# df = pd.read_pickle(f"{repload}2048/result.pickle")
df = pd.read_pickle(f"{repload}result.pickle")

#%%
# pattern = 'pusure'
# filtered_columns = df.filter(like=pattern, axis=1).columns.tolist()
# print(filtered_columns)
#%%
df = df[['t',
         "FN_CCONE",
         "FT_CCONE",
         "pusure_ccone",
         "pctg_glis_ad",
         "NUT",
         "RCINC",
         "DIMP",
         "THMAX",
         ]]
print(df._is_copy is None)

#%%
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
dt = df['t'][1] - df['t'][0]

# %% frequency = f(t) : 
f1 = 2.
f2 = 20.
# ttot = df.iloc[-1]['t']
ttot = 128.
print(f"tmin = {df.iloc[0]['t']}")
print(f"ttot = {ttot}")
df['freq'] = f1 + ((f2-f1)/ttot)*df['t'] 
print(f"fmin = {df.iloc[0]['freq']}")
print(f"fmax = {df.iloc[-1]['freq']}")
#%%
icrit = 0.
    # ccone :
indi_ccone = df[np.abs(df["FN_CCONE"])>icrit].index

#%% splitting into slices :
nbseg = 10
indsplit = np.array_split(df.index,nbseg)
# indi_ccone_split = np.array_split(indi_ccone,nbseg)

#%%
#%%
if globalstat:
    kw1 = {'colname' : 'FN_CCONE', 'dt' : dt}
    stat_ccone = statchoc(df,**kw1)
#
    kw1 = {'colname' : 'FT_CCONE', 'dt' : dt}
    stat_ccone_tang = statchoc(df,**kw1)
# statenergy : pour avoir les valeurs integres sur la duree du choc : 
    kw1 = {'colname' : 'pusure_ccone', 'colimpact' : 'FN_CCONE', 'dt' : dt}
    stat_ccone_pus = statenergy(df,**kw1)
    Eusccone = stat_ccone_pus[0]
    print(f"Eusccone = {Eusccone}")
# data contact : 
    kw1 = {'colval' : 'DIMP', 'colimpact' : 'FN_CCONE'}
    stat_dimp = moyimpact(df,**kw1)
    kw1 = {'colval' : 'THMAX', 'colimpact' : 'FN_CCONE'}
    stat_thmax = moyimpact(df,**kw1)
    kw1 = {'colval' : 'RCINC', 'colimpact' : 'FN_CCONE'}
    stat_rcinc = moyimpact(df,**kw1)
    kw1 = {'colval' : 'NUT', 'colimpact' : 'FN_CCONE'}
    stat_nut = moyimpact(df,**kw1)

#%%
if globalstat:
    # col_names = [r"$F_n$",r"$F_t$",r"$E_wear$",r"$t_{mean}$"]
    dict_gene = {"$F_n$"        : str("$" + "%.2f" % np.abs(stat_ccone[4]) ) + "$ N",
                 "$F_t$"        : str("$" + "%.2f" % stat_ccone_tang[4]) + "$ N",
                 "$E_{wear}$"   : str("$" + "%.2f" % (stat_ccone_pus[0]*1.e3)) + "$ mJ", 
                 "$t_{mean}$"   : "$" + str(int(np.round(stat_ccone[1] *1.e6,0))) + "$ $\mu$s",
                 "$t_{tot}$"    : "$" + str("%.2f" % stat_ccone[3]) + "$ s",
                 "$N_{shock}$"  : "$" + str(int(stat_ccone[0])) + "$",
                 } 
    # df_latex = pd.DataFrame(dict_mean,index=row_names)
    df_latex = pd.DataFrame(dict_gene,index=[0])
    # df_latex = pd.DataFrame(dict_gene)
    latex_df = df_latex.to_latex(escape=False)
    with open(f'{repsect1}F_Ewear_ccone.tex', 'w') as f:
        f.write(latex_df)
    #%%
    dict_params = {"$\delta_{imp}$"      : str("$" + "%.2f" % np.abs(stat_dimp[0]*1.e6) ) + "\mu$ m",
                 "$\theta_{max}$"      : str("$" + "%.2f" % stat_thmax[0]) + "\degree$ ",
                 "$\\frac{R_{curv}}{R_{circ}}$" : str("$" + "%.4f" % (stat_rcinc[0])) + "$", 
                 "$\theta_{nut}$" : str("$" + "%.4f" % (stat_nut[0])) + "\degree$", 
                 } 
    # df_latex = pd.DataFrame(dict_mean,index=row_names)
    df_latex = pd.DataFrame(dict_params,index=[0])
    latex_df = df_latex.to_latex(escape=False)
    with open(f'{repsect1}Cparams_ccone.tex', 'w') as f:
        f.write(latex_df)
#%%
cumulatestat = True
if cumulatestat:
    # col_names = [r"$F_n$",r"$F_t$",r"$E_wear$",r"$t_{mean}$"]
    dict_gene = {"$E_{wear}$"   : str("$" + "%.2f" % (stat_ccone_pus[2]*1.e3)) + "$ mJ", 
                 "$t_{tot}$"    : "$" + str("%.2f" % stat_ccone[3]) + "$ s",
                 "$N_{shock}$"  : "$" + str(int(stat_ccone[0])) + "$",
                 } 
    # df_latex = pd.DataFrame(dict_mean,index=row_names)
    df_latex = pd.DataFrame(dict_gene,index=[0])
    # df_latex = pd.DataFrame(dict_gene)
    latex_df = df_latex.to_latex(escape=False)
    with open(f'{repsect1}Cumulated_stats_ccone.tex', 'w') as f:
        f.write(latex_df)
#%%
nbseg = 10
indsplit = np.array_split(df.index,nbseg)
# indi_ccone_split = np.array_split(indi_ccone,nbseg)

#%%
lnchoc_ccone = []
lmoyt_ccone = []
lsumt_ccone = []
lsigmat_ccone = []
lmoyf_ccone = []
lsigmaf_ccone = []
lmoyus_ccone = []
lsigmaus_ccone = []
lenertot_ccone = []
# contact parameters :
lmoydimp = []
lmoythmax = []
lmoynut = []
lmoyrcinc = []
lstddimp = []
lstdthmax = []
lstdnut = []
lstdrcinc = []

for i,indi in enumerate(indsplit):
    df1 = df.iloc[indi]
    df1.sort_values(by='t',inplace=True)
    df1.reset_index(drop=True,inplace=True) 
    kw1 = {'colname' : 'FN_CCONE', 'dt' : dt}
    stat_ccone = statchoc(df1,**kw1)
    kw1 = {'colname' : 'pusure_ccone', 'colimpact' : 'FN_CCONE', 'dt' : dt}
    stat_ccone_pus = statenergy(df1,**kw1)
    kw1 = {'colval' : 'DIMP', 'colimpact' : 'FN_CCONE'}
    stat_dimp = moyimpact(df1,**kw1)
    kw1 = {'colval' : 'THMAX', 'colimpact' : 'FN_CCONE'}
    stat_thmax = moyimpact(df1,**kw1)
    kw1 = {'colval' : 'RCINC', 'colimpact' : 'FN_CCONE'}
    stat_rcinc = moyimpact(df1,**kw1)
    kw1 = {'colval' : 'NUT', 'colimpact' : 'FN_CCONE'}
    stat_nut = moyimpact(df1,**kw1)
    #
    # nb choc :
    lnchoc_ccone.append(stat_ccone[0])
    # mean choc time :
    lmoyt_ccone.append(stat_ccone[1])
    # std choc time :
    lsigmat_ccone.append(stat_ccone[2])
    # total contact time :
    lsumt_ccone.append(stat_ccone[3])
    # mean normal force :
    lmoyf_ccone.append(stat_ccone[4])
    # std normal force :
    lsigmaf_ccone.append(stat_ccone[5])
    # mean pusure :
    lmoyus_ccone.append(stat_ccone_pus[0])
    # std pusure :
    lsigmaus_ccone.append(stat_ccone_pus[1])
    # sum energy usure :
    lenertot_ccone.append(stat_ccone_pus[2])
    # contact parameters :
    lmoydimp.append(stat_dimp[0])
    lstddimp.append(stat_dimp[1])
    lmoythmax.append(stat_thmax[0])
    lstdthmax.append(stat_thmax[1])
    lmoyrcinc.append(stat_rcinc[0])
    lstdrcinc.append(stat_rcinc[1])
    lmoynut.append(stat_nut[0])
    lstdnut.append(stat_nut[1])

#%%
time_interval = df.iloc[-1]['t']/(nbseg)
x = np.linspace(0.,df.iloc[-1]['t']-time_interval,nbseg)

freq_interval = (df.iloc[-1]['freq'] - df.iloc[0]['freq'])/(nbseg)
freq = np.linspace(df.iloc[0]['freq'],df.iloc[-1]['freq']-freq_interval,nbseg)
colccone = 'purple'
#%%
ymaxf = np.max([ np.max(np.abs(fi)) for i,fi in enumerate(lmoyf_ccone) ])

#%% mean normal force :
fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
ax.set_ylim(ymax=1.1*ymaxf)
# ax.set_ylim(ymax=1.1*np.max(np.abs(lmoyf_ccone[ipin])))
bars = plt.bar(freq,np.abs(lmoyf_ccone),width=freq_interval,align='edge',color=colccone,alpha=0.8)

# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
    # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

for i,xi in enumerate(freq):
    ax.annotate(r'$\sigma :{%d}$' % (lsigmaf_ccone[i]) , 
    (xi, np.abs(lmoyf_ccone[i])),
    xytext=(1,4), textcoords='offset points',
    family='sans-serif', fontsize=9, color='black') 

ax.set_ylabel("Mean Normal Force (N)",fontsize=12)
ax.set_xlabel("Loading Frequency (Hz)",fontsize=12)
ax.set_xlim(xmin=f1,xmax=f2)
plt.xticks(rotation=0, ha='right')
if figshow:
    plt.show()
title=f"{repsect1}FN_ccone"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
#%% mean impact time :
ymaxt = np.max([ np.max(np.abs(fi)) for i,fi in  enumerate(lmoyt_ccone) ])

fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
ax.set_ylim(ymax=1.1*ymaxt*1.e3)
# ax.set_ylim(ymax=1.1*np.max(np.abs(lmoyf_ccone[ipin])))
bars = plt.bar(freq,np.abs(lmoyt_ccone)*1.e3,width=freq_interval,align='edge',color=colccone,alpha=0.8)

# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
    # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

for i,xi in enumerate(freq):
    ax.annotate(r'$\sigma :{%d}$' % (lsigmat_ccone[i]*1.e3) , 
    (xi, np.abs(lmoyt_ccone[i]*1.e3)),
    xytext=(1,4), textcoords='offset points',
    family='sans-serif', fontsize=9, color='black') 

ax.set_ylabel("Mean Impact Time (ms)",fontsize=12)
ax.set_xlabel("Loading Frequency (Hz)",fontsize=12)
ax.set_xlim(xmin=f1,xmax=f2)
plt.xticks(rotation=0, ha='right')
if figshow:
    plt.show()
title=f"{repsect1}tchoc_ccone"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
plt.close('all')

#%% total energy :
ymaxewear = np.max([ np.max(np.abs(fi)) for i,fi in  enumerate(lenertot_ccone) ])

fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
ax.set_ylim(ymax=1.1*ymaxewear)
bars = plt.bar(freq,np.abs(lenertot_ccone),width=freq_interval,align='edge',color=colccone,alpha=0.8)

# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')

ax.set_ylabel("Total wear energy  (J)",fontsize=12)
ax.set_xlabel("Loading Frequency (Hz)",fontsize=12)
ax.set_xlim(xmin=f1,xmax=f2)
plt.xticks(rotation=0, ha='right')
if figshow:
    plt.show()
title=f"{repsect1}enertot_ccone"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
plt.close('all')

#%% dimp :
ymaxdimp = np.max([ np.max(np.abs(fi)) for i,fi in  enumerate(lmoydimp) ])

fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
ax.set_ylim(ymax=1.1*ymaxdimp*1.e6)
bars = plt.bar(freq,np.abs(lmoydimp)*1.e6,width=freq_interval,align='edge',color=colccone,alpha=0.8)

# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')

for i,xi in enumerate(freq):
    ax.annotate(r'$\sigma :{%.2f}$' % (lstddimp[i]*1.e6) , 
    (xi, np.abs(lmoydimp[i])*1.e6),
    xytext=(1,4), textcoords='offset points',
    family='sans-serif', fontsize=9, color='black') 

ax.set_ylabel("Mean penetration value ("+r"$\mu$"+"m)",fontsize=12)
ax.set_xlabel("Loading Frequency (Hz)",fontsize=12)
ax.set_xlim(xmin=f1,xmax=f2)
plt.xticks(rotation=0, ha='right')
if figshow:
    plt.show()
title=f"{repsect1}dimp_ccone"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
plt.close('all')

#%% rcinc :
ymaxrcinc = np.max([ np.max(np.abs(fi)) for i,fi in  enumerate(lmoyrcinc) ])
yminrcinc = np.min([ np.min(np.abs(fi)) for i,fi in  enumerate(lmoyrcinc) ])

fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
ax.set_ylim(ymax=1.0001*ymaxrcinc,ymin=1.)
ax.ticklabel_format(axis='y', style='plain', useOffset=False)
# ax.set_ylim(ymax=1.1*ymaxrcinc)
bars = plt.bar(freq,np.abs(lmoyrcinc),width=freq_interval,align='edge',color=colccone,alpha=0.8)

# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')

for i,xi in enumerate(freq):
    ax.annotate(r'$\sigma :{%.4f}$' % (lstdrcinc[i]) , 
    (xi, np.abs(lmoyrcinc[i])),
    xytext=(1,4), textcoords='offset points',
    family='sans-serif', fontsize=8, color='black') 

ax.set_ylabel(r"$\frac{R_{curv}}{R_{circ}}$",fontsize=12)
ax.set_xlabel("Loading Frequency (Hz)",fontsize=12)
ax.set_xlim(xmin=f1,xmax=f2)
plt.xticks(rotation=0, ha='right')
if figshow:
    plt.show()
title=f"{repsect1}rcinc_ccone"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
plt.close('all')

#%% nutation :
ymaxnut = np.max([ np.max(np.abs(fi)) for i,fi in  enumerate(lmoynut) ])

fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
ax.set_ylim(ymax=1.1*ymaxnut)
bars = plt.bar(freq,np.abs(lmoynut),width=freq_interval,align='edge',color=colccone,alpha=0.8)

# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')

for i,xi in enumerate(freq):
    ax.annotate(r'$\sigma :{%.2f}$' % (lstdnut[i]) , 
    (xi, np.abs(lmoynut[i])),
    xytext=(1,4), textcoords='offset points',
    family='sans-serif', fontsize=9, color='black') 

ax.set_ylabel(r"$\theta_{nut} \quad (\degree)$",fontsize=12)
ax.set_xlabel("Loading Frequency (Hz)",fontsize=12)
ax.set_xlim(xmin=f1,xmax=f2)
plt.xticks(rotation=0, ha='right')
if figshow:
    plt.show()
title=f"{repsect1}nut_ccone"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
plt.close('all')

#%% thmax :
ymaxthmax = np.max([ np.max(np.abs(fi)) for i,fi in  enumerate(lmoythmax) ])

fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
ax.set_ylim(ymax=1.1*ymaxthmax)
bars = plt.bar(freq,np.abs(lmoythmax),width=freq_interval,align='edge',color=colccone,alpha=0.8)

# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')

for i,xi in enumerate(freq):
    ax.annotate(r'$\sigma :{%d}$' % (lstdthmax[i]) , 
    (xi, np.abs(lmoythmax[i])),
    xytext=(1,4), textcoords='offset points',
    family='sans-serif', fontsize=9, color='black') 

ax.set_ylabel(r"$\theta_{max} \quad (\degree)$",fontsize=12)
ax.set_xlabel("Loading Frequency (Hz)",fontsize=12)
ax.set_xlim(xmin=f1,xmax=f2)
plt.xticks(rotation=0, ha='right')
if figshow:
    plt.show()
title=f"{repsect1}thmax_ccone"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
plt.close('all')
#%%
sys.exit()