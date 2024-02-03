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

red_equation = ColorMath.color_text(equation, ColorMath.red)
green_equation = ColorMath.color_text(equation, ColorMath.green)
darkgreen_equation = ColorMath.color_text(equation, ColorMath.darkgreen)
blue_equation = ColorMath.color_text(equation, ColorMath.blue)

print("Original equation:", equation)
print("red equation:", red_equation)
print("green equation:", darkgreen_equation)
# print("Red equation:",

#%%
def statenergy(df,**kwargs):
    col1 = kwargs['colname']
    dt = kwargs['dt']
    df['tag'] = df.loc[:,col1].abs() > 0.
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
    stdener = np.std(intener)

    print(f"Colonne : {col1}")
    print(f"energie moyenne durant un choc = {meanener} J")
    print(f"                          std = {stdener} J")
    return meanener,stdener

def statchoc(df,**kwargs):
    col1 = kwargs['colname']
    dt = kwargs['dt']
    df['tag'] = df[col1].abs() > 0.
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

def statchoc2(df,**kwargs):
    col1 = kwargs['colname1']
    col2 = kwargs['colname2']
    dt = kwargs['dt']
    df['tag'] = (df[col1].abs() > 0.) & (df[col2].abs() > 0.)
    fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
    lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
    # prb1 = [(i,j) for i,j in zip(fst,lst)]
    dt = df.iloc[fst[0]+1]['t'] - df.iloc[fst[0]]['t']
    # on vire le dernier choc :
    fst = fst[:-1]
    lst = lst[:-1]
    tchoc = [ dt*(j-i+1) for i,j in zip(fst,lst) ]
    meantchoc = np.mean(tchoc)
    stdtchoc = np.std(tchoc)
    sumtchoc = np.sum(tchoc)
    instants_chocs = df.iloc[fst]['t']
    tlast = df.iloc[lst[-1]]['t']
    print(f"Colonne 1 : {col1}, Colonne 2 : {col2}")
    print(f"nbr de micro-impacts : {len(instants_chocs)}")
    print(f"instant du dernier choc = {tlast} s")
    print(f"temps de choc moyen = {meantchoc} s")
    print(f"ecart type tps de choc = {stdtchoc} s")
    print(f"temps de contact total = {sumtchoc} s")
    return meantchoc,stdtchoc,sumtchoc,len(tchoc)
#%%
globalstat = False
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
linert = True
lamode = True
lkxp = False
lpion = False
lpcirc = True
Fext = 193.
mu = 0.6
xi = 0.05
amode_m = 0.02
amode_ad = 0.02
vlimoden = 1.e-5
spinini = 0.
dte = 1.e-6

vlostr = int(-np.log10(vlimoden))
dtstr = int(-np.log10(dte))
xistr = int(100.*xi)

namerep = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}'

amodemstr = str(int(amode_m*100.))
amodeadstr = str(int(amode_ad*100.))

if lamode:
    namerep = f'{namerep}_amodem_{amodemstr}_amodead_{amodeadstr}'
if (lkxp):
  namerep = f'{namerep}_kxp'
if (linert):
    namerep = f'{namerep}_inert'

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

df = pd.read_pickle(f"{repload}result.pickle")

#%%
# pattern = 'pusure'
# filtered_columns = df.filter(like=pattern, axis=1).columns.tolist()
# print(filtered_columns)
#%%
df = df[['t',
         'FN_pcircb1',
         'FN_pcircb2',
         'FN_pcircb3',
         'FN_pcirch1',
         'FN_pcirch2',
         'FN_pcirch3',
         'FT_pcircb1',
         'FT_pcircb2',
         'FT_pcircb3',
         'FT_pcirch1',
         'FT_pcirch2',
         'FT_pcirch3',
         "FN_CCONE",
         "FT_CCONE",
         "pusure_ccone",
         "pctg_glis_ad",
         "pusure_pcirc.3",
         "pusure_pcirc.4",
         "pusure_pcirc.5",
         ]]
print(df._is_copy is None)
df['FN_pb'] = np.sqrt(df['FN_pcircb1']**2 + df['FN_pcircb1']**2 + df['FN_pcircb1'])
df['FN_ph'] = np.sqrt(df['FN_pcirch1']**2 + df['FN_pcirch1']**2 + df['FN_pcirch1'])

#%%
df.sort_values(by='t',inplace=True)
df.reset_index(drop=True,inplace=True)
dt = df['t'][1] - df['t'][0]

#%%
icrit = 0.
    # ccone :
indi_ccone = df[np.abs(df["FN_CCONE"])>icrit].index
    # pion haut
indi_ph = df[(np.abs(df["FN_pcirch1"])>icrit) | 
             (np.abs(df["FN_pcirch2"])>icrit) |
             (np.abs(df["FN_pcirch3"])>icrit)].index
indni_ph = df.drop(indi_ph).index

    # pion bas :
indi_pb = df[(np.abs(df["FN_pcircb1"])>icrit) | 
             (np.abs(df["FN_pcircb2"])>icrit) |
             (np.abs(df["FN_pcircb3"])>icrit)].index
indni_pb = df.drop(indi_pb).index

indi_pb1 = df[(np.abs(df["FN_pcircb1"])>icrit)].index
indi_pb2 = df[(np.abs(df["FN_pcircb2"])>icrit)].index
indi_pb3 = df[(np.abs(df["FN_pcircb3"])>icrit)].index

indi_ph1 = df[(np.abs(df["FN_pcirch1"])>icrit)].index
indi_ph2 = df[(np.abs(df["FN_pcirch2"])>icrit)].index
indi_ph3 = df[(np.abs(df["FN_pcirch3"])>icrit)].index

indi_pb12 = indi_pb1.intersection(indi_pb2)
indi_pb13 = indi_pb1.intersection(indi_pb3)
indi_pb23 = indi_pb2.intersection(indi_pb3)

#%% splitting into slices :
nbseg = 10
indsplit = np.array_split(df.index,nbseg)
indi_pb_split = np.array_split(indi_pb,nbseg)
indi_ph_split = np.array_split(indi_ph,nbseg)
indi_ccone_split = np.array_split(indi_ccone,nbseg)

#%%
if globalstat:
    #  detection des chocs :
    kw1 = {'colname' : 'FN_pcircb1', 'dt' : dt}
    stat_pb1 = statchoc(df,**kw1)
    #
    kw1 = {'colname' : 'FN_pcircb2', 'dt' : dt}
    stat_pb2 = statchoc(df,**kw1)
    #
    kw1 = {'colname' : 'FN_pcircb3', 'dt' : dt}
    stat_pb3 = statchoc(df,**kw1)
    #
#%%
if globalstat:
    #
    kw1 = {'colname1' : 'FN_pcircb1', 'colname2' : 'FN_pcircb3', 'dt' : dt}
    stat_pb13 = statchoc2(df,**kw1)
    #
    kw1 = {'colname1' : 'FN_pcircb1', 'colname2' : 'FN_pcircb2', 'dt' : dt}
    stat_pb12 = statchoc2(df,**kw1)
    #
    kw1 = {'colname1' : 'FN_pcircb2', 'colname2' : 'FN_pcircb3', 'dt' : dt}
    stat_pb23 = statchoc2(df,**kw1)
    #
    kw1 = {'colname' : 'FN_CCONE', 'dt' : dt}
    stat_ccone = statchoc(df,**kw1)
#%%
if globalstat:
    kw1 = {'colname' : 'FT_pcircb1', 'dt' : dt}
    stat_pb1_tang = statchoc(df,**kw1)
    #
    kw1 = {'colname' : 'FT_pcircb2', 'dt' : dt}
    stat_pb2_tang = statchoc(df,**kw1)
    #
    kw1 = {'colname' : 'FT_pcircb3', 'dt' : dt}
    stat_pb3_tang = statchoc(df,**kw1)
#%%
if globalstat:
    kw1 = {'colname' : 'pusure_pcirc.3', 'dt' : dt}
    stat_pb1_pus = statenergy(df,**kw1)
    #
    kw1 = {'colname' : 'pusure_pcirc.4', 'dt' : dt}
    stat_pb2_pus = statenergy(df,**kw1)
    #
    kw1 = {'colname' : 'pusure_pcirc.5', 'dt' : dt}
    stat_pb3_pus = statenergy(df,**kw1)

    Euspb1 = stat_pb1_pus[0]
    Euspb2 = stat_pb2_pus[0]
    Euspb3 = stat_pb3_pus[0]
    print(f"Euspb1 = {Euspb1}")
    print(f"Euspb2 = {Euspb2}")
    print(f"Euspb3 = {Euspb3}")
#%% on trace les tableaux : 
pin1 = r"$\textcolor{red}{\mbox{pin}_1}$" 
pin2 = r"$\textcolor{mydarkgreen}{\mbox{pin}_2}$" 
pin3 = r"$\textcolor{blue}{\mbox{pin}_3}$" 
row_names = [pin1,pin2,pin3]
dict_tchoctot = {pin1 : [ str("$" + "%.2f" % stat_pb1[3] ) + "$ s","$" +  str("%.2f" % stat_pb12[2]) + "$ s","$" + str("%.2f" % stat_pb13[2]) + "$ s"] ,
                 pin2 : [ str("$" + "%.2f" % stat_pb12[2]) + "$ s","$" +  str("%.2f" % stat_pb2[3] ) + "$ s", "$" + str("%.2f" % stat_pb23[2]) + "$ s"] ,
                 pin3 : [ str("$" + "%.2f" % stat_pb13[2]) + "$ s","$" +  str("%.2f" % stat_pb23[2]) + "$ s","$" + str( "%.2f" % stat_pb3[3]) + "$ s"]} 
df_tchoctot = pd.DataFrame(dict_tchoctot,index=row_names)
latex_tchoctot = df_tchoctot.to_latex(escape=False)
with open(f'{repsect1}tchoctot.tex', 'w') as f:
    f.write(latex_tchoctot)
#%% meantchoc
dict_stat = {pin1 : [ "$" + str(int(np.round(stat_pb1[1] *1.e6,0))) + "$ $\mu$s", "$" +  str(int(np.round(stat_pb12[0]*1.e6,0))) + "$ $\mu$s", "$" + str(int(np.round(stat_pb13[0]*1.e6,0))) + "$ $\mu$s"] ,
             pin2 : [ "$" + str(int(np.round(stat_pb12[0]*1.e6,0))) + "$ $\mu$s", "$" +  str(int(np.round(stat_pb2[1] *1.e6,0))) + "$ $\mu$s", "$" + str(int(np.round(stat_pb23[0]*1.e6,0))) + "$ $\mu$s"] ,
             pin3 : [ "$" + str(int(np.round(stat_pb13[0]*1.e6,0))) + "$ $\mu$s", "$" +  str(int(np.round(stat_pb23[0]*1.e6,0))) + "$ $\mu$s", "$" + str(int(np.round(stat_pb3[1]*1.e6,0)))  + "$ $\mu$s"]} 
df_latex = pd.DataFrame(dict_stat,index=row_names)
latex_df = df_latex.to_latex(escape=False)
with open(f'{repsect1}tchocmean.tex', 'w') as f:
    f.write(latex_df)
#%% impact number 
dict_stat = {pin1 : [ "$" + str(int(stat_pb1[0] )) + "$", "$" +  str(int(stat_pb12[3])) + "$", "$" + str(int(stat_pb13[3])) + "$"] ,
             pin2 : [ "$" + str(int(stat_pb12[3])) + "$", "$" +  str(int(stat_pb2[0] )) + "$", "$" + str(int(stat_pb23[3])) + "$"] ,
             pin3 : [ "$" + str(int(stat_pb13[3])) + "$", "$" +  str(int(stat_pb23[3])) + "$", "$" + str(int(stat_pb3[0] )) + "$"] } 
df_latex = pd.DataFrame(dict_stat,index=row_names)
latex_df = df_latex.to_latex(escape=False)
with open(f'{repsect1}impactnumber.tex', 'w') as f:
    f.write(latex_df)
#%% Fn, Ft, Eweat 
dict_stat = {"$F_n$"      : [ str("$" + "%.2f" % stat_pb1[4] ) + "$ N","$" +  str("%.2f" % stat_pb2[4]) + "$ N","$" + str("%.2f" % stat_pb3[4]) + "$ N"] ,
             "$F_t$"      : [ str("$" + "%.2f" % stat_pb1_tang[4]) + "$ N","$" +  str("%.2f" % stat_pb2_tang[4] ) + "$ N","$" + str("%.2f" % stat_pb3_tang[4]) + "$ N"] ,
             "$E_{wear}$" : [ str("$" + "%.2f" % stat_pb1_pus[0]) + "$ J","$" +  str("%.2f" % stat_pb2_pus[0]) + "$ J","$" + str("%.2f" % stat_pb3_pus[0] ) + "$ J"]} 
df_latex = pd.DataFrame(dict_stat,index=row_names)
latex_df = df_latex.to_latex(escape=False)
with open(f'{repsect1}F_Ewear.tex', 'w') as f:
    f.write(latex_df)
#%%
fig, ax = plt.subplots(figsize=(1, 1), dpi=600)
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_tchoctot.values,
        #  colLabels=df_tchoctot.columns,
         colLabels=df_tchoctot.keys(),
         rowLabels=df_tchoctot.keys(),
         cellLoc = 'center', 
         loc='center',
         )
table.auto_set_font_size(False)
table.set_fontsize(10)
table[(0, 0)].set_text_props(color='red')
table[(0, 1)].set_text_props(color='green')
table[(0, 2)].set_text_props(color='blue')
table[(1, -1)].set_text_props(color='red')
table[(2, -1)].set_text_props(color='green')
table[(3, -1)].set_text_props(color='blue')
for (row, col), cell in table.get_celld().items():
  if (row == 0) | (col == -1):
    cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    # cell.PAD = 0.5
  if (col == -1):
    cell.PAD = (0.2)
table.auto_set_column_width(col=list(np.arange(4)))

fig.tight_layout()
plt.savefig(f"{repsect1}tchoctot.png")
# plt.show()

#%% traitement statistique :
lnchocpb = [[],[],[]]
lmoytpb = [[],[],[]]
lsumtpb = [[],[],[]]
lsigmatpb = [[],[],[]]
lmoyfpb = [[],[],[]]
lsigmafpb = [[],[],[]]
lmoyuspb = [[],[],[]]
lsigmauspb = [[],[],[]]
for i,indi in enumerate(indsplit):
    df1 = df.iloc[indi]
    df1.sort_values(by='t',inplace=True)
    df1.reset_index(drop=True,inplace=True) 
    kw1 = {'colname' : 'FN_pcircb1', 'dt' : dt}
    stat_pb1 = statchoc(df1,**kw1)
    kw1 = {'colname' : 'FN_pcircb2', 'dt' : dt}
    stat_pb2 = statchoc(df1,**kw1)
    kw1 = {'colname' : 'FN_pcircb3', 'dt' : dt}
    stat_pb3 = statchoc(df1,**kw1)
    kw1 = {'colname' : 'pusure_pcirc.3', 'dt' : dt}
    stat_pb1_pus = statenergy(df1,**kw1)
    kw1 = {'colname' : 'pusure_pcirc.4', 'dt' : dt}
    stat_pb2_pus = statenergy(df1,**kw1)
    kw1 = {'colname' : 'pusure_pcirc.5', 'dt' : dt}
    stat_pb3_pus = statenergy(df1,**kw1)
    # nb choc :
    lnchocpb[0].append(stat_pb1[0])
    lnchocpb[1].append(stat_pb2[0])
    lnchocpb[2].append(stat_pb3[0])
    # mean choc time :
    lmoytpb[0].append(stat_pb1[1])
    lmoytpb[1].append(stat_pb2[1])
    lmoytpb[2].append(stat_pb3[1])
    # std choc time :
    lsigmatpb[0].append(stat_pb1[2])
    lsigmatpb[1].append(stat_pb2[2])
    lsigmatpb[2].append(stat_pb3[2])
    # total contact time :
    lsumtpb[0].append(stat_pb1[3])
    lsumtpb[1].append(stat_pb2[3])
    lsumtpb[2].append(stat_pb3[3])
    # mean normal force :
    lmoyfpb[0].append(stat_pb1[4])
    lmoyfpb[1].append(stat_pb2[4])
    lmoyfpb[2].append(stat_pb3[4])
    # std normal force :
    lsigmafpb[0].append(stat_pb1[5])
    lsigmafpb[1].append(stat_pb2[5])
    lsigmafpb[2].append(stat_pb3[5])
    # mean normal force :
    lmoyuspb[0].append(stat_pb1_pus[0])
    lmoyuspb[1].append(stat_pb2_pus[0])
    lmoyuspb[2].append(stat_pb3_pus[0])
    # mean normal force :
    lsigmauspb[0].append(stat_pb1_pus[1])
    lsigmauspb[1].append(stat_pb2_pus[1])
    lsigmauspb[2].append(stat_pb3_pus[1])

#%%
time_interval = df.iloc[-1]['t']/(nbseg)
x = np.linspace(0.,df.iloc[-1]['t']-time_interval,nbseg)

#%% mean normal force :
colorpin = ['red','green','blue']
ymaxf = np.max([ np.max(np.abs(fi)) for i,fi in  enumerate(lmoyfpb) ])
for ipin in np.arange(3): 
    fig = plt.figure(figsize=(8,6), dpi=1000)
    ax=plt.axes()
    ax.set_ylim(ymax=1.1*ymaxf)
    # ax.set_ylim(ymax=1.1*np.max(np.abs(lmoyfpb[ipin])))
    bars = plt.bar(x,np.abs(lmoyfpb[ipin]),width=time_interval,align='edge',color=colorpin[ipin],alpha=0.8)

    # Rounding the edges
    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
        bar.set_capstyle('round')
        # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

    for i,xi in enumerate(x):
        ax.annotate(r'$\sigma :{%d}$' % (lsigmafpb[ipin][i]) , 
        (xi, np.abs(lmoyfpb[ipin][i])),
        xytext=(1,4), textcoords='offset points',
        family='sans-serif', fontsize=9, color='black') 

    ax.set_ylabel("Mean Normal Force (N)",fontsize=12)
    ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
    plt.xticks(rotation=0, ha='right')
    if figshow:
        plt.show()
    title=f"{repsect1}FN_pb{ipin+1}"
    fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 

#%% mean dissipated energy :
colorpin = ['red','green','blue']
ymaxe = np.max([ np.max(np.abs(fi)*1.e4) for i,fi in  enumerate(lmoyuspb) ])
for ipin in np.arange(3): 
    fig = plt.figure(figsize=(8,6), dpi=1000)
    ax=plt.axes()
    ax.set_ylim(ymax=1.1*ymaxe)
    # ax.set_ylim(ymax=1.1*np.max(np.abs(lmoyuspb[ipin])*1.e4))
    bars = plt.bar(x,np.abs(lmoyuspb[ipin])*1.e4,width=time_interval,align='edge',color=colorpin[ipin],alpha=0.8)

    # Rounding the edges
    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
        bar.set_capstyle('round')
        # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

    for i,xi in enumerate(x):
        ax.annotate(r'$\sigma :$' + r"${:.2f}$".format(lsigmauspb[ipin][i]*1.e4), 
        (xi, np.abs(lmoyuspb[ipin][i])*1.e4),
        xytext=(1,4), textcoords='offset points',
        family='sans-serif', fontsize=9, color='black') 

    ax.set_ylabel("Mean Wear Energy ("+r"$10^{-4}$"+" J)",fontsize=12)
    ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
    plt.xticks(rotation=0, ha='right')
    if figshow:
        plt.show()
    title=f"{repsect1}Ewear_pb{ipin+1}"
    fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
#%% mean impact time :
colorpin = ['red','green','blue']
ymaxt = np.max([ np.max(np.abs(fi)*1.e3) for i,fi in  enumerate(lmoytpb) ])
for ipin in np.arange(3): 
    fig = plt.figure(figsize=(8,6), dpi=1000)
    ax=plt.axes()
    ax.set_ylim(ymax=1.1*ymaxt)
    # ax.set_ylim(ymax=1.1*np.max(np.abs(lmoytpb[ipin])*1.e3))
    bars = plt.bar(x,np.abs(lmoytpb[ipin])*1.e3,width=time_interval,align='edge',color=colorpin[ipin],alpha=0.8)

    # Rounding the edges
    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
        bar.set_capstyle('round')
        # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

    for i,xi in enumerate(x):
        ax.annotate(r'$\sigma :$' + r"${:.2f}$".format(lsigmatpb[ipin][i]*1.e3), 
        (xi, np.abs(lmoytpb[ipin][i])*1.e3),
        xytext=(1,4), textcoords='offset points',
        family='sans-serif', fontsize=9, color='black') 

    ax.set_ylabel("Mean Impact Time (ms)",fontsize=12)
    ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
    plt.xticks(rotation=0, ha='right')
    if figshow:
        plt.show()
    title=f"{repsect1}tchoc_pb{ipin+1}"
    fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 
#%%

sys.exit()
#%%
lmoyA0 = []
lsigmaA0 = []
[ lnchocpb.append(len(df.iloc[indi]['FN_pb'])) for i,indi in enumerate(indi_pb_split) ]
[ lmoyA0.append(np.mean(ai)) for i,ai in enumerate(A0) ]
[ lsigmaA0.append(np.std(ai)) for i,ai in enumerate(A0) ]


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
# hpeaks = [50.].append([80.]*(nbseg-1))
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

    A0.append(np.concatenate((fi.iloc[peakspos],-fi.iloc[peaksneg])))

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
[ lmoyA0.append(np.mean(ai)) for i,ai in enumerate(A0) ]
[ lsigmaA0.append(np.std(ai)) for i,ai in enumerate(A0) ]

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
bars = plt.bar(x,lmoyA0,width=time_interval,align='edge',color='green',alpha=0.8)
# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
    # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

ax.set_ylabel("Mean Impact Force Amplitude (N)",fontsize=12)
ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
plt.xticks(rotation=0, ha='right')
plt.show()
title=f"{rep_save}mean_peak_amplitude"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 

# standard deviation of peaks amplitude :
fig = plt.figure(figsize=(8,6), dpi=1000)
ax=plt.axes()
# ax.set_xlim(xmin=0.,xmax=df.iloc[-1]['tL'])
bars = plt.bar(x,lsigmaA0,width=time_interval,align='edge',color='blue',alpha=0.8)
# Rounding the edges
for bar in bars:
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    bar.set_capstyle('round')
    # bar.set_path_effects([PathPatch.PathPatchEffect(capstyle='round')])

ax.set_ylabel("Impact Force Amplitude Standard Deviation (N)",fontsize=12)
ax.set_xlabel(r"$t$"+" (s)",fontsize=12)
plt.xticks(rotation=0, ha='right')
plt.show()
title=f"{rep_save}std_peak_amplitude"
fig.savefig(title+".png",bbox_inches='tight',facecolor='white') 

#%%
plt.bar(np.arange(len(indpeaks)),lmoy)
# plt.show()
plt.close('all')
plt.bar(np.arange(len(indpeaks)),lsigma)
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
