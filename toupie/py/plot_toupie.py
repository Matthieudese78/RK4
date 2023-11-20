# import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy as sp
import pylab
import matplotlib.ticker as mtick
from matplotlib.pyplot import rcParams as rc
# Pour les couleurs : 
import matplotlib.cm as cm
# from decimal import Decimal

color1 = ['blue', 'red', 'green', 'orange', 'purple','pink' , 'cyan']

#%%
# Noms des colonnes : 
# ['RAIDEUR_MOD','RAIDEUR_STAT','RATIO_SM','COOR']

WX = "WX"
WY = "WY"
WZ = "WZ"
t = "t"

#%% repertoires:
repertoire = "/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/dev_git/rotations_rigides/dgibi_trav/toupie/data_dgibi/"   

# repertoire = "../data/FZ/"   

rep_sauv = "/home/matthieu/Documents/Cast3M/corps_rigide_castem/fortran/dev_git/rotations_rigides/dgibi_trav/toupie/py/png/"
#%% Script


#%% TESTS DATAFRAMES:

# =============================================================================
# df_W = pd.read_csv(repertoire + script + ".csv",delimiter=(';'))
# df_2 = pd.read_csv(repertoire + script + ".csv")
# df_3 = pd.read_csv(repertoire + script + ".csv",delimiter=(';'),header=None)
# 
# df_5 = pd.read_csv(repertoire + script + ".csv",delimiter=(';'),header=0)
# df_6= pd.read_csv(repertoire + script + ".csv",delimiter=(';'),header=0,index_col=0)
# df_7= pd.read_csv(repertoire + script + ".csv",delimiter=(';'),header=0,index_col=None)
# 
# df_4 = pd.read_csv(repertoire + script + ".csv",delimiter=(','),header=0)
# =============================================================================
#%% BON DATATFRAME : 
    # vitesse de rotation
script = "W_toupie"
df_W = pd.read_csv(repertoire + script + ".csv",delimiter=(','),header=0)
    # moment cinétique
script = "Pi_toupie"
df_Pi = pd.read_csv(repertoire + script + ".csv",delimiter=(','),header=0)
    # energie cinétique
script = "Ec_toupie"
df_Ec = pd.read_csv(repertoire + script + ".csv",delimiter=(','),header=0)

#%% AFFICHER LES NOMS DE COLUMNS : EXEMPLES
# =============================================================================
# df_4.columns
# df_4.keys()
# list(df_4.columns)
# sorted(df_4)
# col_name = df_4[df_4.columns[0]]
# =============================================================================

#%% nombre de colonnes
Lcol = len(df_W.columns)
Lrow = len(df_W[df_W.columns[0]][:])
#%% Utilisation de .iloc pour afficher les valeurs du DataFrame :
Lcol = len(df_W.iloc[0])
#%% Longueur des strings = noms des colonnes
Lname = len(df_W.columns[0])

#%% On complète les noms de colonnes avec des espaces :

WX = ("WX").ljust(Lname," ")
WY = ("WY").ljust(Lname," ")
WZ = ("WZ").ljust(Lname," ")

PIX = ("PIX").ljust(Lname," ")
PIY = ("PIY").ljust(Lname," ")
PIZ = ("PIZ").ljust(Lname," ")

EC = ("EC").ljust(Lname," ")

t = ("t").ljust(Lname," ")

#%% Echantillonage du temps
# temps total :
tfin = df_W[t][Lrow-1]
# Frame Per Second :
FPS_gif = 5
# pas de temps pour le GIF
FPS_data = Lrow/tfin
# frequence d'echantillonage du signal d'entree 
F_ech = 1./(FPS_data/FPS_gif)
T_ech = np.round((FPS_data/FPS_gif))
T_ech = int(T_ech)
# Nombre de pas de temps :
NPDT = int(np.round(Lrow*F_ech))
# NPDT = int(NPDT/10)
#%% LOOP
# WX :
i=0
for i in np.arange(NPDT):
    fig = plt.figure(figsize=(8,6), dpi=220)
    # plt.plot(df_W[t][0:int((i)*T_ech)], df_W[WX][0:int((i)*T_ech)], c = cm.hsv(1/Lrow),label=(r'$W_x$'))
    plt.plot(df_W[t][0:int((i)*T_ech)], df_W[WX][0:int((i)*T_ech)], c = 'r',label=(r'$W_x$'))
    plt.plot()
    ax = plt.gca()
    ax.set_ylim(-1.05*np.max(df_W[WX]),1.05*np.max(df_W[WX]))
    ax.set_xlim(0.,1.02*tfin)
    ax.set_facecolor('None')
    ax.grid(linestyle='-', linewidth='0.25', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=16)         
    plt.xlabel('t')
    plt.ylabel(r'$W_x$')
    fig.savefig(rep_sauv + "WX_"+str(i)+".png", bbox_inches='tight')
    i+=i
    
# WY :
i=0
for i in np.arange(NPDT):
    fig = plt.figure(figsize=(8,6), dpi=220)
    # plt.plot(df_W[t][0:int((i)*T_ech)], df_W[WY][0:int((i)*T_ech)], c = cm.hsv(50/Lrow),label=(r'$W_y$'))
    plt.plot(df_W[t][0:int((i)*T_ech)], df_W[WY][0:int((i)*T_ech)], c = 'g',label=(r'$W_y$'))
    plt.plot()
    ax = plt.gca()
    ax.set_ylim(-1.05*np.max(df_W[WY]),1.05*np.max(df_W[WY]))
    ax.set_xlim(0.,1.02*tfin)
    ax.set_facecolor('None')
    ax.grid(linestyle='-', linewidth='0.25', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=16)         
    plt.xlabel('t')
    plt.ylabel(r'$W_y$')
    fig.savefig(rep_sauv + "WY_"+str(i)+".png", bbox_inches='tight')
    i+=i

# WZ :
i=0
for i in np.arange(NPDT):
    fig = plt.figure(figsize=(8,6), dpi=220)
    # plt.plot(df_W[t][0:int((i)*T_ech)], df_W[WZ][0:int((i)*T_ech)], c = cm.hsv(80/Lrow),label=(r'$W_z$'))
    plt.plot(df_W[t][0:int((i)*T_ech)], df_W[WZ][0:int((i)*T_ech)], c = 'b',label=(r'$W_z$'))
    plt.plot()
    ax = plt.gca()
    ax.set_ylim(0.,1.1*np.max(df_W[WZ]))
    ax.set_xlim(0.,1.02*tfin)
    ax.set_facecolor('None')
    ax.grid(linestyle='-', linewidth='0.25', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=16)         
    plt.xlabel('t')
    plt.ylabel(r'$W_z$')
    fig.savefig(rep_sauv + "WZ_"+str(i)+".png", bbox_inches='tight')
    i+=i
    
#%% Moment cinetique :
# PIX :
i=0
for i in np.arange(NPDT):
    fig = plt.figure(figsize=(8,6), dpi=220)
    # plt.plot(df_Pi[t][0:int((i)*T_ech)], df_Pi[PIX][0:int((i)*T_ech)], c = cm.hsv(1/Lrow),label=(r'$\Pi_x$'))
    plt.plot(df_Pi[t][0:int((i)*T_ech)], df_Pi[PIX][0:int((i)*T_ech)], c = 'r',label=(r'$\Pi_x$'))
    plt.plot()
    ax = plt.gca()
    ax.set_ylim(-1.05*np.max(df_Pi[PIX]),1.05*np.max(df_Pi[PIX]))
    ax.set_xlim(0.,1.02*tfin)
    ax.set_facecolor('None')
    ax.grid(linestyle='-', linewidth='0.25', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=16)         
    plt.xlabel('t')
    plt.ylabel(r'$\Pi_x$')
    fig.savefig(rep_sauv + "PIX_"+str(i)+".png", bbox_inches='tight')
    i+=i
    
# PIY :
i=0
for i in np.arange(NPDT):
    fig = plt.figure(figsize=(8,6), dpi=220)
    # plt.plot(df_Pi[t][0:int((i)*T_ech)], df_Pi[PIY][0:int((i)*T_ech)], c = cm.hsv(50/Lrow),label=(r'$\Pi_y$'))
    plt.plot(df_Pi[t][0:int((i)*T_ech)], df_Pi[PIY][0:int((i)*T_ech)], c = 'g',label=(r'$\Pi_y$'))
    plt.plot()
    ax = plt.gca()
    ax.set_ylim(-1.05*np.max(df_Pi[PIY]),1.05*np.max(df_Pi[PIY]))
    ax.set_xlim(0.,1.02*tfin)
    ax.set_facecolor('None')
    ax.grid(linestyle='-', linewidth='0.25', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=16)         
    plt.xlabel('t')
    plt.ylabel(r'$\Pi_y$')
    fig.savefig(rep_sauv + "PIY_"+str(i)+".png", bbox_inches='tight')
    i+=i

# PIZ :
i=0
for i in np.arange(NPDT):
    fig = plt.figure(figsize=(8,6), dpi=220)
    # plt.plot(df_Pi[t][0:int((i)*T_ech)], df_Pi[PIZ][0:int((i)*T_ech)], c = cm.hsv(80/Lrow),label=(r'$\Pi_z$'))
    plt.plot(df_Pi[t][0:int((i)*T_ech)], df_Pi[PIZ][0:int((i)*T_ech)], c = 'b',label=(r'$\Pi_z$'))
    plt.plot()
    ax = plt.gca()
    ax.set_ylim(0.,1.1*np.max(df_Pi[PIZ]))
    ax.set_xlim(0.,1.02*tfin)
    ax.set_facecolor('None')
    ax.grid(linestyle='-', linewidth='0.25', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=16)         
    plt.xlabel('t')
    plt.ylabel(r'$\Pi_z$')
    fig.savefig(rep_sauv + "PIZ_"+str(i)+".png", bbox_inches='tight')
    i+=i
    
#%% Energie cinetique : 
i=0
for i in np.arange(NPDT):
    fig = plt.figure(figsize=(8,6), dpi=220)
    # plt.plot(df_Ec[t][0:int((i)*T_ech)], df_Ec[EC][0:int((i)*T_ech)], c = cm.hsv(80/Lrow),label=(r'$Ec$'))
    plt.plot(df_Ec[t][0:int((i)*T_ech)], df_Ec[EC][0:int((i)*T_ech)], c = 'b',label=(r'$Ec$'))
    plt.plot()
    ax = plt.gca()
    ax.set_ylim(0.,1.1*np.max(df_Ec[EC]))
    ax.set_xlim(0.,1.02*tfin)
    ax.set_facecolor('None')
    ax.grid(linestyle='-', linewidth='0.25', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=16)         
    plt.xlabel('t')
    plt.ylabel(r'$Ec$')
    fig.savefig(rep_sauv + "EC_"+str(i)+".png", bbox_inches='tight')
    i+=i

    
    
    