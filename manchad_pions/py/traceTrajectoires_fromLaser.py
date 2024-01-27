# -*- coding: utf-8 -*-
import json
import pickle
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
import matplotlib.figure
import os.path as osp
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
# import cv2
import copy
import os
from scipy.interpolate import interp1d
import scipy.signal


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


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

def desaturate(signal,lt=-np.Inf,ht=np.Inf) : 
    # si valeur signal < lt ou valeur signal > ht : on prend la derniere valeur connue
    # dans l'intervalle [lt,bt]
    prev = np.arange(len(signal))
    prev[(signal < lt)|(signal>ht)] = 0
    prev = np.maximum.accumulate(prev)
    return signal[prev]

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

# tracé du mouvement relatif de la manchette dans l'adaptateur
# à 4 hauteurs : 
# haut :                z = 878.5mm (28.5 en dessous du haut de la manchette)
# pions haut :          z = 697mm
# pions bas             z = 304mm
# bas de l'adaptateur : z = 189.5mm (en config usee)
#
# diametre ext manchette  = 63.5mm
# diametre ext pions      = 68.83mm
# diametre int adaptateur = 70mm
# extrait mail a Roch le 6/11/19
# J’ai le plan des manchettes de remplacement joint. Si je lis bien : 
# -	Diametre du fut : 63.5mm
# -	Diametre des plots de centrage : 68.83mm

# Si je mesure une manchette récupérée à l’ALN :
# -	Diamètre du fut 64.5 mm
# -	Diamètre des plots : 68.3 mm (je mesure fut + un plot à 66.4mm -> delta du au plot = 1.9mm -> diametre fut + 2 plots : 68.3mm)
# 
# le jeu pour les pions : on peut considérer que si un pion touche  dans un sens, les deux autres ne touchent pas dans l'autre sens
# donc le jeu : 70 - 64.5 - 1.9 = 3.6

trigger_level = 2

nbseg=10

show = False
# show = True

# manchette = 'usee'
manchette = 'neuve'

# cas = '20200623_1635'
# cas = '20200623_1645'
# cas = '20200623_1700'
# cas = '20200623_1710'
# cas = '20200623_1715'
# cas = '20200625_1200'
# cas = '20200625_1525'
# cas = '20200625_1550'
# manchette usee

lcasusee=[
 '20201126_1036',
 '20201126_1056',
 '20201126_1111',
 '20201126_1117',
 '20201126_1125',
 '20201126_1138',
 '20201126_1142',
 '20201126_1151',
 '20201126_1426',
 '20201126_1450',
 '20201126_1455',
 '20201126_1500',
 '20201126_1503',
 '20201126_1507',
 '20201126_1511',
 '20201126_1515',
 '20201126_1520',
 '20201126_1523',
 '20201126_1527',
 '20201126_1531',
 '20201126_1534',
 '20201126_1543',
 '20201126_1553',
 '20201126_1558',
 '20201126_1721',
 '20201126_1735',
 '20201126_1743',
 '20201126_1750',
 '20201126_1800',
 '20201126_1812',
 '20201126_1820',
 '20201126_1824',
 '20201126_1828',
 '20201126_1831',
 '20201126_1835',
 '20201126_1838',
 '20201126_1841',
 '20201126_1845',
 '20201126_1853',
 '20201126_1856',
 '20201126_1900',
 '20201126_1903',
 '20201126_1906',
 '20201126_1910',
 '20201126_1915',
 '20201126_1919',
 '20201126_1922',
 '20201126_1925',
 '20201126_1928',
 '20201126_1931',
 '20201126_1934',
 '20201127_0931',
 '20201127_0937',
 '20201127_0951']

# manchette neuve
lcasneuf = ['20201127_1350']
#  '20201127_1350',
#  '20201127_1406',
#  '20201127_1409',
#  '20201127_1413',
#  '20201127_1416',
#  '20201127_1419',
#  '20201127_1422',
#  '20201127_1427',
#  '20201127_1429',
#  '20201127_1436',
#  '20201127_1448',
#  '20201127_1454',
#  '20201127_1457',
#  '20201127_1500',
#  '20201127_1505',
#  '20201127_1507',
#  '20201127_1511',
#  '20201127_1515',
#  '20201127_1525',
#  '20201127_1545',
#  '20201127_1549',
#  '20201127_1552',
#  '20201127_1602',
#  '20201127_1605',
#  '20201127_1608',
#  '20201127_1611',
#  '20201127_1622',
#  '20201127_1631',
#  '20201127_1635',
#  '20201127_1640',
#  '20201127_1643',
#  '20201127_1646',
#  '20201127_1649',
#  '20201127_1652',
#  '20201127_1657',
#  '20201127_1700',
#  '20201127_1703',
#  '20201127_1706',
#  '20201127_1709',
#  '20201127_1712',
#  '20201127_1715',
#  '20201127_1719',
#  '20201127_1722',
#  '20201127_1726',
#  '20201127_1730',
#  '20201130_0905',
#  '20201130_0918']

if manchette == 'neuve' :
    lcas = lcasneuf

    jeuH = 1. # au niveau de la collerette ie  contact cône-cône?
    jeuB = 6.5 # sortie d'adaptateur?
    jeuPH = 2. # pion haut
    jeuPB = 2. # pion bas
    hpion = 2.4 # ??
else :
    lcas = lcasusee

    jeuH = 1.
    jeuB = 6.5
    jeuPH = 2.
    jeuPB = 6.5
    hpion = 2.4



lcasshow = lcas
# lcasshow = [
#  '20201126_1036',
#  '20201126_1056',
#  '20201126_1111',
#  '20201126_1117',
#  '20201126_1125',
#  '20201126_1138',
#  '20201126_1142',
#  '20201126_1151',
#  '20201126_1426',
#  '20201126_1450',
#  '20201126_1455',
#  '20201126_1500',
#  '20201126_1503',
#  '20201126_1507',
#  '20201126_1511',
#  '20201126_1515',
#  '20201126_1520',
#  '20201126_1523',
#  '20201126_1527',
#  '20201126_1531',
#  '20201126_1534',
#  '20201126_1543',
#  '20201126_1553',
#  '20201126_1558',
#  '20201126_1721',
#  '20201126_1735',
#  '20201126_1743',
#  '20201126_1750',
#  '20201126_1800',
#  '20201126_1812',
#  '20201126_1820',
#  '20201126_1824',
#  '20201126_1828',
#  '20201126_1831',
#  '20201126_1835',
#  '20201126_1838',
#  '20201126_1841',
#  '20201126_1845',
#  '20201126_1853',
#  '20201126_1856',
#  '20201126_1900',
#  '20201126_1903',
#  '20201126_1906',
#  '20201126_1910',
#  '20201126_1915',
#  '20201126_1919',
#  '20201126_1922',
#  '20201126_1925',
#  '20201126_1928',
#  '20201126_1931',
#  '20201126_1934',
#  '20201127_0931',
#  '20201127_0937',
#  '20201127_0951',
#  '20201127_1350',
#  '20201127_1406',
#  '20201127_1409',
#  '20201127_1413',
#  '20201127_1416',
#  '20201127_1419',
#  '20201127_1422',
#  '20201127_1427',
#  '20201127_1429',
#  '20201127_1436',
#  '20201127_1448',
#  '20201127_1454',
#  '20201127_1457',
#  '20201127_1500',
#  '20201127_1505',
#  '20201127_1507',
#  '20201127_1511',
#  '20201127_1515',
#  '20201127_1525',
#  '20201127_1545',
#  '20201127_1549',
#  '20201127_1552',
#  '20201127_1602',
#  '20201127_1605',
#  '20201127_1608',
#  '20201127_1611',
#  '20201127_1622',
#  '20201127_1631',
#  '20201127_1635',
#  '20201127_1640',
#  '20201127_1643',
#  '20201127_1646',
#  '20201127_1649',
#  '20201127_1652',
#  '20201127_1657',
#  '20201127_1700',
#  '20201127_1703',
#  '20201127_1706',
#  '20201127_1709',
#  '20201127_1712',
#  '20201127_1715',
#  '20201127_1719',
#  '20201127_1722',
#  '20201127_1726',
#  '20201127_1730',
#  '20201130_0905',
#  '20201130_0918'
#  ]



# repertoire = '/home/J03245/Documents/INTERNES/MANCHETTE/ESSAI/VIDEOS2/DATA'
# repertoireOut = '/home/J03245/Documents/INTERNES/MANCHETTE/ESSAI/VIDEOS2/OUT'
# repertoire = 'data/donneesLaser/'
repertoire = '/home/matthieu/Documents/EDF/mesures/data/donneesLaser/'
repertoireOut = './fig/out/'

if not os.path.exists(repertoireOut):
    os.makedirs(repertoireOut)
    print(f"FOLDER : {repertoireOut} created.")
else:
    print(f"FOLDER : {repertoireOut} already exists.")
icas=0

# moyenneLaser = 'manuel'
moyenneLaser = 'auto'

for cas in lcas :
    icas+=1
    if cas in lcasshow :
        moyenne = {}
        # filename = osp.join(repertoire,'%s_laser.pickle'%cas)
        filename = (repertoire + '%s_laser.pickle'%cas)       
        with open(filename, 'rb') as pickle_file:          
            mesures_laser = pickle.load(pickle_file)

        for i in range(len(mesures_laser['TTL'])) :
            if mesures_laser['TTL'][i] > trigger_level :
                imin = i
                break
        ltags=['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','Courant','Tension','TTL','Force','tL']
        for tag in ltags : 
            mesures_laser[tag]=mesures_laser[tag][i:]

        if manchette == 'neuve':
            colorstandard = 'b'
            # manchette neuve
            if icas > 44 :
                bloquee = True
            else :
                bloquee = False
            pointH  = np.array([0,0,878])
            pointPH = np.array([0,0,697])
            pointPB = np.array([0,0,304])
            pointB  = np.array([0,0,189])
            # position centrée des lasers issue de WN manchette bloquee
            if moyenneLaser == 'manuel' :
                if icas < 14 or icas == 45 :
                    moyenne[ 'L1' ]= 0.2
                elif icas == 46 :
                    moyenne[ 'L1' ]= -0.2
                else :
                    moyenne[ 'L1' ]= 0.15
                moyenne[ 'L2' ]= -2.25
                if icas > 22 and icas < 33 :
                    moyenne[ 'L3' ]= -3.0
                elif icas == 46 :
                    moyenne[ 'L3' ]= -2.6
                else :
                    moyenne[ 'L3' ]= -2.9
                if icas < 13 :
                    moyenne[ 'L5' ]= 1.0
                elif icas < 33 :
                    moyenne[ 'L5' ]= 0.8
                elif icas ==46  :
                    moyenne[ 'L5' ]= -0.1
                else :
                    moyenne[ 'L5' ]= 0.65
                moyenne[ 'L6' ]= -0.2
                moyenne[ 'L7' ]= -2
                moyenne[ 'L8' ]= -0.5
                moyenne[ 'L9' ]= -3.6
                moyenne[ 'L10'] = 1.0
            else :
                ltags=['L1','L2','L3','L5','L6','L7','L8','L9','L10']
                for tag in ltags :
                    moyenne[tag] = np.mean(mesures_laser[tag])
            # position centrée des lasers issue de WN manchette libre

            # O1 pour manchette neuve
            O10 = 84
            O1=O10
            if icas < 2 :
                O1=O10
            elif icas < 8 :
                O1=O10-180
            elif icas < 15 :
                O1=O10
            elif icas < 17 :
                O1=O10-180
            elif icas < 22 :
                O1=O10
            elif icas < 28 :
                O1=O10-180
            elif icas < 36 :
                O1=O10
            elif icas < 41 :
                O1=O10-180
            else :
                O1=O10
            # manchette neuve
            pL1 = np.array([0,-15,715])
            pL2 = np.array([0,-100,224])
            pL3 = np.array([0,100,224])
            pL4 = np.array([0,0,0])
            pL5 = np.array([0,0,986])
            pL6 = np.array([0,0,985])
            pL7 = np.array([0,0,-551])
            pL8 = np.array([0,0,-547])
            if bloquee :
                # manchette neuve avec bloqueur
                # pL9 = np.array([0,0,107])
                pL9 = np.array([0,0,107])
                pL10= np.array([0,0,104])
            else :
                # manchette neuve libre
                pL9 = np.array([0,0,139])
                pL10= np.array([0,0,135])
        else :
            # manchette usee
            colorstandard = 'r'
            pointH  = np.array([0,0,866])
            pointPH = np.array([0,0,685])
            pointPB = np.array([0,0,292])
            pointB  = np.array([0,0,177])

            if icas>8 :
                bloquee = False
            else :
                bloquee = True
            #O1 pour manchette usée 
            O10 = -59

            if icas < 14 :
                O1=O10
            elif icas < 20 :
                O1=O10-180
            elif icas < 27 :
                O1=O10
            elif icas < 28 :
                O1=O10-180
            elif icas < 35 :
                O1=O10
            elif icas < 41 :
                O1=O10-180
            elif icas < 48 :
                O1=O10
            elif icas < 54 :
                O1=O10-180
            else :
                O1=O10

            if (moyenneLaser == 'manuel') :
            # position centrée des lasers issue de verif_laser
                if icas<4 :
                    moyenne[ 'L1' ]= 1.3 #3 premiers
                elif icas<5 :
                    moyenne[ 'L1' ]= 0.8 #4e
                elif icas<8 :
                    moyenne[ 'L1' ]= 1.  #5-7
                elif icas<25 :
                    moyenne[ 'L1' ]= 1.1 # jusqu'a '20201126_1558' inclus
                elif icas == 54 :
                    moyenne[ 'L1' ]= 0.7
                    # print("icas = 54")
                else :
                    moyenne[ 'L1' ]= 0.6 # apres
                if icas<3 :
                    moyenne[ 'L2' ]= -1.1 # 2 premiers
                    moyenne[ 'L3' ]= -1.5 # 2 premiers
                elif icas<4 :
                    moyenne[ 'L2' ]= -1.5 # 3
                    moyenne[ 'L3' ]= -2 # 3
                elif icas<8 :
                    moyenne[ 'L2' ]= -1.4 # 4-7
                    moyenne[ 'L3' ]= -1.9 # 4-7
                elif icas<25 :
                    moyenne[ 'L2' ]= -1.5 # jusqu'a '20201126_1558' inclus
                    moyenne[ 'L3' ]= -2.1 # jusqu'a '20201126_1558' inclus
                elif icas == 54 :
                    moyenne[ 'L2' ]= -1.6
                    moyenne[ 'L3' ]= -2.3
                else :
                    moyenne[ 'L2' ]= -1.7 # apres
                    moyenne[ 'L3' ]= -2.3 # apres
                if icas<3 :        
                    moyenne[ 'L5' ]= 1.2 # 1-2
                elif icas<4 :
                    moyenne[ 'L5' ]= 1.5 # 3
                elif icas<25 :
                    moyenne[ 'L5' ]= 0.9 # jusqu'a '20201126_1558' inclus
                elif icas == 54 :
                    moyenne[ 'L5' ]= 0.
                else :
                    moyenne[ 'L5' ]= 0.2
                if icas == 54 :
                    moyenne[ 'L6' ]= 0
                    moyenne[ 'L7' ]= -1.3 # pas top ca oscille tout le temps
                    moyenne[ 'L8' ]= 0.9
                    moyenne[ 'L9' ]= -1.7 # pas top non plus
                    moyenne[ 'L10'] = 1.45
                else :
                    moyenne[ 'L6' ]= -0.06
                    moyenne[ 'L7' ]= -1.5 # pas top ca oscille tout le temps
                    moyenne[ 'L8' ]= 0.2
                    moyenne[ 'L9' ]= -1.6 # pas top non plus
                    moyenne[ 'L10'] = 1.2
            else :
                ltags=['L1','L2','L3','L5','L6','L7','L8','L9','L10']
                for tag in ltags :
                    moyenne[tag] = np.mean(mesures_laser[tag])
        # manchette usee
            pL1 = np.array([0,-15,715])
            pL2 = np.array([0,-100,224])
            pL3 = np.array([0,100,224])
            pL4 = np.array([0,0,0])
            pL5 = np.array([0,0,974])
            pL6 = np.array([0,0,973])
            pL7 = np.array([0,0,-607])
            pL8 = np.array([0,0,-603])
            if bloquee :
                # manchette usee avec bloqueur
                pL9 = np.array([0,0,107])
                pL10= np.array([0,0,104])
            else :
                # manchette usee libre
                pL9 = np.array([0,0,139])
                pL10= np.array([0,0,135])
        
        # print(mesures_laser['L4'][:10])
        mesures_laser['L4'] = unwrap(mesures_laser['L4'],1)
        # print(mesures_laser['L4'][:10])
        mesures_laser['tL'] = mesures_laser['tL']-mesures_laser['tL'][0]
        

        ltags=['L1','L2','L3','L5','L6','L7','L8','L9','L10']
        for tag in ltags :
            mesures_laser[tag]=mesures_laser[tag]-moyenne[tag]


        xA  = (pL1[2]*(mesures_laser['L2']+mesures_laser['L3'])-2*pL2[2]*mesures_laser['L1'])/(2*pL1[2]-2*pL2[2])
        tyA = (mesures_laser['L1']-xA)/pL1[2]
        tzA = (mesures_laser['L2']-mesures_laser['L3'])/(2*pL3[1])

        tAL = mesures_laser['tL']
        fs = 1./(tAL[1]-tAL[0])

        forceFiltered1 = butter_bandstop_filter(mesures_laser['Force'],50,2,fs)
        forceFiltered2 = butter_bandstop_filter(forceFiltered1,150,2,fs)
        forceFiltered3 = butter_bandstop_filter(forceFiltered2,250,2,fs)
        forceFiltered4 = butter_bandstop_filter(forceFiltered3,350,2,fs)
        forceFiltered = butter_bandstop_filter(forceFiltered4,450,2,fs)

        courantFiltered1 = butter_bandstop_filter(mesures_laser['Courant'],50,2,fs)
        courantFiltered2 = butter_bandstop_filter(courantFiltered1,150,2,fs)
        courantFiltered3 = butter_bandstop_filter(courantFiltered2,250,2,fs)
        courantFiltered4 = butter_bandstop_filter(courantFiltered3,350,2,fs)
        courantFiltered = butter_bandstop_filter(courantFiltered4,450,2,fs)


        tensionFiltered1 = butter_bandstop_filter(mesures_laser['Tension'],50,2,fs)
        tensionFiltered2 = butter_bandstop_filter(tensionFiltered1,150,2,fs)
        tensionFiltered3 = butter_bandstop_filter(tensionFiltered2,250,2,fs)
        tensionFiltered4 = butter_bandstop_filter(tensionFiltered3,350,2,fs)
        tensionFiltered = butter_bandstop_filter(tensionFiltered4,450,2,fs)

        # courantFilteredB = butter_lowpass_filter(mesures_laser['Courant'],40,fs)
        courantFilteredB = butter_lowpass_filter(courantFiltered,35,fs)
        forceFilteredB = butter_lowpass_filter(mesures_laser['Force'],35,fs)

        # depl a partir de haut et medium
        xM  = (-pL10[2]*mesures_laser['L5']+pL5[2]*mesures_laser['L9'])/(pL5[2]-pL9[2])
        yM  = (-pL9[2]*mesures_laser['L6']+pL6[2]*mesures_laser['L10'])/(pL6[2]-pL10[2])
        tyM = (mesures_laser['L5']-xM)/pL5[2]
        txM = (mesures_laser['L10']-yM)/pL10[2]

        # depl a partir de medium et bas
        xM2  = (-pL8[2]*mesures_laser['L9']+pL9[2]*mesures_laser['L7'])/(pL9[2]-pL7[2])
        yM2  = (-pL7[2]*mesures_laser['L10']+pL10[2]*mesures_laser['L8'])/(pL10[2]-pL8[2])
        tyM2 = (mesures_laser['L9']-xM2)/pL9[2]
        txM2 = (mesures_laser['L8']-yM2)/pL8[2]

        # depl a partir de haut et bas
        xM3  = (-pL8[2]*mesures_laser['L5']+pL5[2]*mesures_laser['L7'])/(pL5[2]-pL7[2])
        yM3  = (-pL7[2]*mesures_laser['L6']+pL6[2]*mesures_laser['L8'])/(pL6[2]-pL8[2])
        tyM3 = (mesures_laser['L5']-xM3)/pL5[2]
        txM3 = (mesures_laser['L8']-yM3)/pL8[2]

        posHx  = (xM-xA)+(tyM-tyA)*pointH[2]
        posHy  = yM + txM * pointH[2]
        posPHx  = (xM-xA)+(tyM-tyA)*pointPH[2]
        posPHy  = yM + txM * pointPH[2]
        posBx  = (xM-xA)+(tyM-tyA)*pointB[2]
        posBy  = yM + txM * pointB[2]
        posPBx  = (xM-xA)+(tyM-tyA)*pointPB[2]
        posPBy  = yM + txM * pointPB[2]

        posHx2  = (xM2-xA)+(tyM2-tyA)*pointH[2]
        posHy2  = yM2 + txM2 * pointH[2]
        posPHx2  = (xM2-xA)+(tyM2-tyA)*pointPH[2]
        posPHy2  = yM2 + txM2 * pointPH[2]
        posBx2  = (xM2-xA)+(tyM2-tyA)*pointB[2]
        posBy2  = yM2 + txM2 * pointB[2]
        posPBx2  = (xM2-xA)+(tyM2-tyA)*pointPB[2]
        posPBy2  = yM2 + txM2 * pointPB[2]

        posHx3  = (xM3-xA)+(tyM3-tyA)*pointH[2]
        posHy3  = yM3 + txM3 * pointH[2]
        posPHx3  = (xM3-xA)+(tyM3-tyA)*pointPH[2]
        posPHy3  = yM3 + txM3 * pointPH[2]
        posBx3  = (xM3-xA)+(tyM3-tyA)*pointB[2]
        posBy3  = yM3 + txM3 * pointB[2]
        posPBx3  = (xM3-xA)+(tyM3-tyA)*pointPB[2]
        posPBy3  = yM3 + txM3 * pointPB[2]

        ymax=5

        

        fig = plt.figure()
        text = fig.suptitle("deplacement relatif manchette vs adaptateur")
        ax1 = plt.subplot(221)
        ax1.plot(tAL,posHx,color=colorstandard,label=("x haut laser"))
        ax1.set_ylim([-ymax,ymax])
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend()
        ax2 = plt.subplot(222)
        ax2.plot(tAL,posPHx,color=colorstandard,label=("x pion haut laser"))
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set_ylim([-ymax,ymax])
        ax2.legend()
        ax3 = plt.subplot(223,sharex=ax1)
        ax3.plot(tAL,posBx,color=colorstandard,label=("x bas laser"))
        ax3.set_ylim([-ymax,ymax])
        ax3.set_xlabel("temps(s)")
        ax3.legend()
        ax4 = plt.subplot(224,sharex=ax2)
        ax4.set_ylim([-ymax,ymax])
        ax4.plot(tAL,posPBx,color=colorstandard,label=("x pion bas laser"))
        ax4.set_xlabel("temps(s)")
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax4.legend()
        # plt.show()
        if not show :
            # printfile=osp.join(repertoireOut,'%s_deplRelatifX.png'%cas)
            printfile = (repertoireOut + '%s_deplRelatifX.png'%cas)
            plt.savefig(printfile)
            plt.close()


        fig = plt.figure()
        teyt = fig.suptitle("deplacement relatif manchette vs adaptateur")
        ay1 = plt.subplot(221)
        ay1.plot(tAL,posHy,color=colorstandard,label=("y haut laser"))
        ay1.legend()
        ay1.set_ylim([-ymax,ymax])
        plt.setp(ay1.get_xticklabels(), visible=False)
        ay2 = plt.subplot(222)
        ay2.plot(tAL,posPHy,color=colorstandard,label=("y pion haut laser"))
        ay2.legend()
        plt.setp(ay2.get_xticklabels(), visible=False)
        plt.setp(ay2.get_yticklabels(), visible=False)
        ay2.set_ylim([-ymax,ymax])
        ay3 = plt.subplot(223,sharex=ay1)
        ay3.plot(tAL,posBy,color=colorstandard,label=("y bas laser"))
        ay3.legend()
        ay3.set_ylim([-ymax,ymax])
        ay4 = plt.subplot(224,sharex=ay2)
        ay4.plot(tAL,posPBy,color=colorstandard,label=("y pion bas laser"))
        ay4.legend()
        plt.setp(ay4.get_yticklabels(), visible=False)
        ay4.set_ylim([-ymax,ymax])
        if not show :
            printfile=osp.join(repertoireOut,'%s_deplRelatifY.png'%cas)
            plt.savefig(printfile)
            plt.close()


        fig = plt.figure()
        teyt = fig.suptitle("mesures force")
        ay1 = plt.subplot(111)
        # ay1.plot(tAL,forceFiltered,color="k",label=("force"))
        ay1.plot(tAL,forceFilteredB,color=colorstandard,label=("force"))
        ay1.legend()
        ay1.set_xlabel("temps (s)")
        ay1.set_ylabel("force (N)")
        # ay1 = plt.subplot(412)
        # ay1.plot(tAL,mesures_laser['Courant'],color="k",label=("courant"))
        # ay1.plot(tAL,courantFiltered,color=colorstandard,label=("courant"))
        # ay1.plot(tAL,courantFilteredB,color="g",label=("courant"))
        # ay1.legend()
        # ay1 = plt.subplot(413)
        # ay1.plot(tAL,mesures_laser['Tension'],color="g",label=("tension"))
        # ay1.legend()
        # ay1 = plt.subplot(414)
        # ay1.plot(tAL,mesures_laser['TTL'],color="c",label=("TTL"))
        # ay1.legend()
        # ay1.legend()
        if not show :
            printfile=osp.join(repertoireOut,'%s_force.png'%cas)
            plt.savefig(printfile)
            plt.close()

        facteurProminence = 5

        posHr  = np.sqrt(posHx**2+posHy**2)
        posPHr = np.sqrt(posPHx**2+posPHy**2)
        posBr  = np.sqrt(posBx**2+posBy**2)
        posPBr = np.sqrt(posPBx**2+posPBy**2)
        prominenceThrH  = jeuH/facteurProminence
        prominenceThrB  = jeuB/facteurProminence
        prominenceThrPB = jeuPB/facteurProminence
        prominenceThrPH = jeuPH/facteurProminence

        posHr3  = np.sqrt(posHx3**2+posHy3**2)
        posPHr3 = np.sqrt(posPHx3**2+posPHy3**2)
        posBr3  = np.sqrt(posBx3**2+posBy3**2)
        posPBr3 = np.sqrt(posPBx3**2+posPBy3**2)
        
        fig = plt.figure()
        teyt = fig.suptitle("flexion manchette")
        ay1 = plt.subplot(111)
        ay1.plot(tAL,posBr3-posBr,color=colorstandard,label=("ecart radial bas"))
        ay1.legend()
        if not show :
            printfile=osp.join(repertoireOut,'%s_flexion.png'%cas)
            plt.savefig(printfile)
            plt.close()

        peaksH_laser = scipy.signal.find_peaks(posHr,prominence=(prominenceThrH,None))[0]
        peaksPH_laser = scipy.signal.find_peaks(posPHr,prominence=(prominenceThrPH,None))[0]
        peaksB_laser = scipy.signal.find_peaks(posBr,prominence=(prominenceThrB,None))[0]
        peaksPB_laser = scipy.signal.find_peaks(posPBr,prominence=(prominenceThrPB,None))[0]

        fig = plt.figure()
        teyt = fig.suptitle("peaks")
        ay1 = plt.subplot(221)
        ay1.plot(tAL,posHr,color=colorstandard,label=("r haut laser"))
        ay1.plot(tAL[peaksH_laser],posHr[peaksH_laser],"x")
        ay1.legend()
        ay1 = plt.subplot(222)
        ay1.plot(tAL,posPHr,color=colorstandard,label=("r pion haut laser"))
        ay1.plot(tAL[peaksPH_laser],posPHr[peaksPH_laser],"x")
        ay1.legend()
        ay1 = plt.subplot(223)
        ay1.plot(tAL,posBr,color=colorstandard,label=("r bas laser"))
        ay1.plot(tAL[peaksB_laser],posBr[peaksB_laser],"x")
        ay1.legend()
        ay1 = plt.subplot(224)
        ay1.plot(tAL,posPBr,color=colorstandard,label=("r pion bas laser"))
        ay1.plot(tAL[peaksPB_laser],posPBr[peaksPB_laser],"x")
        ay1.legend()
        # plt.show()
        if not show :
            printfile=osp.join(repertoireOut,'%s_peaks.png'%cas)
            plt.savefig(printfile)
            plt.close()

        posHr_split = np.array_split(posHr,nbseg)
        posPHr_split = np.array_split(posPHr,nbseg)
        posBr_split = np.array_split(posBr,nbseg)
        posPBr_split = np.array_split(posPBr,nbseg)
    
        posHr3_split = np.array_split(posHr3,nbseg)
        posPHr3_split = np.array_split(posPHr3,nbseg)
        posBr3_split = np.array_split(posBr3,nbseg)
        posPBr3_split = np.array_split(posPBr3,nbseg)

        peaksH_laser = []
        peaksPH_laser = []
        peaksB_laser=[]
        peaksPB_laser=[]

        peaksH3_laser = []
        peaksPH3_laser = []
        peaksB3_laser=[]
        peaksPB3_laser=[]

        for (h,ph,b,pb) in zip(posHr_split,posPHr_split,posBr_split,posPBr_split) :
            peaksH_laser.append(scipy.signal.find_peaks(h,prominence=(prominenceThrH,None))[0])
            peaksPH_laser.append(scipy.signal.find_peaks(ph,prominence=(prominenceThrPH,None))[0])
            peaksB_laser.append(scipy.signal.find_peaks(b,prominence=(prominenceThrB,None))[0])
            peaksPB_laser.append(scipy.signal.find_peaks(pb,prominence=(prominenceThrPB,None))[0])
        for (h,ph,b,pb) in zip(posHr3_split,posPHr3_split,posBr3_split,posPBr3_split) :
            peaksH3_laser.append(scipy.signal.find_peaks(h,prominence=(prominenceThrH,None))[0])
            peaksPH3_laser.append(scipy.signal.find_peaks(ph,prominence=(prominenceThrPH,None))[0])
            peaksB3_laser.append(scipy.signal.find_peaks(b,prominence=(prominenceThrB,None))[0])
            peaksPB3_laser.append(scipy.signal.find_peaks(pb,prominence=(prominenceThrPB,None))[0])
            
        if manchette == 'usee' :
            # pour manchette usee on trace le bas
            fig = plt.figure(figsize=(10,5))
            teyt = fig.suptitle("extremum de deplacement bas")
            matplotlib.pyplot.subplots_adjust(left=0.05,right=0.95)
            i=0
            c=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
            assert(nbseg%2 == 0)
            jeumax = jeuB/2+.5
            hpion2 = jeumax/10
            p1 = patches.Wedge((0,0),hpion2,-15, 15, color="red")
            p2 = patches.Wedge((0,0),hpion2,120-15,120+15, color="red")
            p3 = patches.Wedge((0,0),hpion2,-120-15,-120+15, color="red")
            posBx_split = np.array_split(posBx,nbseg)
            posBy_split = np.array_split(posBy,nbseg)
            l4_split = np.array_split(mesures_laser['L4'],nbseg)

            for peaks,bx,by,l4 in zip(peaksB_laser,posBx_split,posBy_split,l4_split) :
                # assert(len(peaks)>0)
                i+=1
                ay1 = plt.subplot(2,nbseg//2,i)
                if len(peaks)>0:
                    p = collections.PatchCollection([p1,p2,p3],color=['red','orange','orange'])
                    ay1.scatter(bx[peaks],by[peaks],color=c[8],s=4,alpha=.2)
                    rotman = np.mean(l4[peaks]*10)-O1+180
                    r = transforms.Affine2D().rotate(-rotman/180*np.pi) +  ay1.transData
                    p.set_transform(r)
                    ay1.add_collection(p)
                draw_circle=plt.Circle((0., 0.), jeuB/2,fill=False)
                ay1.add_artist(draw_circle)
                ay1.set_aspect('equal','box')
                ay1.set_ylim([-jeumax,jeumax])
                ay1.set_xlim([-jeumax,jeumax])
                if (i%(nbseg//2) != 1) :
                    ay1.set_yticks([])
                if (i <= nbseg//2) :
                    ay1.set_xticks([])
            if not show :
                printfile=osp.join(repertoireOut,'%s_extremumDeplBas.png'%cas)
                plt.savefig(printfile)
                plt.close()

            # pour manchette usee on trace le bas
            fig = plt.figure(figsize=(10,5))
            teyt = fig.suptitle("extremum de deplacement bas (+flexion)")
            matplotlib.pyplot.subplots_adjust(left=0.05,right=0.95)
            i=0
            c=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
            assert(nbseg%2 == 0)
            jeumax = jeuB/2+.5
            hpion2 = jeumax/10
            p1 = patches.Wedge((0,0),hpion2,-15, 15, color="red")
            p2 = patches.Wedge((0,0),hpion2,120-15,120+15, color="red")
            p3 = patches.Wedge((0,0),hpion2,-120-15,-120+15, color="red")
            posBx_split = np.array_split(posBx3,nbseg)
            posBy_split = np.array_split(posBy3,nbseg)
            l4_split = np.array_split(mesures_laser['L4'],nbseg)

            for peaks,bx,by,l4 in zip(peaksB3_laser,posBx_split,posBy_split,l4_split) :
                # assert(len(peaks)>0)
                i+=1
                ay1 = plt.subplot(2,nbseg//2,i)
                if len(peaks)>0:
                    p = collections.PatchCollection([p1,p2,p3],color=['red','orange','orange'])
                    ay1.scatter(bx[peaks],by[peaks],color=c[7],s=4,alpha=.2)
                    rotman = np.mean(l4[peaks]*10)-O1+180
                    r = transforms.Affine2D().rotate(-rotman/180*np.pi) +  ay1.transData
                    p.set_transform(r)
                    ay1.add_collection(p)
                draw_circle=plt.Circle((0., 0.), jeuB/2,fill=False)
                ay1.add_artist(draw_circle)
                ay1.set_aspect('equal','box')
                ay1.set_ylim([-jeumax,jeumax])
                ay1.set_xlim([-jeumax,jeumax])
                if (i%(nbseg//2) != 1) :
                    ay1.set_yticks([])
                if (i <= nbseg//2) :
                    ay1.set_xticks([])
            if not show :
                printfile=osp.join(repertoireOut,'%s_extremumDeplBasFlexion.png'%cas)
                plt.savefig(printfile)
                plt.close()

        else :
            # pour manchette neuve on trace pion bas
            fig = plt.figure(figsize=(10,5))
            teyt = fig.suptitle("extremum de deplacement pion bas")
            matplotlib.pyplot.subplots_adjust(left=0.05,right=0.95)
            i=0
            c=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
            assert(nbseg%2 == 0)
            jeumax = jeuPB/2+.5
            hpion2 = jeumax/10
            
            p1 = patches.Wedge((0,0),hpion2,-15, 15, color="red")
            p2 = patches.Wedge((0,0),hpion2,120-15,120+15, color="red")
            p3 = patches.Wedge((0,0),hpion2,-120-15,-120+15, color="red")

            posPBx_split = np.array_split(posPBx,nbseg)
            posPBy_split = np.array_split(posPBy,nbseg)
            l4_split = np.array_split(mesures_laser['L4'],nbseg)

            for peaks,pbx,pby,l4 in zip(peaksPB_laser,posPBx_split,posPBy_split,l4_split) :
                # assert(len(peaks)>0)
                i+=1
                ay1 = plt.subplot(2,nbseg//2,i)
                if len(peaks)>0 :
                    p = collections.PatchCollection([p1,p2,p3],color=['red','orange','orange'])
                    ay1.scatter(pbx[peaks],pby[peaks],color=c[2],s=4,alpha=.2)
                    rotman = np.mean(l4[peaks]*10)-O1+180
                    r = transforms.Affine2D().rotate(-rotman/180*np.pi) +  ay1.transData
                    p.set_transform(r)
                    ay1.add_collection(p)
                draw_circle=plt.Circle((0., 0.), jeuPB/2,fill=False)
                ay1.add_artist(draw_circle)
                ay1.set_aspect('equal','box')
                ay1.set_ylim([-jeumax,jeumax])
                ay1.set_xlim([-jeumax,jeumax])
                if (i%(nbseg//2) != 1) :
                    ay1.set_yticks([])
                if (i <= nbseg//2) :
                    ay1.set_xticks([])
                # ay1.legend()
            if not show :
                printfile=osp.join(repertoireOut,'%s_extremumDeplPionBas.png'%cas)
                plt.savefig(printfile)
                plt.close()

            # pour manchette neuve on trace pion bas
            fig = plt.figure(figsize=(10,5))
            teyt = fig.suptitle("extremum de deplacement pion bas + flexion")
            matplotlib.pyplot.subplots_adjust(left=0.05,right=0.95)
            i=0
            c=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
            assert(nbseg%2 == 0)
            jeumax = jeuPB/2+.5
            hpion2 = jeumax/10
            
            p1 = patches.Wedge((0,0),hpion2,-15, 15, color="red")
            p2 = patches.Wedge((0,0),hpion2,120-15,120+15, color="red")
            p3 = patches.Wedge((0,0),hpion2,-120-15,-120+15, color="red")

            posPBx_split = np.array_split(posPBx3,nbseg)
            posPBy_split = np.array_split(posPBy3,nbseg)
            l4_split = np.array_split(mesures_laser['L4'],nbseg)

            for peaks,pbx,pby,l4 in zip(peaksPB3_laser,posPBx_split,posPBy_split,l4_split) :
                # assert(len(peaks)>0)
                i+=1
                ay1 = plt.subplot(2,nbseg//2,i)
                if len(peaks)>0 :
                    p = collections.PatchCollection([p1,p2,p3],color=['red','orange','orange'])
                    ay1.scatter(pbx[peaks],pby[peaks],color=c[7],s=4,alpha=.2)
                    rotman = np.mean(l4[peaks]*10)-O1+180
                    r = transforms.Affine2D().rotate(-rotman/180*np.pi) +  ay1.transData
                    p.set_transform(r)
                    ay1.add_collection(p)
                draw_circle=plt.Circle((0., 0.), jeuPB/2,fill=False)
                ay1.add_artist(draw_circle)
                ay1.set_aspect('equal','box')
                ay1.set_ylim([-jeumax,jeumax])
                ay1.set_xlim([-jeumax,jeumax])
                if (i%(nbseg//2) != 1) :
                    ay1.set_yticks([])
                if (i <= nbseg//2) :
                    ay1.set_xticks([])
                # ay1.legend()
            if not show :
                printfile=osp.join(repertoireOut,'%s_extremumDeplPionBasFlexion.png'%cas)
                plt.savefig(printfile)
                plt.close()

        fig = plt.figure(figsize=(10,5))
        teyt = fig.suptitle("extremum de deplacement pions haut")
        matplotlib.pyplot.subplots_adjust(left=0.05,right=0.95)
        i=0
        c=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
        jeumax = jeuPH/2+.5
        hpion2 = jeumax/10
        
        p1 = patches.Wedge((0,0),hpion2,-15, 15, color="red")
        p2 = patches.Wedge((0,0),hpion2,120-15,120+15, color="red")
        p3 = patches.Wedge((0,0),hpion2,-120-15,-120+15, color="red")
        
        posPHx_split = np.array_split(posPHx,nbseg)
        posPHy_split = np.array_split(posPHy,nbseg)
        l4_split = np.array_split(mesures_laser['L4'],nbseg)

        for peaks,phx,phy,l4 in zip(peaksPH_laser,posPHx_split,posPHy_split,l4_split) :
            # assert(len(peaks)>0)
            i+=1
            # print('peaks %s'%peaks)
            ay1 = plt.subplot(2,nbseg//2,i)
            if len(peaks)>0 :
                p = collections.PatchCollection([p1,p2,p3],color=['red','orange','orange'])
                # ay1.scatter(posPBy[peaks],posPBx[peaks],color=c[i-1],s=4,alpha=.2,label=("pion bas"))
                ay1.scatter(phx[peaks],phy[peaks],color=c[4],s=4,alpha=.2)
                rotman = np.mean(l4[peaks]*10)-O1+180
                r = transforms.Affine2D().rotate(-rotman/180*np.pi) +  ay1.transData
                p.set_transform(r)
                ay1.add_collection(p)
            draw_circle=plt.Circle((0., 0.), jeuPH/2,fill=False)
            ay1.add_artist(draw_circle)
            ay1.set_aspect('equal','box')
            ay1.set_ylim([-jeumax,jeumax])
            ay1.set_xlim([-jeumax,jeumax])
            if (i%(nbseg//2) != 1) :
                ay1.set_yticks([])
            if (i <= nbseg//2) :
                ay1.set_xticks([])
            # ay1.legend()
        if not show :
            printfile=osp.join(repertoireOut,'%s_extremumDeplHaut.png'%cas)
            plt.savefig(printfile)
            plt.close()
        # plt.show()

        posHxSplit = np.array_split(posHx,nbseg)
        posHySplit = np.array_split(posHy,nbseg)

        fig = plt.figure(figsize=(10,5))
        teyt = fig.suptitle("trajectoire haut")
        matplotlib.pyplot.subplots_adjust(left=0.05,right=0.95)
        i=0
        c=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
        assert(nbseg%2 == 0)
        jeumax = jeuH/2+.5
        for hx,hy in zip(posHxSplit,posHySplit) :
            i+=1
            # print('peaks %s'%peaks)
            ay1 = plt.subplot(2,nbseg//2,i)
            draw_circle=plt.Circle((0., 0.), jeuH/2,fill=False)
            ay1.add_artist(draw_circle)
            # ay1.scatter(posPBy[peaks],posPBx[peaks],color=c[i-1],s=4,alpha=.2,label=("pion bas"))
            ay1.scatter(hx,hy,color=c[6],s=4,alpha=.2)
            ay1.set_aspect('equal','box')
            ay1.set_ylim([-jeumax,jeumax])
            ay1.set_xlim([-jeumax,jeumax])
            if (i%(nbseg//2) != 1) :
                ay1.set_yticks([])
            if (i <= nbseg//2) :
                ay1.set_xticks([])
            # ay1.legend()
        if not show :
            printfile=osp.join(repertoireOut,'%s_trajectoireHaut.png'%cas)
            plt.savefig(printfile)
            plt.close()



        fig = plt.figure()
        teyt = fig.suptitle("rotation manchette vs adaptateur")
        ay1 = plt.subplot(111)
        # rotdeg = butter_lowpass_filter(mesures_laser['L4']*10,30,fs)
        rotdeg = mesures_laser['L4']*10
        ay1.plot(tAL,rotdeg-O1,color=colorstandard,label=("manchette (degre)"))
        ay1.legend()
        ay1.set_xlabel("temps (s)")
        if not show :
            printfile=osp.join(repertoireOut,'%s_rotationManchette.png'%cas)
            plt.savefig(printfile)
            plt.close()
        # plt.show()


        fig = plt.figure()
        teyt = fig.suptitle("oscillation manchette")
        ay1 = plt.subplot(111)
        # rotdeg = butter_lowpass_filter(mesures_laser['L4']*10,30,fs)
        ay1.plot(tAL,np.arctan(tyM-tyA)/np.pi*180,color=colorstandard,label=("oscillation (degre)"))
        ay1.legend()
        ay1.set_xlabel("temps (s)")
        if not show :
            printfile=osp.join(repertoireOut,'%s_oscillationManchette.png'%cas)
            plt.savefig(printfile)
            plt.close()
        # plt.show()

        if show :
            plt.show()

        # fig = plt.figure(figsize=(4,4))
        # text = fig.suptitle("")
        # ax1 = plt.subplot(111)

        # vertices =[(-hpion*np.sqrt(3),0), (0, -hpion), (0,hpion)] 

        # #function to rotate and translate the standard shape to a new position
        # p1 = patches.Wedge((0,0),hpion,-15, 15, color="red")
        # p2 = patches.Wedge((0,0),hpion,120-15,120+15, color="red")
        # p3 = patches.Wedge((0,0),hpion,-120-15,-120+15, color="red")
        # p = collections.PatchCollection([p1,p2,p3],color='red')

        # ax1.add_collection(p)

        # ax1.set_xlim([-4,4])
        # ax1.set_ylim([-4,4])
        # draw_circle=plt.Circle((0., 0.), jeuPB/2,fill=False)
        # ax1.add_artist(draw_circle)

        # # line=[text,line1,line2,line3,line4]
        # line=[text,p]
        # def init():
        #     text.set_text("T = %s (s)"%0)
        #     return [text,p]
        # def animate(i):
        #     text.set_text("T = %.3f (s)"%(i/fs))
        #     bx=posBx[i+imin]
        #     by=posBy[i+imin]
        #     r = transforms.Affine2D().rotate((rotdeg[i+imin]-O1+180)/180*np.pi) +  transforms.Affine2D().translate(bx,by) + ax1.transData
        #     p.set_transform(r)
        #     return [text,p]


        # anim = FuncAnimation(fig, animate, init_func=init,
        #                                frames=len(tAL)-imin, interval=10, blit=False)

        # # longtest = 1000
        # # anim = FuncAnimation(fig, animate, init_func=init,
        # #                                frames=longtest, interval=(tAL[-1]-tAL[0])/len(tAL)*10000, blit=False)
        # printfile=osp.join(repertoireOut,'%s_traceTrajectoiresVideo.mp4'%cas)
        # anim.save(printfile, writer='ffmpeg')

