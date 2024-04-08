#!/bin/python3 
  # goal : Creation d'un tableau de reprise
#%%
import numpy as np
import pandas as pd
import os
import subprocess
import shutil
import glob
import sys
# import json
import rotation
from csv_to_pickle import csv2pickle
#%%
linert = True
limpact = True
lclean = True
lstdout = True
#%% lraidtimo :
#%% script / dir :
filename = 'pendule_timo.dgibi'
rawname = filename.split('.')[0]
original_directory = os.getcwd()
source = f'../{filename}'
repcast = f'../../build/'
if linert:
    repcast = f'../../cinert/'
#%%
#%% nombre de slices :
nslice = 3
# slicevtk = 50
#%% parametres du calcul 
xi = 0.1
t = 2.
dte = 2.e-6
nsort = 10
# 3 modes de flexion :
nmode_ela = 6
n_tronq = 0
# Fext = 137. * np.sqrt(2.)
    # pour b_lam = 6.5 :
# Fext = 0.72*(2.*79.44)
    # pour b_lam = 7.5 :
# Fext = 0.83*(2.*79.44)

# rappel : on a aussi remis thlim a 10^-5 dans devfb10.
# amo_ccone = 3.4
# on donne les 1ers angles en degres !

theta_ini_x = 45.

#%% repertoire de sauvegarde : 
dtstr = int(1.e6*dte)
# dtstr = int(-np.log10(dte/(1.e6*dte)))
xistr = int(100.*xi)
thinistr = int(h_lam*1.e3)
blstr = int(b_lam*1.e3)
lspringstr = int(lspring*1.e2)
nameglob = f'calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}_dt_{dtstr}_xi_{xistr}_mu_{mu}_hl_{hlstr}_bl_{blstr}_lspr_{lspringstr}'
if (lamode=="vrai"):
  amodemstr = int(amode_m*100.)
  amodeadstr = int(amode_ad*100.)
  nameglob = f'{nameglob}_amodem_{amodemstr}_amodead_{amodeadstr}'
# si couplage inertiel, on le met :
if (stoia=="vrai"):
  nameglob = f'{nameglob}_stoia'
if (manchette=="vrai"):
  nameglob = f'{nameglob}_manchette'
if (limpact=="vrai"):
  nameglob = f'{nameglob}_impact'
if (linert=="vrai"):
  nameglob = f'{nameglob}_inert'

repglob = f'../{nameglob}/'
#%%########################### CALCUL 0
# calcul initial : 
#   - avec fort amortissement
#   - sans pot vibrant (poids seuls)
##############################
# important : numeroter a 0 car le chargement n'a pas commence (cf manchadela_pions.dgibi) 
slice = 0
#
dictini = {
  #          reprise : 
             'reprise' : "vrai",
  #          lsin : 
             'lsin' : lsin,
  #          lsinb : 
             'lsinb' : lsinb,
  #          raideur de choc xp : 
             'lkxp' : lkxp,
  #          temps de choc xp : 
             'tcxp' : tcxp,
  #          bloquee : 
             'lbloq' : lbloq,
  #          lraidtimo : 
             'lraidtimo' : lraidtimo,
  #          ttot : 
             'ttot' : ttot,
  #          lexp : decroissance exponentielle du chargement
             'lexp' : "faux",
  #          blqry : bloq rotq ry adaptateur
             'blqry' : blqry,
  #          lmad2 : on retire le 2eme mode de l'adapter ?
             'lmad2' : lmad2,
  #          lraidiss : on met les raidisseurs ? 
             'raidiss' : raidiss,
  #          ressorts a lames : 
  #          hauteur : 
             'b_lam' : b_lam,
             'h_lam' : h_lam,
             'lspring' : lspring,
             'lsprbl' : lsprbl,
  #          trep : 
             'trep' : 0.,
  #          slice : 
             'slice' : slice,
             'slicevtk' : slicevtk,
  #          amo_ccone : 
             'amo_ccone' : 3.4*200.,
  #          xi : 
             'xi' : xi,
  #          lamode : 
             'lamode' : "vrai",
  #          amode : 
             'amode_ad' : 50.,
             'amode_m' :  50.,
  #          f1 : 
             'f1' : f1,
  #          f2 : 
             'f2' : f2,
  #          t : 
             't' : 0.1,
  #          dte : 
             'dte' : dte,
  #          nsort : 
             'nsort' : nsort,
  #          nmode : 
             'nmode' : nmode,
  #          n_tronq : 
             'n_tronq' : n_tronq,
  #          nmode_ad : 
             'nmode_ad' : nmode_ad,
  #          amplitude F sinus : 
             'spinini' : spinini,
  #          amplitude F sinus : 
             'Fext' : 0.,
  #          vlimoden 
            #  'vlimoden' : vlimoden,
             'vlimoden' : 1.e-5,
  #          ddls rig : manchette vect de rotation 
             'theta_ry' : theta_ry,
             'theta_rz' : theta_rz,
  #          ddls rig : vitesse de rotation 
             'wxini' : wxini,
             'wyini' : wyini,
             'wzini' : wzini,
  #          ddls rig : translations 
             'uxini' : uxini,
             'uyini' : uyini,
             'uzini' : uzini,
             'vxini' : vxini,
             'vyini' : vyini,
             'vzini' : vzini,
            }

dfini = pd.DataFrame(dictini, index=[0])
# coordonnees modales :
for i in np.arange(nmode - n_tronq):
    nameu = f"q{i+1}"
    namev = f"q{i+1}v"
    new_cols = [nameu, namev]
    new_vals = [ 0. , 0. ]
    for col,val in zip(new_cols,new_vals):
      dfini[col] = val

for i in np.arange(nmad):
    nameuad = f"q{i+1}ad"
    namevad = f"q{i+1}vad"
    new_cols = [nameuad, namevad]
    new_vals = [ 0. , 0. ]
    for col,val in zip(new_cols,new_vals):
      dfini[col] = val


# creation du repo :
repslice1 = f'{repglob}calc_{slice}/data/'

if not os.path.exists(repslice1):
    os.makedirs(repslice1)
    print(f"FOLDER : {repslice1} created.")
else:
    print(f"FOLDER : {repslice1} already exists.")

repfig = f'{repglob}calc_{slice}/fig/'
if not os.path.exists(repfig):
    os.makedirs(repfig)
    print(f"FOLDER : {repfig} created.")
else:
    print(f"FOLDER : {repfig} already exists.")

repvtk = f'{repglob}calc_{slice}/VTK/'
if not os.path.exists(repvtk):
    os.makedirs(repvtk)
    print(f"FOLDER : {repvtk} created.")
else:
    print(f"FOLDER : {repvtk} already exists.")

# on copie le script :
destination = f'{repglob}calc_{slice}/'
shutil.copy(source,f"{destination}{filename}")
# on copie l'executable :
shutil.copy(f'{repcast}cast_64_21',f"{destination}cast_64_21")

# Read the content of the existing file
with open(f"{destination}{filename}", 'r') as file:
    lines = file.readlines()

# Update the values based on the DataFrame
# for index, row in dfini.iterrows():
for colname, colvalue in dfini.iteritems():
    # Assuming the column names match the tags in the file (e.g., val1, val2)
    tag = f'*# {colname} :'
    # print(tag)
    replacement = f'{colname} = {colvalue.values[0]} ;'
    # print(replacement)

    # Find and replace the line with the updated value
    for i in range(len(lines)):
        if lines[i].strip() == tag:
            lines[i + 1] = f'{replacement}\n'
            break

# Write the modified content back to the file
with open(f"{destination}{filename}", 'w') as file:
    file.writelines(lines)

# 1ER CALCUL
os.chdir(destination)
#
print(f"calcul {slice} / {nslice}")
# castem21 $script > /dev/null 2>error.log  
if lstdout:
  command_to_run = [f'castem21 {filename}']
else:
  command_to_run = [f'castem21 {filename} > /dev/null 2>error.log']
result = subprocess.run(command_to_run, shell=True, check=True)
# Check the return code
if result.returncode == 0:
    print("Subprocess completed successfully.")
else:
    print(f"Subprocess failed with return code {result.returncode}")

# print("Error Output:")
# print(result.stderr.decode())
# on rentre :
os.chdir(original_directory)

# sys.exit()

# sauvegarde du pickle :
print("sauvegarde du 1er calcul...")
kwpi = {'rep_load' : f"{repglob}calc_{slice}/data/", 
        'rep_save' : f"{repglob}pickle/",
        'name_save' : f"{rawname}_{slice}"}
csv2pickle(**kwpi)

# sys.exit()

#%%########################## CALCUL 1
# CALCUL 1 : DEBUT DU CHARGEMENT POT :
############################# 
  # slice num ? 
slice = 1
print(f"slice : {slice}") 

scriptload = f"{rawname}_{slice - 1}.pickle"
repload = f"{repglob}pickle/"

repslice1 = f'{repglob}calc_{slice}/data/'
if not os.path.exists(repslice1):
    os.makedirs(repslice1)
    print(f"FOLDER : {repslice1} created.")
else:
    print(f"FOLDER : {repslice1} already exists.")

repfig = f'{repglob}calc_{slice}/fig/'
if not os.path.exists(repfig):
    os.makedirs(repfig)
    print(f"FOLDER : {repfig} created.")
else:
    print(f"FOLDER : {repfig} already exists.")

repvtk = f'{repglob}calc_{slice}/VTK/'
if not os.path.exists(repvtk):
    os.makedirs(repvtk)
    print(f"FOLDER : {repvtk} created.")
else:
    print(f"FOLDER : {repvtk} already exists.")
# on prend le script du premier calcul comme ca on a les memes parametre a coup sur :
source = f'{repglob}calc_{slice-1}/{filename}'
# on copie le script :
destination = f'{repglob}calc_{slice}/'
shutil.copy(source,f"{destination}{filename}")
# on copie l'executable :
shutil.copy(f'{repcast}cast_64_21',f"{destination}cast_64_21")

print(f"    files copied") 
#  lecture du dataframe :
df = pd.read_pickle(f"{repload}{scriptload}")
df.sort_values(by='t',inplace=True)
#  trnasfo quaternion to vecteur de rotation : 
q = np.array([df['quat1'].iloc[-1],df['quat2'].iloc[-1],df['quat3'].iloc[-1],df['quat4'].iloc[-1]])
vect = rotation.quat2vect2(q)
if (lraidtimo=="vrai"):
  # transfo quaternion to vecteur de rotation : adapter 
  qad = np.array([df['quat1_ad'].iloc[-1],df['quat2_ad'].iloc[-1],df['quat3_ad'].iloc[-1],df['quat4_ad'].iloc[-1]])
  vect_ad = rotation.quat2vect2(qad)
  print(f"rotation vect adapter : {vect_ad}")

# vect : en radians !! le .dgibi est adapte pour (reprise et (slice >EG 2))

#  
  # manchette & adaptateur : 4 modes elastiques. 
dict_rep = {
  #          t : 
             't' : t,
  #          reprise : 
             'reprise' : "vrai",
  #          lsin : 
             'lsin' : lsin,
  #          lsinb : 
             'lsinb' : lsinb,
  #          raideur de choc xp : 
             'lkxp' : lkxp,
  #          temps de choc xp : 
             'tcxp' : tcxp,
  #          bloquee : 
             'lbloq' : lbloq,
  #          lraidtimo : 
             'lraidtimo' : lraidtimo,
  #          ttot : 
             'ttot' : ttot,
  #          amo_ccone : 
             'amo_ccone' : 3.4,
  #          lexp : 
             'lexp' : lexp,
  #          blqry : bloq rotq ry adaptateur
             'blqry' : blqry,
  #          ressorts a lames : 
             'b_lam' : b_lam,
             'h_lam' : h_lam,
             'lspring' : lspring,
             'lsprbl' : lsprbl,
  #          fefin : valeur de la force a la fin du calcul si lexp :
             'fefin' : fefin,
  #          lmad2 : on retire le 2eme mode de l'adapter 
             'lmad2' : lmad2,
  #          lraidiss : on met les raidisseurs ? 
             'raidiss' : raidiss,
  #          lamode : 
             'lamode' : lamode,
  #          amode : 
             'amode_ad' : amode_ad,
             'amode_m' : amode_m,
  #          slice : 
             'slice' : df['slice'].drop_duplicates().values[0]+1,
             'slicevtk' : slicevtk,
  #          amo_ccone : 
             'amo_ccone' : amo_ccone,
  #          xi : 
             'xi' : xi,
  #          nmode : 
             'nmode' : df['nmode'].drop_duplicates().values[0],
  #          nmode_ad : 
             'nmode_ad' : df['nmode_ad'].drop_duplicates().values[0],
  #          instant reprise : ici a 0 , le chargement commence.
             'trep' : 0.,
  #          amplitude F sinus ou sinus balaye : 
             'Fext' : Fext,  
  #          vlimoden 
             'vlimoden' : df['vlimoden'].drop_duplicates().values[0],
  #          ddls rig : manchette vect de rotation 
             'theta_rx' : vect[0],
             'theta_ry' : vect[1],
             'theta_rz' : vect[2],
  #          ddls rig : vitesse de rotation 
             'wxini' : df['WX'].iloc[-1],
             'wyini' : df['WY'].iloc[-1],
             'wzini' : df['WZ'].iloc[-1],
             'arxini' : df['AX'].iloc[-1],
             'aryini' : df['AY'].iloc[-1],
             'arzini' : df['AZ'].iloc[-1],
  #          ddls rig : translations 
             'uxini' : df['qtx'].iloc[-1],
             'uyini' : df['qty'].iloc[-1],
             'uzini' : df['qtz'].iloc[-1],
             'vxini' : df['qvtx'].iloc[-1],
             'vyini' : df['qvty'].iloc[-1],
             'vzini' : df['qvtz'].iloc[-1],
             'axini' : df['qatx'].iloc[-1],
             'ayini' : df['qaty'].iloc[-1],
             'azini' : df['qatz'].iloc[-1],
             'nsauvini' : df['nsauvini'].iloc[-1],
            }
dfini = pd.DataFrame(dict_rep, index=[0])

print(f"nsauvini : {df['nsauvini'].iloc[-1]}")
    
# for i in np.arange(nmode - n_tronq):
for i in np.arange(dfini['nsauvini'][0]):
    nameu = f"q{i+1}"
    namev = f"q{i+1}v"
    namea = f"q{i+1}a"
    new_cols = [nameu, namev, namea]
    new_vals = [ df[nameu].iloc[-1] , df[namev].iloc[-1], df[namea].iloc[-1]]
    for col,val in zip(new_cols,new_vals):
      dfini[col] = val

if (not (lraidtimo=="vrai")):
    # ddls elastiques :
    for i in np.arange(nmad):
        nameuad = f"q{i+1}ad"
        namevad = f"q{i+1}vad"
        nameaad = f"q{i+1}aad"
        new_cols = [nameuad, namevad, nameaad]
        new_vals = [ df[nameuad].iloc[-1], df[namevad].iloc[-1], df[nameaad].iloc[-1] ]
        for col,val in zip(new_cols,new_vals):
          dfini[col] = val

if (lraidtimo=="vrai"):
    # ddls rigides =
    dfini['theta_rx_ad'] = vect_ad[0]
    dfini['theta_ry_ad'] = vect_ad[1]
    dfini['theta_rz_ad'] = vect_ad[2]
    # ddls rig = vitesse de rotation 
    dfini['wxini_ad']  = df['WX_AD'].iloc[-1]
    dfini['wyini_ad']  = df['WY_AD'].iloc[-1]
    dfini['wzini_ad']  = df['WZ_AD'].iloc[-1]
    dfini['arxini_ad'] = df['AX_AD'].iloc[-1]
    dfini['aryini_ad'] = df['AY_AD'].iloc[-1]
    dfini['arzini_ad'] = df['AZ_AD'].iloc[-1]
    # ddls rig = translations 
    dfini['uxini_ad']  = df['qtx_ad'].iloc[-1]
    dfini['uyini_ad']  = df['qty_ad'].iloc[-1]
    dfini['uzini_ad']  = df['qtz_ad'].iloc[-1]
    dfini['vxini_ad']  = df['qvtx_ad'].iloc[-1]
    dfini['vyini_ad']  = df['qvty_ad'].iloc[-1]
    dfini['vzini_ad']  = df['qvtz_ad'].iloc[-1]
    dfini['axini_ad']  = df['qatx_ad'].iloc[-1]
    dfini['ayini_ad']  = df['qaty_ad'].iloc[-1]
    dfini['azini_ad']  = df['qatz_ad'].iloc[-1]

# Read the content of the existing file
with open(f"{destination}{filename}", 'r') as file:
    lines = file.readlines()

# Update the values based on the DataFrame
# for index, row in dfini.iterrows():
for colname, colvalue in dfini.iteritems():
    # Assuming the column names match the tags in the file (e.g., val1, val2)
    tag = f'*# {colname} :'
    # print(tag)
    replacement = f'{colname} = {colvalue.values[0]} ;'
    # print(replacement)

    # Find and replace the line with the updated value
    for i in range(len(lines)):
        if lines[i].strip() == tag:
            lines[i + 1] = f'{replacement}\n'
            break

# Write the modified content back to the file
with open(f"{destination}{filename}", 'w') as file:
    file.writelines(lines)

print(f"    script initialized") 

# 2nd calcul :
print(f"    calc {slice} / {nslice}") 
os.chdir(destination)
#
# command_to_run = [f'castem21 {filename}']
result = subprocess.run(command_to_run, shell=True, check=True)
# Check the return code
if result.returncode == 0:
    print("Subprocess completed successfully.")
else:
    print(f"Subprocess failed with return code {result.returncode}")

# on rentre :
os.chdir(original_directory)

# sauvegarde du pickle :
print(f"saving {slice}th calc...")
kwpi = {'rep_load' : f"{repglob}calc_{slice}/data/", 
        'rep_save' : f"{repglob}pickle/",
        'name_save' : f"{rawname}_{slice}"}
csv2pickle(**kwpi)

#%%########################## LOOP
# LOOP  :
############################# 
  # slice num ? 
for slice in range(2,nslice+1):
  print(f"slice : {slice}") 

  scriptload = f"{rawname}_{slice - 1}.pickle"
  repload = f"{repglob}pickle/"

  repslice1 = f'{repglob}calc_{slice}/data/'
  if not os.path.exists(repslice1):
      os.makedirs(repslice1)
      print(f"FOLDER : {repslice1} created.")
  else:
      print(f"FOLDER : {repslice1} already exists.")

  repfig = f'{repglob}calc_{slice}/fig/'
  if not os.path.exists(repfig):
      os.makedirs(repfig)
      print(f"FOLDER : {repfig} created.")
  else:
      print(f"FOLDER : {repfig} already exists.")

  repvtk = f'{repglob}calc_{slice}/VTK/'
  if not os.path.exists(repvtk):
      os.makedirs(repvtk)
      print(f"FOLDER : {repvtk} created.")
  else:
      print(f"FOLDER : {repvtk} already exists.")
  # on prend le script du dernier calcul comme ca on a les memes parametre a coup sur :
  source = f'{repglob}calc_{slice-1}/{filename}'
  # on copie le script :
  destination = f'{repglob}calc_{slice}/'
  shutil.copy(source,f"{destination}{filename}")
  # on copie l'executable :
  shutil.copy(f'{repcast}cast_64_21',f"{destination}cast_64_21")

  print(f"    files copied") 
  #  lecture du dataframe :
  df = pd.read_pickle(f"{repload}{scriptload}")
  df.sort_values(by='t',inplace=True)
  #  transfo quaternion to vecteur de rotation : sleeve
  q = np.array([df['quat1'].iloc[-1],df['quat2'].iloc[-1],df['quat3'].iloc[-1],df['quat4'].iloc[-1]])
  vect = rotation.quat2vect2(q)
  if (lraidtimo=="vrai"):
    #  transfo quaternion to vecteur de rotation : adapter 
    qad = np.array([df['quat1_ad'].iloc[-1],df['quat2_ad'].iloc[-1],df['quat3_ad'].iloc[-1],df['quat4_ad'].iloc[-1]])
    vect_ad = rotation.quat2vect2(qad)
    print(f"rotation vect adapter : {vect_ad}")
  # Rq : vect : en radians !! le .dgibi est adapte pour (reprise et (slice >EG 2))

  print(f"trep = {df['t'].iloc[-1]:.6f} ")
  #  
    # manchette & adaptateur : 4 modes elastiques. 
  dict_rep = {
    #          reprise : 
               'reprise' : "vrai",
    #          lsin : 
               'lsin' : lsin,
    #          lsinb : 
               'lsinb' : lsinb,
    #          raideur de choc xp : 
               'lkxp' : lkxp,
    #          temps de choc xp : 
               'tcxp' : tcxp,
    #          bloquee : 
               'lbloq' : lbloq,
    #          lraidtimo : 
               'lraidtimo' : lraidtimo,
    #          ttot : 
               'ttot' : ttot,
    #          amo_ccone : 
               'amo_ccone' : 3.4,
    #          xi : 
               'xi' : xi,
    #          lexp : 
               'lexp' : lexp,
    #          ressorts a lames : 
               'b_lam' : b_lam,
               'h_lam' : h_lam,
               'lspring' : lspring,
               'lsprbl' : lsprbl,
    #          fefin : 
               'fefin' : fefin,
    #          lamode : 
               'lamode' : lamode,
    #          amode : 
               'amode_ad' : amode_ad,
               'amode_m' : amode_m,
    #          lmad2 : on retire le 2eme mode de l'adapter 
               'lmad2' : lmad2,
    #          lraidiss : on met les raidisseurs ? 
               'raidiss' : raidiss,
    #          slice : 
               'slice' : df['slice'].drop_duplicates().values[0]+1,
               'slicevtk' : slicevtk,
    #          nmode : 
               'nmode' : df['nmode'].drop_duplicates().values[0],
    #          nmode_ad : 
               'nmode_ad' : df['nmode_ad'].drop_duplicates().values[0],
    #          instant reprise : 
               'trep' : df['t'].iloc[-1],
    #          amplitude F sinus : 
               'Fext' : df['Fext0'].drop_duplicates().values[0],
    #          vlimoden 
               'vlimoden' : df['vlimoden'].drop_duplicates().values[0],
    #          ddls rig : manchette vect de rotation 
               'theta_rx' : vect[0],
               'theta_ry' : vect[1],
               'theta_rz' : vect[2],
    #          ddls rig : vitesse de rotation 
               'wxini' : df['WX'].iloc[-1],
               'wyini' : df['WY'].iloc[-1],
               'wzini' : df['WZ'].iloc[-1],
               'arxini' : df['AX'].iloc[-1],
               'aryini' : df['AY'].iloc[-1],
               'arzini' : df['AZ'].iloc[-1],
    #          ddls rig : translations 
               'uxini' : df['qtx'].iloc[-1],
               'uyini' : df['qty'].iloc[-1],
               'uzini' : df['qtz'].iloc[-1],
               'vxini' : df['qvtx'].iloc[-1],
               'vyini' : df['qvty'].iloc[-1],
               'vzini' : df['qvtz'].iloc[-1],
               'axini' : df['qatx'].iloc[-1],
               'ayini' : df['qaty'].iloc[-1],
               'azini' : df['qatz'].iloc[-1],
               'nsauvini' : df['nsauvini'].iloc[-1],
              }
  dfini = pd.DataFrame(dict_rep, index=[0])

#   for i in np.arange(nmode - n_tronq):
  for i in np.arange(dfini['nsauvini'][0]):
      nameu = f"q{i+1}"
      namev = f"q{i+1}v"
      namea = f"q{i+1}a"
      new_cols = [nameu, namev, namea]
      new_vals = [ df[nameu].iloc[-1] , df[namev].iloc[-1], df[namea].iloc[-1]]
      for col,val in zip(new_cols,new_vals):
        dfini[col] = val

  if (not (lraidtimo=="vrai")):
    for i in np.arange(nmad):
        nameuad = f"q{i+1}ad"
        namevad = f"q{i+1}vad"
        nameaad = f"q{i+1}aad"
        new_cols = [nameuad, namevad, nameaad]
        new_vals = [ df[nameuad].iloc[-1], df[namevad].iloc[-1], df[nameaad].iloc[-1] ]
        for col,val in zip(new_cols,new_vals):
          dfini[col] = val

  if (lraidtimo=="vrai"):
      # ddls rigides =
      dfini['theta_rx_ad'] = vect_ad[0]
      dfini['theta_ry_ad'] = vect_ad[1]
      dfini['theta_rz_ad'] = vect_ad[2]
      # ddls rig = vitesse de rotation 
      dfini['wxini_ad']  = df['WX_AD'].iloc[-1]
      dfini['wyini_ad']  = df['WY_AD'].iloc[-1]
      dfini['wzini_ad']  = df['WZ_AD'].iloc[-1]
      dfini['arxini_ad'] = df['AX_AD'].iloc[-1]
      dfini['aryini_ad'] = df['AY_AD'].iloc[-1]
      dfini['arzini_ad'] = df['AZ_AD'].iloc[-1]
      # ddls rig = translations 
      dfini['uxini_ad']  = df['qtx_ad'].iloc[-1]
      dfini['uyini_ad']  = df['qty_ad'].iloc[-1]
      dfini['uzini_ad']  = df['qtz_ad'].iloc[-1]
      dfini['vxini_ad']  = df['qvtx_ad'].iloc[-1]
      dfini['vyini_ad']  = df['qvty_ad'].iloc[-1]
      dfini['vzini_ad']  = df['qvtz_ad'].iloc[-1]
      dfini['axini_ad']  = df['qatx_ad'].iloc[-1]
      dfini['ayini_ad']  = df['qaty_ad'].iloc[-1]
      dfini['azini_ad']  = df['qatz_ad'].iloc[-1]

  # Read the content of the existing file
  with open(f"{destination}{filename}", 'r') as file:
      lines = file.readlines()

  # Update the values based on the DataFrame
  # for index, row in dfini.iterrows():
  for colname, colvalue in dfini.iteritems():
      # Assuming the column names match the tags in the file (e.g., val1, val2)
      tag = f'*# {colname} :'
      # print(tag)
      replacement = f'{colname} = {colvalue.values[0]} ;'
      # print(replacement)

      # Find and replace the line with the updated value
      for i in range(len(lines)):
          if lines[i].strip() == tag:
              lines[i + 1] = f'{replacement}\n'
              break

  # Write the modified content back to the file
  with open(f"{destination}{filename}", 'w') as file:
      file.writelines(lines)

  print(f"    script initialized") 

  # 2nd calcul :
  print(f"    calc {slice} / {nslice}") 
  os.chdir(destination)
  #
  # command_to_run = [f'castem21 {filename}']
  result = subprocess.run(command_to_run, shell=True, check=True)
  # Check the return code
  if result.returncode == 0:
      print("Subprocess completed successfully.")
  else:
      print(f"Subprocess failed with return code {result.returncode}")

  # on rentre :
  os.chdir(original_directory)

  # sauvegarde du pickle :
  print(f"saving {slice}th calc...")
  kwpi = {'rep_load' : f"{repglob}calc_{slice}/data/", 
          'rep_save' : f"{repglob}pickle/",
          'name_save' : f"{rawname}_{slice}"}
  csv2pickle(**kwpi)


#%%########################## SAVE SCRIPTS
# sauvegarde du script de batch :
#############################
# Get the absolute path of the running script
script_path = os.path.abspath(__file__)
# Get the file name
script_name = os.path.basename(script_path)
shutil.copy(script_path,repglob)
print(f"batch script {script_name} saved into {repglob}")
shutil.copy(f'{repglob}calc_1/{filename}',repglob)
print(f"1st .dgibi saved into {repglob}")

ps_files = glob.glob(f'{repglob}calc_1/*.ps')
for ps_file in ps_files:
    shutil.copy(ps_file,repglob)
print(f"calc_1/fig/*.ps saved into {repglob}") 

repvtk = f"{repglob}VTK/"
if not os.path.exists(repvtk):
    os.makedirs(repvtk)
    print(f"FOLDER : {repvtk} created.")
else:
    print(f"FOLDER : {repvtk} already exists.")

vtu_files = glob.glob(f'{repglob}calc_{nslice}/VTK/*.vtu')
pvd_files = glob.glob(f'{repglob}calc_{nslice}/VTK/*.pvd')
for vtu, pvd in zip(vtu_files,pvd_files):
    shutil.copy(vtu,repvtk)
    shutil.copy(pvd,repvtk)
print(f"vtk last slice saved")

#%%########################## MV SLICE 0
# deplacement du calcul 0 :
#############################
rep0 = f'{repglob}pickle/slice_0/'
if not os.path.exists(rep0):
    os.makedirs(rep0)
    print(f"FOLDER : {rep0} created.")
else:
    print(f"FOLDER : {rep0} already exists.")
calc0 = f"{repglob}pickle/{rawname}_0.pickle"

shutil.move(calc0,f"{rep0}result.pickle")
print(f"slice 0 moved to {rep0}")

#%%######################################### CLEAN
# menage :
############################################
if lclean:
  print(f"cleaning...")
  pattern = "calc"
  import os
  import shutil

  for current_directory, subdirectories, files in os.walk(repglob, topdown=False):
      # Check if any subdirectory contains the specified pattern
      matching_subdirectories = [subdir for subdir in subdirectories if pattern in subdir]

      for matching_subdirectory in matching_subdirectories:
          # Delete the subdirectory
          subdirectory_path = os.path.join(current_directory, matching_subdirectory)
          try:
              shutil.rmtree(subdirectory_path)
              print(f"Deleted subdirectory: {subdirectory_path}")
          except Exception as e:
              print(f"Error deleting {subdirectory_path}: {e}")

  print(f"END")

sys.exit()
#%%########################## 
# calcul indiv  :
############################# 
slice = 128
lstdout = True
print(f"slice : {slice}") 

scriptload = f"{rawname}_{slice - 1}.pickle"
repload = f"{repglob}pickle/"

repslice1 = f'{repglob}calc_{slice}/data/'
if not os.path.exists(repslice1):
    os.makedirs(repslice1)
    print(f"FOLDER : {repslice1} created.")
else:
    print(f"FOLDER : {repslice1} already exists.")

repfig = f'{repglob}calc_{slice}/fig/'
if not os.path.exists(repfig):
    os.makedirs(repfig)
    print(f"FOLDER : {repfig} created.")
else:
    print(f"FOLDER : {repfig} already exists.")

# on prend le script du dernier calcul comme ca on a les memes parametre a coup sur :
source = f'{repglob}calc_{slice-1}/{filename}'
# on copie le script :
destination = f'{repglob}calc_{slice}/'
shutil.copy(source,f"{destination}{filename}")
# on copie l'executable :
shutil.copy(f'{repcast}cast_64_21',f"{destination}cast_64_21")

print(f"    files copied") 
#  lecture du dataframe :
df = pd.read_pickle(f"{repload}{scriptload}")
df.sort_values(by='t',inplace=True)
#  trnasfo quaternion to vecteur de rotation : 
q = np.array([df['quat1'].iloc[-1],df['quat2'].iloc[-1],df['quat3'].iloc[-1],df['quat4'].iloc[-1]])
vect = rotation.quat2vect2(q)
if (lraidtimo=="vrai"):
  #  transfo quaternion to vecteur de rotation : adapter 
  qad = np.array([df['quat1_ad'].iloc[-1],df['quat2_ad'].iloc[-1],df['quat3_ad'].iloc[-1],df['quat4_ad'].iloc[-1]])
  vect_ad = rotation.quat2vect2(qad)
# vect : en radians !! le .dgibi est adapte pour (reprise et (slice >EG 2))

print(f"trep = {df['t'].iloc[-1]:.6f} ")
#  
  # manchette & adaptateur : 4 modes elastiques. 
dict_rep = {
  #          reprise : 
             'reprise' : "vrai",
  #          lsin : 
             'lsin' : lsin,
  #          lsinb : 
             'lsinb' : lsinb,
  #          raideur de choc xp : 
             'lkxp' : lkxp,
  #          temps de choc xp : 
             'tcxp' : tcxp,
  #          bloquee : 
             'lbloq' : lbloq,
  #          ttot : 
             'ttot' : ttot,
  #          lexp : 
             'lexp' : lexp,
  #          ressorts a lames : 
             'b_lam' : b_lam,
             'h_lam' : h_lam,
             'lspring' : lspring,
  #          fefin : 
             'fefin' : fefin,
  #          lamode : 
             'lamode' : lamode,
  #          amode : 
             'amode_ad' : amode_ad,
             'amode_m' : amode_m,
  #          lmad2 : on retire le 2eme mode de l'adapter 
             'lmad2' : lmad2,
  #          lraidiss : on met les raidisseurs ? 
             'raidiss' : raidiss,
  #          slice : 
             'slice' : df['slice'].drop_duplicates().values[0]+1,
  #          nmode : 
             'nmode' : df['nmode'].drop_duplicates().values[0],
  #          nmode_ad : 
             'nmode_ad' : df['nmode_ad'].drop_duplicates().values[0],
  #          instant reprise : 
             'trep' : df['t'].iloc[-1],
  #          amplitude F sinus : 
             'Fext' : df['Fext0'].drop_duplicates().values[0],
  #          vlimoden 
             'vlimoden' : df['vlimoden'].drop_duplicates().values[0],
  #          ddls rig : manchette vect de rotation 
             'theta_rx' : vect[0],
             'theta_ry' : vect[1],
             'theta_rz' : vect[2],
  #          ddls rig : vitesse de rotation 
             'wxini' : df['WX'].iloc[-1],
             'wyini' : df['WY'].iloc[-1],
             'wzini' : df['WZ'].iloc[-1],
             'arxini' : df['AX'].iloc[-1],
             'aryini' : df['AY'].iloc[-1],
             'arzini' : df['AZ'].iloc[-1],
  #          ddls rig : translations 
             'uxini' : df['qtx'].iloc[-1],
             'uyini' : df['qty'].iloc[-1],
             'uzini' : df['qtz'].iloc[-1],
             'vxini' : df['qvtx'].iloc[-1],
             'vyini' : df['qvty'].iloc[-1],
             'vzini' : df['qvtz'].iloc[-1],
             'axini' : df['qatx'].iloc[-1],
             'ayini' : df['qaty'].iloc[-1],
             'azini' : df['qatz'].iloc[-1],
             'nsauvini' : df['nsauvini'].iloc[-1],
            }
dfini = pd.DataFrame(dict_rep, index=[0])

#  for i in np.arange(nmode - n_tronq):
for i in np.arange(dfini['nsauvini'][0]):
    nameu = f"q{i+1}"
    namev = f"q{i+1}v"
    namea = f"q{i+1}a"
    new_cols = [nameu, namev, namea]
    new_vals = [ df[nameu].iloc[-1] , df[namev].iloc[-1], df[namea].iloc[-1]]
    for col,val in zip(new_cols,new_vals):
      dfini[col] = val

if (not (lraidtimo=="vrai")):
  for i in np.arange(nmad):
      nameuad = f"q{i+1}ad"
      namevad = f"q{i+1}vad"
      nameaad = f"q{i+1}aad"
      new_cols = [nameuad, namevad, nameaad]
      new_vals = [ df[nameuad].iloc[-1], df[namevad].iloc[-1], df[nameaad].iloc[-1] ]
      for col,val in zip(new_cols,new_vals):
        dfini[col] = val

if (lraidtimo=="vrai"):
    # ddls rigides =
    dfini['theta_rx_ad'] = vect_ad[0]
    dfini['theta_ry_ad'] = vect_ad[1]
    dfini['theta_rz_ad'] = vect_ad[2]
    # ddls rig = vitesse de rotation 
    dfini['wxini_ad']  = df['WX_AD'].iloc[-1]
    dfini['wyini_ad']  = df['WY_ad'].iloc[-1]
    dfini['wzini_ad']  = df['WZ_ad'].iloc[-1]
    dfini['arxini_ad'] = df['AX_ad'].iloc[-1]
    dfini['aryini_ad'] = df['AY_ad'].iloc[-1]
    dfini['arzini_ad'] = df['AZ_ad'].iloc[-1]
    # ddls rig = translations 
    dfini['uxini_ad']  = df['qtx_ad'].iloc[-1]
    dfini['uyini_ad']  = df['qty_ad'].iloc[-1]
    dfini['uzini_ad']  = df['qtz_ad'].iloc[-1]
    dfini['vxini_ad']  = df['qvtx_ad'].iloc[-1]
    dfini['vyini_ad']  = df['qvty_ad'].iloc[-1]
    dfini['vzini_ad']  = df['qvtz_ad'].iloc[-1]
    dfini['axini_ad']  = df['qatx_ad'].iloc[-1]
    dfini['ayini_ad']  = df['qaty_ad'].iloc[-1]
    dfini['azini_ad']  = df['qatz_ad'].iloc[-1]

# Read the content of the existing file
with open(f"{destination}{filename}", 'r') as file:
    lines = file.readlines()

# Update the values based on the DataFrame
# for index, row in dfini.iterrows():
for colname, colvalue in dfini.iteritems():
    # Assuming the column names match the tags in the file (e.g., val1, val2)
    tag = f'*# {colname} :'
    # print(tag)
    replacement = f'{colname} = {colvalue.values[0]} ;'
    # print(replacement)

    # Find and replace the line with the updated value
    for i in range(len(lines)):
        if lines[i].strip() == tag:
            lines[i + 1] = f'{replacement}\n'
            break

# Write the modified content back to the file
with open(f"{destination}{filename}", 'w') as file:
    file.writelines(lines)

print(f"    script initialized") 

# 2nd calcul :
print(f"    calc {slice} / {nslice}") 
os.chdir(destination)
# command to run :
if lstdout:
  command_to_run = [f'castem21 {filename}']
else:
  command_to_run = [f'castem21 {filename} > /dev/null 2>error.log']
#
# command_to_run = [f'castem21 {filename}']
result = subprocess.run(command_to_run, shell=True, check=True)
# Check the return code
if result.returncode == 0:
    print("Subprocess completed successfully.")
else:
    print(f"Subprocess failed with return code {result.returncode}")

# on rentre :
os.chdir(original_directory)

#%% sauvegarde du pickle :
slice = 128 

print(f"saving {slice}th calc...")
kwpi = {'rep_load' : f"{repglob}calc_{slice}/data/", 
        'rep_save' : f"{repglob}pickle/",
        'name_save' : f"{rawname}_{slice}"}
csv2pickle(**kwpi)
# %%
