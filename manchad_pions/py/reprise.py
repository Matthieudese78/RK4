#!/bin/python3 
  # goal : Creation d'un tableau de reprise
#%%
import numpy as np
import pandas as pd
import os
import subprocess
import shutil
# import glob
# import sys
# import json
import rotation
from csv_to_pickle import csv2pickle
#%% script / dir :
filename = 'manchadela_pions.dgibi'
rawname = filename.split('.')[0]
original_directory = os.getcwd()
source = f'../{filename}'
repcast = f'../../build/'
lstdout = False
#%% nombre de slices :
nslice = 10
#%% parametres du calcul 
ttot = 120.
f1 = 2.
f2 = 20.
t = 1.
lexp  = "vrai" 
dte = 1.e-6
nsort = 2000
nmode = 10
n_tronq = 6
nmode_ad = 7
Fext = 250.
fefin = 10.
vlimoden = 1.e-4
amo_ccone = 3.4
# on donne les 1ers angles en degres !
theta_rx = 0.
theta_ry = 0.
spinini = 0.
wxini = 0.
wyini = 0.
wzini = 0.
uxini = 0.
uyini = 0.
uzini = 0.
vxini = 0.
vyini = 0.
vzini = 0.

#%% repertoire de sauvegarde : 
vlostr = int(-np.log10(vlimoden))
repglob = f'../calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}/'

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
  #          lexp : 
             'lexp' : "faux",
  #          trep : 
             'trep' : 0.,
  #          slice : 
             'slice' : slice,
  #          ttot : 
             'ttot' : ttot,
  #          amo_ccone : 
             'amo_ccone' : 1.e2,
  #          lamode : 
             'lamode' : "vrai",
  #          amode : 
             'amode' : 30.,
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
             'vlimoden' : vlimoden,
  #          ddls rig : manchette vect de rotation 
             'theta_rx' : theta_rx,
             'theta_ry' : theta_ry,
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

for i in np.arange(nmode_ad):
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
#  trnasfo quaternion to vecteur de rotation : 
q = np.array([df['quat1'].iloc[-1],df['quat2'].iloc[-1],df['quat3'].iloc[-1],df['quat4'].iloc[-1]])
vect = rotation.quat2vect(q)
# vect : en radians !! le .dgibi est adapte pour (reprise et (slice >EG 2))

#  
  # manchette & adaptateur : 4 modes elastiques. 
dict_rep = {
  #          t : 
             't' : t,
  #          reprise : 
             'reprise' : "vrai",
  #          lexp : 
             'lexp' : lexp,
  #          fefin : valeur de la force a la fin du calcul si lexp :
             'fefin' : fefin,
  #          lamode : 
             'lamode' : "faux",
  #          slice : 
             'slice' : df['slice'].drop_duplicates().values[0]+1,
  #          amo_ccone : 
             'amo_ccone' : amo_ccone,
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
  #          ddls rig : translations 
             'uxini' : df['qtx'].iloc[-1],
             'uyini' : df['qty'].iloc[-1],
             'uzini' : df['qtz'].iloc[-1],
             'vxini' : df['qvtx'].iloc[-1],
             'vyini' : df['qvty'].iloc[-1],
             'vzini' : df['qvtz'].iloc[-1],
            }
dfini = pd.DataFrame(dict_rep, index=[0])

for i in np.arange(nmode - n_tronq):
    nameu = f"q{i+1}"
    namev = f"q{i+1}v"
    new_cols = [nameu, namev]
    new_vals = [ df[nameu].iloc[-1] , df[namev].iloc[-1] ]
    for col,val in zip(new_cols,new_vals):
      dfini[col] = val
for i in np.arange(nmode_ad):
    nameuad = f"q{i+1}ad"
    namevad = f"q{i+1}vad"
    new_cols = [nameuad, namevad]
    new_vals = [ df[nameuad].iloc[-1], df[namevad].iloc[-1] ]
    for col,val in zip(new_cols,new_vals):
      dfini[col] = val

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
  #  trnasfo quaternion to vecteur de rotation : 
  q = np.array([df['quat1'].iloc[-1],df['quat2'].iloc[-1],df['quat3'].iloc[-1],df['quat4'].iloc[-1]])
  vect = rotation.quat2vect(q)
  # vect : en radians !! le .dgibi est adapte pour (reprise et (slice >EG 2))

  #  
    # manchette & adaptateur : 4 modes elastiques. 
  dict_rep = {
    #          reprise : 
               'reprise' : "vrai",
    #          lexp : 
               'lexp' : lexp,
    #          fefin : 
               'fefin' : fefin,
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
    #          ddls rig : translations 
               'uxini' : df['qtx'].iloc[-1],
               'uyini' : df['qty'].iloc[-1],
               'uzini' : df['qtz'].iloc[-1],
               'vxini' : df['qvtx'].iloc[-1],
               'vyini' : df['qvty'].iloc[-1],
               'vzini' : df['qvtz'].iloc[-1],
              }
  dfini = pd.DataFrame(dict_rep, index=[0])

  for i in np.arange(nmode - n_tronq):
      nameu = f"q{i+1}"
      namev = f"q{i+1}v"
      new_cols = [nameu, namev]
      new_vals = [ df[nameu].iloc[-1] , df[namev].iloc[-1] ]
      for col,val in zip(new_cols,new_vals):
        dfini[col] = val
  for i in np.arange(nmode_ad):
      nameuad = f"q{i+1}ad"
      namevad = f"q{i+1}vad"
      new_cols = [nameuad, namevad]
      new_vals = [ df[nameuad].iloc[-1], df[namevad].iloc[-1] ]
      for col,val in zip(new_cols,new_vals):
        dfini[col] = val

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


#%%########################## MV SLICE 0
# deplacement du calcul 0 :
#############################
rep0 = f'{repglob}/pickle/slice_0/'
if not os.path.exists(rep0):
    os.makedirs(rep0)
    print(f"FOLDER : {rep0} created.")
else:
    print(f"FOLDER : {rep0} already exists.")
calc0 = f"{repglob}pickle/{rawname}_0.pickle"

shutil.move(calc0,f"{rep0}result.pickle")
print(f"slice 0 moved to {rep0}")

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

#%%######################################### CLEAN
# menage :
############################################
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

