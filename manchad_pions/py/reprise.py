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
import json
import rotation
from csv_to_pickle import csv2pickle
#%% script / dir :
filename = 'manchadela_pions.dgibi'
original_directory = os.getcwd()
source = f'../{filename}'
lstdout = True
#%% nombre de slices :
nslice = 10
#%% calcul initial :
ttot = 120.
f1 = 2.
f2 = 20.
t = 0.5
dte = 1.e-6
nsort = 1000
nmode = 10
n_tronq = 6
nmode_ad = 7
Fext = 100.
vlimoden = 1.e-4
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

dictini = {
  #          reprise : 
             'reprise' : "vrai",
  #          trep : 
             'trep' : 0.,
  #          slice : 
             'slice' : 1,
  #          ttot : 
             'ttot' : 120.,
  #          f1 : 
             'f1' : 2.,
  #          f2 : 
             'f2' : 20.,
  #          t : 
             't' : t,
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
             'Fext' : Fext,
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

vlostr = int(-np.log10(vlimoden))
repglob = f'../calc_fext_{int(Fext)}_spin_{int(spinini)}_vlo_{vlostr}/'

#%%
# creation du repo :
repslice1 = f'{repglob}calc_{1}/data/'

if not os.path.exists(repslice1):
    os.makedirs(repslice1)
    print(f"FOLDER : {repslice1} created.")
else:
    print(f"FOLDER : {repslice1} already exists.")

repfig = f'{repglob}calc_{1}/fig/'
if not os.path.exists(repfig):
    os.makedirs(repfig)
    print(f"FOLDER : {repfig} created.")
else:
    print(f"FOLDER : {repfig} already exists.")
#%%
# on copie le script :
destination = f'{repglob}calc_{1}/'
shutil.copy(source,f"{destination}{filename}")
# on copie l'executable :
shutil.copy('../../../build/cast_64_21',f"{destination}cast_64_21")

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

#%% 1ER CALCUL
os.chdir(destination)
#
print(f"calcul 1 / {nslice}")
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

#%% sauvegarde du pickle :
print("sauvegarde du 1er calcul...")
kwpi = {'rep_load' : f"{repglob}calc_{1}/data/", 
        'rep_save' : f"{repglob}pickle/",
        'name_save' : f"manchadela_pions_{1}"}
csv2pickle(**kwpi)

# sys.exit()
#%% lecture du dataframe  :
  # slice num ? 
for slice in range(2,nslice+1):
  print(f"slice : {slice}") 

  scriptload = f"manchadela_pions_{slice - 1}.pickle"
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
  source = f'{repglob}calc_{1}/{filename}'
  # on copie le script :
  destination = f'{repglob}calc_{slice}/'
  shutil.copy(source,f"{destination}{filename}")
  # on copie l'executable :
  shutil.copy('../../../build/cast_64_21',f"{destination}cast_64_21")

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

  #%% sauvegarde du pickle :
  print(f"saving {slice}th calc...")
  kwpi = {'rep_load' : f"{repglob}calc_{slice}/data/", 
          'rep_save' : f"{repglob}pickle/",
          'name_save' : f"manchadela_pions_{slice}"}
  csv2pickle(**kwpi)



# %% sauvegarde du script de batch :
# Get the absolute path of the running script
script_path = os.path.abspath(__file__)
# Get the file name
script_name = os.path.basename(script_path)
shutil.copy(script_path,repglob)
print(f"batch script {script_name} saved into {repglob}")
shutil.copy(f'{repglob}calc_1/{filename}',repglob)
print(f"1st .dgibi saved into {repglob}")
#%% menage :
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
# %%  MENAGE
# print(f"cleaning...")
# # Define the pattern for CSV files
# csv_pattern = os.path.join(repglob, '**', '*.csv')
# ps_pattern = os.path.join(repglob, '**', '*.ps')
# trace_pattern = os.path.join(repglob, '**', '*.trace')

# # Get a list of all CSV files using glob
# csv_files = glob.glob(csv_pattern, recursive=True)
# ps_files = glob.glob(ps_pattern, recursive=True)
# trace_files = glob.glob(trace_pattern, recursive=True)

# # Delete each CSV file
# for csv_file,ps_file,trace_file in zip(csv_files,ps_files,trace_files):
#     try:
#         os.remove(csv_file)
#         os.remove(ps_file)
#         os.remove(trace_file)
#         print(f"Deleted: {csv_file}")
#         print(f"Deleted: {ps_file}")
#         print(f"Deleted: {trace_file}")
#     except Exception as e:
#         print(f"Error deleting {csv_file}: {e}")