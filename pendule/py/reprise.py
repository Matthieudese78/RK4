#!/bin/python3
# goal : Creation d'un tableau de reprise
# %%
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import subprocess
import shutil

# import glob
# import sys
# import json
# import rotation
from csv_to_pickle import csv2pickle


# %% pour prediction de l'instant d'impact :
def pendulum_motion(y, t, L, g, M, J):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = (M * g * (L / 2.0) * np.sin(theta)) / J
    return [dtheta_dt, domega_dt]

# %%
lclean = True
lstdout = True
# %% lraidtimo :
# %% script / dir :
filename = "pendule_timo.dgibi"
rawname = filename.split(".")[0]
original_directory = os.getcwd()
source = f"../{filename}"
destination = f"../calc/"
repcast = f"../../build/"
# slicevtk = 50
# %% parametres du calcul
stoia = "vrai"
manchette = "faux"
trig = "vrai"
limpact = "vrai"
linert = "vrai"
lnortot = "faux"
lnorcomp = "vrai"
lnormtot = "faux"
# algo 
rk4 = "vrai" 
nmb = "faux" 
sw = "faux" 
#
flvtk = "vrai"
#
h = 0.6
g = 9.81
M = 0.59868
Jx = 0.07185
#
bamo = 0
Kchoc = 5.5e07
xi = bamo / (2.0 * M * (np.sqrt(Kchoc / M)))
# pour trig vitesse normale a l'impact (m/s):
vimpact = 4.
# tourne avec 20 modes : dte = 2.e7
dte = 1.0e-6
nsort = 10
#
nmode_ela = 20
typmode = 1
if (typmode==2):
  nmode_ela = 0
# angle d'incidence :
# theta_ini_x = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
thinc = [10.]
#%%
# prediction de l'instant d'impact :
tc = []
if (not (trig=="vrai")):
  for i, thi in enumerate(thinc):
      thini = np.pi/2.0 - thi*np.pi/180.0
      initial_conditions = [thini, 0.0]
      # Time points to integrate over
      t = np.linspace(0, 2.0, int(1e5))
      # Integrate the differential equation
      solution = odeint(pendulum_motion, initial_conditions, t, args=(h, g, M, Jx))
      # Extract theta and omega from the solution
      theta, omega = solution[:, 0], solution[:, 1]
      # instant choc :
      thc = thini + 2.0*(thi*np.pi/180.0)
      crit = 1.0e-4
      ichoc = np.where(np.abs(theta - thc) < crit)[0]
      tci = np.min(t[ichoc]) + 2.0e-2
      tc.append(tci)
      # plt.plot(t, theta, label='Theta (rad)')
      # plt.plot(t, omega, label='Omega (rad/s)')
      # plt.xlabel('Time (s)')
      # plt.ylabel('Values')
      # plt.title('Pendulum Motion')
      # plt.legend()
      # plt.grid(True)
      # plt.show()

if (trig=="vrai"):
  # le supplement d'amplitude du a la rotation ini de la poutre est pris en compte dans jeu1 du .dgibi
  for i, thi in enumerate(thinc):
    #   tc.append((vimpact/g)+2.e-2)
      tc.append(0.5)

print("Instants choc : ")
[print(tci) for tci in tc]
# %% loop :
for icalc, thi in enumerate(thinc):
    thini = np.pi / 2.0 - (thi * np.pi / 180.0)
    # un quart de periode de pendule :
    # t = 0.25*(2.*np.pi*np.sqrt(h/g)*(1.+((np.pi-thini)**2/16.)))
    t = tc[icalc]
    print(f"t = {t}")
    # repertoire de sauvegarde :
    nameglob = ""
    # si couplage inertiel, on le met :
    if stoia == "vrai":
        nameglob = f"{nameglob}stoia/"
    if manchette == "vrai":
        nameglob = f"{nameglob}manchette/"
    if trig == "vrai":
        nameglob = f"{nameglob}trig/"
    if limpact == "vrai":
        nameglob = f"{nameglob}impact/"
    if linert == "vrai":
        nameglob = f"{nameglob}inert/"
    if lnorcomp == "vrai":
        nameglob = f"{nameglob}norcomp/"

    nameglob = f"{nameglob}xi_{int(100.*xi)}/thinc_{int(thi)}/nmode_{nmode_ela}"

    repsave = f"{destination}data/{nameglob}/"

    dictini = {
        #          stoia :
        "stoia": stoia,
        #          manchette :
        "manchette": manchette,
        #          trig :
        "trig": trig,
        #          limpact :
        "limpact": limpact,
        #          linert :
        "linert": linert,
        #          lnorcomp :
        "lnorcomp": lnorcomp,
        #          lnortot :
        "lnortot": lnortot,
        #          lnormtot :
        "lnormtot": lnormtot,
        #          algo :
        "rk4"     : rk4, 
        "nmb"     : nmb, 
        "sw"      : sw ,
        #          bamo :
        "bamo": bamo,
        #          t :
        "t": t,
        #          dte :
        "dte": dte,
        #          nsort :
        "nsort": nsort,
        #          typmode :
        "typmode": typmode,
        #          nmode :
        "nmode_ela": nmode_ela,
        #          thinc :
        "thinc": thi,
        #          vimpact : pour le cas trig
        "vimpact": vimpact,
        #
        "flvtk" : flvtk,
    }

    dfini = pd.DataFrame(dictini, index=[0])
    # coordonnees modales :

    # creation des repos :
    if not os.path.exists(destination):
        os.makedirs(destination)
        print(f"FOLDER : {destination} created.")
    else:
        print(f"FOLDER : {destination} already exists.")

    if not os.path.exists(f"{destination}fig/"):
        os.makedirs(f"{destination}fig/")
        print(f"FOLDER : {destination}fig/ created.")
    else:
        print(f"FOLDER : {destination}fig/ already exists.")

    if not os.path.exists(f"{destination}depl_vtk/"):
        os.makedirs(f"{destination}depl_vtk/")
        print(f"FOLDER : {destination}depl_vtk/ created.")
    else:
        print(f"FOLDER : {destination}depl_vtk/ already exists.")

    if not os.path.exists(repsave):
        os.makedirs(repsave)
        print(f"FOLDER : {repsave} created.")
    else:
        print(f"FOLDER : {repsave} already exists.")

    # on copie le script :
    shutil.copy(source, f"{destination}{filename}")
    # on copie l'executable :
    shutil.copy(f"{repcast}cast_64_21", f"{destination}cast_64_21")

    # Read the content of the existing file
    with open(f"{destination}{filename}", "r") as file:
        lines = file.readlines()

    # Update the values based on the DataFrame
    # for index, row in dfini.iterrows():
    for colname, colvalue in dfini.iteritems():
        # Assuming the column names match the tags in the file (e.g., val1, val2)
        tag = f"*# {colname} :"
        # print(tag)
        replacement = f"{colname} = {colvalue.values[0]} ;"
        # print(replacement)

        # Find and replace the line with the updated value
        for i in range(len(lines)):
            if lines[i].strip() == tag:
                lines[i + 1] = f"{replacement}\n"
                break

    # Write the modified content back to the file
    with open(f"{destination}{filename}", "w") as file:
        file.writelines(lines)

    # 1ER CALCUL
    os.chdir(destination)
    #
    print(f"calcul {icalc+1} / {len(thinc)}")
    print(f"angle incidence {thi}")
    print(f"angle initial {90. - thi}")
    # castem21 $script > /dev/null 2>error.log
    if lstdout:
        command_to_run = [f"castem21 {filename}"]
    else:
        command_to_run = [f"castem21 {filename} > /dev/null 2>error.log"]
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
    kwpi = {
        "rep_load": repsave,
        "rep_save": f"./pickle/{nameglob}/",
        "name_save": f"result",
    }
    csv2pickle(**kwpi)

# sys.exit()
