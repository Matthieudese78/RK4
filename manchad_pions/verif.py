#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# relation amor pour un contact ccone total (a plat) a 1 m . s^-1 :
# 2.*np.pi*R_tete*1.*eta = x %
R_tete = 4.68e-2
amor = 1.
vref = 1. 
eta = amor / (2.*np.pi*R_tete*vref)

#%%

print(np.log(0.1))
print(np.log(10.))
#%%
a = 1.
b = 0.8*a
e = np.sqrt(1. - (b**2/a**2))

#%%
crit = 1.e-12
prec = crit/R_tete
#%%
w_ini = 2.*np.pi 
VZini = (-1.)*R_tete*w_ini 
#%% Paramètres
fs = 5000  # Fréquence d'échantillonnage en Hz
duree = 1  # Durée en secondes
x1 = 12  # Fréquence initiale en Hz
x2 = x1 + 2  # Fréquence initiale en Hz
T1 = (1./x1)
T2 = (1./x2)
t1 = np.linspace(0.,2.*T1,int(fs*2.*T1),endpoint=False)
t2 = np.linspace(0.,2.*T2,int(fs*2.*T2),endpoint=False)
tt = np.linspace(0.,T2,int(fs*T2),endpoint=False)

s1 = np.sin(2.*np.pi*x1*t1)
s2 = np.sin(2.*np.pi*x2*t2)

st1 = np.sin(2.*np.pi*x1*tt)*(1. - np.array([ti/T2 for i,ti in enumerate(tt)]))
st2 = np.sin(2.*np.pi*x2*tt)*np.array([ti/T2 for i,ti in enumerate(tt)])

s = np.append(s1,np.append((st1+st2),s2))
duree = 2.*T1+T2+2.*T2
# t = np.linspace(0.,duree,int(fs*duree)-1,endpoint=False)
t = np.linspace(0.,duree,num=len(s),endpoint=True)

plt.plot(t, s,)
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Transition en douceur de {} Hz à {} Hz'.format(x1, x2))
plt.grid(True)
plt.show()

#%% Générer le temps
t = np.linspace(0, duree, int(fs * duree), endpoint=False)

# Générer le signal sinusoidal initial (fréquence x)
signal_initial = np.sin(2 * np.pi * x * t)

# Générer le signal sinusoidal final (fréquence x+2)
signal_final = np.sin(2 * np.pi * (x + 2) * t)

[ti/duree for i,ti in enumerate(t)]

# Appliquer la transition en douceur
transitioned_signal = signal_initial * (1. - np.array([ti/duree for i,ti in enumerate(t)])) + signal_final * np.array([ti/duree for i,ti in enumerate(t)])

# Afficher le signal résultant
ttot = t = np.linspace(0, 3.*duree, int(fs * 3.*duree), endpoint=False)
plt.plot(ttot, np.append(np.append(signal_initial,transitioned_signal),signal_final))
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.title('Transition en douceur de {} Hz à {} Hz'.format(x, x + 2))
plt.grid(True)
plt.show()

#%%
T = 120.
tp = 2.
tt = 0.5
n = 50
# tt = (T - (tp*n)) / (n-1)
n = (T + tt) / (tp + tt)
#%%
mmanch = 11.46 
grav = 9.81
Kvert = 3.27043021 * 1.e9
u = (mmanch*grav) / Kvert 
#%%
XATRANS = 0.012533086738156748
Rcurv_Rcirc =   0.99756267933158926
XDN =   -9.4530744488239481E-005
crit = XATRANS*((Rcurv_Rcirc)-1.)
# %%
circ = 2.*np.pi*46.8e-3
lmax = (194.4*np.pi/180.)*46.8e-3
# %% Verif variation altitude P2 :
alt = 4.68e-3*np.sin(0.5*np.pi/180.)
horiz = 4.68e-3*np.cos(1.*np.pi/180.)

# %% Check limite flat case
#   Si on veut flat si XNUT <= 1.e-3° :
nutlim = 1.e-4*np.pi/180.

# %% CALCINC : ATAN(Y/X)
ang = np.arctan(-3.3742894657438890E-002/-3.3765999995097301E-002)*180./np.pi

# %%

# Paramètres du signal
frequence_initiale = 2  # Hz
frequence_finale = 20   # Hz
duree = 1200             # secondes
nombre_points = 1000000    # nombre de points pour l'échantillonnage

# Générer le temps échantillonné
temps = np.linspace(0, duree, nombre_points, endpoint=False)

# Générer le signal sinus balayé en fréquence
frequence = np.linspace(frequence_initiale, frequence_finale, nombre_points)
signal = np.sin(2 * np.pi * frequence * temps)

# Tracer le signal
plt.plot(temps, signal)
plt.title('Signal Sinusoidal Balayé en Fréquence')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# %%