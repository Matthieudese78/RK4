
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

# Define parameters
duration = 128  # Signal duration in seconds
fs = 1.e6  # Sampling frequency in Hz
t = np.arange(0, duration, 1/fs)  # Time vector

# Plot the signal between 59 and 60 seconds
start_time = 59.  # seconds
end_time = 60.  # seconds

# Find indices corresponding to the time range
start_index = int(start_time * fs)
end_index = int(end_time * fs)

# Generate chirped sinusoidal signal
f_start = 2  # Starting frequency in Hz
f_end = 20  # Ending frequency in Hz
signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear',phi=-90.)
instantaneous_phase = np.unwrap(np.angle(signal))  # Unwrap the phase to handle phase wrapping
instantaneous_frequency = np.gradient(instantaneous_phase) * fs / (2 * np.pi)
# Plot the signal
# plt.plot(t, signal)
plt.plot(t[start_index:end_index], signal[start_index:end_index])
# plt.plot(instantaneous_phase, signal)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Chirped Sinusoidal Signal')
plt.grid(True)
plt.show()

# Open the file for writing
with open(f'signal{int(np.log10(fs))}.acc', 'w') as file:
    # Write each value of the signal to a separate line
    for value in signal:
        file.write(f'{value}\n')