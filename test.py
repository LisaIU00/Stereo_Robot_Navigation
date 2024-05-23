import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio per i grafici
frames = np.arange(1, 101)
dMain = np.random.rand(100) * 10  # Dati casuali per dMain
distanza_z = np.random.rand(100) * 100  # Dati casuali per la distanza z

# Creazione dei grafici
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Grafico 1: dMain
axs[0].plot(frames, dMain, color='blue')
axs[0].set_title('Average Disparity (dMain)')
axs[0].set_xlabel('Frame Number')
axs[0].set_ylabel('Disparity')

# Grafico 2: Distanza z
axs[1].plot(frames, distanza_z, color='green')
axs[1].set_title('Distance z (mm)')
axs[1].set_xlabel('Frame Number')
axs[1].set_ylabel('Distance (mm)')

# Aggiustamento dello spaziatura tra i grafici
plt.tight_layout()

# Visualizzazione dell'immagine
plt.show()