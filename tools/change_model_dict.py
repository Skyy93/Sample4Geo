import torch
import os
from tqdm import tqdm

def update_state_dict(state_dict, old_prefix='model.', new_prefix='base_model.'):
    """Hilfsfunktion, um die Schlüssel im Zustandswörterbuch zu aktualisieren."""
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace(old_prefix, new_prefix)
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict

def load_and_update_weights(folder_path):
    """Laden und aktualisieren der Gewichte für alle .pth Dateien im angegebenen Ordner."""
    pth_files = []
    # Zuerst alle Dateipfade erfassen
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
                
    # Fortschrittsbalken mit tqdm
    for file_path in tqdm(pth_files, desc="Updating weights"):
        # Laden Sie die gespeicherten Gewichte, stellen Sie sicher, dass sie auf der CPU sind
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))

        # Aktualisieren Sie die Schlüssel im Zustandswörterbuch
        updated_state_dict = update_state_dict(state_dict)

        # Überschreiben der originalen .pth Datei mit den aktualisierten Gewichten
        torch.save(updated_state_dict, file_path)
        print(f"Updated weights saved to {file_path}")

# Setzen Sie den Pfad zu Ihrem Ordner mit den .pth Dateien
folder_path = 'soundingearth/training/convnext_base.fb_in22k_ft_in1k_384/old'
load_and_update_weights(folder_path)
