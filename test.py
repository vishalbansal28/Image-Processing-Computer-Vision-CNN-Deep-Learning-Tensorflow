import numpy as np

# Load the .npz file
data = np.load('orl_faces.npz')

# List all arrays in the .npz file
print(data.files)

for key in data.files:
    print(f'{key}: {data[key].shape}, {data[key].dtype}')
