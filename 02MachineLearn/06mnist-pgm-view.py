import glob
import os
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

directory_path = './resMnist'

file_pattern = os.path.join(directory_path, 't10k-*-*.pgm')
pgm_files = glob.glob(file_pattern)
print(pgm_files)

for pgm_file in pgm_files:
    img = Image.open(pgm_file)

    plt.imshow(img, cmap='gray')
    plt.title(f"image: {pgm_file}")
    plt.axis('off')
    plt.show()