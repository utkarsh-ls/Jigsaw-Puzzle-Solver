import gzip
import numpy as np


f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 1

import numpy as np
from PIL import Image
f.read(16)
for i in range(int(6e4)):
    print(i, end='\r')
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, image_size, image_size, 1)
    image = np.asarray(data).squeeze()
    Image.fromarray(image).save(f"mnist/{i}.png")
