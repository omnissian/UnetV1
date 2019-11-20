import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
img_object = Image.open('/storage/UserName/Task/validation/mask_build/212_509_750.tiff')
img_object = np.array(img_object)
plt.imshow(img_object)

#python -m pip install matplotlib
