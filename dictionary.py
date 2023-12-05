import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

def lakes(image):
    A = ~image
    AA = np.ones((A.shape[0] + 2, A.shape[1] + 2))
    AA[1:-1, 1:-1] = A
    return np.max(label(AA)) - 1

def has_wline(image):
    lines = np.sum(image, 1) // image.shape[1]
    return 1 in lines

def has_vline(image):
    lines = np.sum(image, 0) // image.shape[0]
    return 1 in lines

def has_bay(image):
    a = ~image
    aa = np.zeros((a.shape[0] + 1, a.shape[1])).astype("uint8")
    aa[:-1, :] = a
    return lakes(~aa) -1

def count_bays(image):
    holes = ~image.copy()
    return np.max(label(holes))

def recognize(region):
    k = lakes(region.image)
    if k == 0:
        if has_vline(region.image):
            if count_bays(region.image) == 5:
                return '*'
            if np.all(region.image == 1):
                return '-'
            return '1'
        if count_bays(region.image) == 5:
            if has_wline(region.image):
                return '*'
            return 'W'
        if count_bays(region.image) == 2:
            return '/'
        if count_bays(region.image[2:-2, 2:-2]) == 5:
            return '*'
        else:
            return 'X'
    if k == 1:
        if has_vline(region.image):
            if count_bays(region.image) > 3:
                return '0'
            else:
                if (region.perimeter**2)/region.area < 59:
                    return 'P'
                else:
                    return 'D'
        else:
            if count_bays(region.image) < 5:
                return 'A'
            else:
                return '0'
    if k == 2:
        if count_bays(region.image) > 4:
            return '8'
        else:
            return 'B'
    return None


image = plt.imread("symbols.png")
image = np.sum(image,2)
image[image > 0] = 1

labeled = label(image)

regions = regionprops(labeled)

proc = {}
for region in regions:
    symbol = recognize(region)
    if symbol not in proc:
        proc[symbol] = 1
    else:
        proc[symbol] += 1

sum = 0
for key in proc.keys():
    proc[key] = proc[key]/np.max(labeled) * 100
    sum += proc[key]

print(proc)

plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(labeled)
plt.show()