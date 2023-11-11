import numpy as np
from skimage import measure

coins_image = np.load('coins.npy')

labeled_coins = measure.label(coins_image)

count_1 = 0
count_2 = 0
count_5 = 0
count_10 = 0
total_value = 0

for region in measure.regionprops(labeled_coins):
    area = region.area
    centroid = region.centroid

    denomination = 0

    if area < 70:
        denomination = 1
        count_1 += 1
    elif area < 150:
        denomination = 2
        count_2 += 1
    elif area < 310:
        denomination = 5
        count_5 += 1
    else:
        denomination = 10
        count_10 += 1

    total_value += denomination

    print(f"Coin at {centroid} with denomination {denomination}")

print(f"Number of coins with denomination 1: {count_1}")
print(f"Number of coins with denomination 2: {count_2}")
print(f"Number of coins with denomination 5: {count_5}")
print(f"Number of coins with denomination 10: {count_10}")
print(f"Total value of all coins: {total_value}")
