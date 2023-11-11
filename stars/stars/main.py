import numpy as np
from skimage.measure import label
from skimage.morphology import binary_erosion

def count_stars(image, mask):
    labelled_data = label(image)
    result = label(binary_erosion(labelled_data, mask))

    return len(np.unique(result)) - 1

def main():
    img_data = np.load("stars.npy")

    plus_mask = np.array([[0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0]])

    cross_mask = np.array([[1, 0, 0, 0, 1],
                          [0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0],
                          [1, 0, 0, 0, 1]])

    stars_plus = count_stars(img_data, plus_mask)
    stars_cross = count_stars(img_data, cross_mask)

    total_stars = stars_plus + stars_cross
    print(f"Total number of stars: {total_stars}")

if __name__ == "__main__":
    main()
