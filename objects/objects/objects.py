import numpy as np
from skimage.measure import label
from skimage.morphology import binary_opening

def analyze_structure(img, structure):
    labeled_img = label(img)
    result = label(binary_opening(labeled_img, structure)).max()
    return result

def main():
    img = np.load("psnpy.txt")

    mask_1 = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1]])

    mask_2 = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 1, 1],
                      [1, 1, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1, 1],
                      [1, 1, 1,1, 1, 1]])

    mask_3 = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1],
                      [1, 1, 0, 0, 1, 1],
                      [1, 1, 0, 0, 1, 1]])

    mask_4 = np.array([[0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1]])

    mask_5 = np.array([[0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1]])

    print(f"Type_1: {analyze_structure(img, mask_1)}")
    print(f"Type_4: {analyze_structure(img, mask_4)}")
    print(f"Type_5: {analyze_structure(img, mask_5)}")
    print(f"Type_2: {analyze_structure(img, mask_2) - analyze_structure(img, mask_1)}")
    print(f"Type_3: {analyze_structure(img, mask_3) - analyze_structure(img, mask_1)}")
    print(f"Total: {np.max(label(img))}")

if __name__ == "__main__":
    main()
