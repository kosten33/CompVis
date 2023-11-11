import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, label

def check_wire_breakage(wires):
    labeled = label(wires)
    breaks = []

    for label_value in range(1, labeled.max() + 1):
        current_wire = np.zeros_like(wires)
        current_wire[labeled == label_value] = 1

        struct = np.ones((3, 1))
        current_wire = binary_erosion(current_wire, struct)

        reduced_labeled = label(current_wire)

        if reduced_labeled.max() > 1:
            num_breaks = reduced_labeled.max()
            breaks.append((label_value, num_breaks))
        else:
            breaks.append((label_value, 0))

    for label_value, num_breaks in breaks:
        if num_breaks == 0:
            print(f"Wire with label {label_value} is not broken.")
        else:
            print(f"Wire with label {label_value} is broken into {num_breaks} part(s).")

    plt.subplot(121)
    plt.imshow(labeled)
    plt.subplot(122)
    plt.imshow(wires)
    plt.show()

def process_wire_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        wires = np.load(image_path)
        check_wire_breakage(wires)

wire_images_folder = "input_images"

# Process wire images
process_wire_images(wire_images_folder)
