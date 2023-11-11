import os
import matplotlib.pyplot as plt
import numpy as np

def resolution_calculator(filename):
    with open(filename, "r") as file:
        img_size = float(file.readline())
        img_data = np.loadtxt(file)

    obj_columns = [col for col in img_data.T if 1 in col]
    obj_height = len(obj_columns)

    if obj_height == 0:
        return 0

    return img_size / obj_height

def image_display(ax, filename, resolution, figure_number):
    with open(filename, "r") as file:
        _ = file.readline()
        img_data = np.loadtxt(file)

    ax.imshow(img_data)
    ax.set_title(f"Объект {figure_number + 1}")
    ax.set_xlabel(f"Разрешение = {resolution} мм/пиксель")

input_directory = "input_images"
input_files = os.listdir(input_directory)
fig_resolutions = []
fig, axes = plt.subplots(2, 3, figsize=(8, 6))

for fig_file in input_files:
    fig_resolutions.append(round(resolution_calculator(os.path.join(input_directory, fig_file)), 3))

for i, fig_file in enumerate(input_files):
    j, k = divmod(i, 3)
    image_display(axes[j, k], os.path.join(input_directory, fig_file), fig_resolutions[i], i)

plt.subplots_adjust(wspace=1.9)
plt.show()
