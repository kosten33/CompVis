import numpy as np
import matplotlib.pyplot as plt

def load_image(filename):
    with open(filename) as file:
        _ = file.readline()
        img_data = np.loadtxt(file)
    return img_data

def calculate_offset(img1, img2):
    correlation_result = np.correlate(img1.ravel(), img2.ravel(), mode='full')
    y, x = divmod(np.argmax(correlation_result), img2.shape[1])
    return y - img1.shape[0] + 1, x - img1.shape[1] + 1

def save_image_with_offset(shifted_img, offset, filename):
    with open(filename, 'w') as file:
        file.write(f"Сдвиг (y, x): {offset}\n")
        np.savetxt(file, shifted_img, fmt='%d')

image1 = load_image("input_images/img1.txt")
image2 = load_image("input_images/img2.txt")

calculated_offset = calculate_offset(image1, image2)
print(f"Сдвиг (y, x): {calculated_offset}")

save_image_with_offset(image2, calculated_offset, "output/image2_shifted.txt")

plt.imshow(image2)
plt.title(f"Сдвиг (y, x): {calculated_offset}")
plt.show()
