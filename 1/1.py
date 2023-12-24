import numpy as np
import cv2

# Загрузка изображения
image = cv2.imread('task1.png', 0)

# Применение алгоритма бинаризации для получения черно-белого изображения
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Использование операций морфологии для устранения шумов и связывания областей
kernel = np.ones((3, 3), np.uint8)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

# Поиск контуров объектов
contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Нахождение контура с наибольшей площадью
largest_contour = max(contours, key=cv2.contourArea)

# Выделение найденного контура на изображении
result_image = cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

# Отображение изображений
cv2.imshow('Original Image', image)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()