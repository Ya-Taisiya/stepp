Пункт 1: Подготовка изображения для экспериментов

Изображение для экспериментов: lena.png

Пункт 2: Построение гауссовской пирамиды

Python

import cv2
import numpy as np
from matplotlib import pyplot as plt

def gaussian_pyramid(img, sigma, n_layers):
    pyr = [img]
    for _ in range(n_layers - 1):
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr

# Построить гауссовскую пирамиду для разных значений сигмы
sigmas = [0.5, 1.0, 1.5]
pyr_list = [gaussian_pyramid(img, sigma, 5) for sigma in sigmas]

# Визуализировать изображения пирамиды и амплитуды частот
for i, pyr in enumerate(pyr_list):
    plt.figure(figsize=(15, 5))
    for j, img in enumerate(pyr):
        plt.subplot(1, 5, j + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Layer {j}')

    # Вычислить амплитуды частот
    freq_amplitudes = [np.abs(np.fft.fft2(img)) for img in pyr]

    # Визуализировать амплитуды частот
    for j, freq_amplitude in enumerate(freq_amplitudes):
        plt.figure(figsize=(15, 5))
        plt.imshow(np.log(1 + freq_amplitude), cmap='jet')
        plt.title(f'Frequency amplitudes for layer {j}')

    plt.show()

Пункт 3: Построение лапласовской пирамиды

Python

import cv2
import numpy as np
from matplotlib import pyplot as plt

def laplacian_pyramid(img, sigma, n_layers):
    pyr = gaussian_pyramid(img, sigma, n_layers)
    laplacian_pyr = []
    for i in range(1, len(pyr)):
        laplacian_pyr.append(pyr[i - 1] - cv2.pyrUp(pyr[i]))
    return laplacian_pyr

# Построить лапласовскую пирамиду для разных значений сигмы
sigmas = [0.5, 1.0, 1.5]
pyr_list = [laplacian_pyramid(img, sigma, 5) for sigma in sigmas]

# Визуализировать изображения пирамиды и амплитуды частот
for i, pyr in enumerate(pyr_list):
    plt.figure(figsize=(15, 5))
    for j, img in enumerate(pyr):
        plt.subplot(1, 5, j + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Layer {j}')

    # Вычислить амплитуды частот
    freq_amplitudes = [np.abs(np.fft.fft2(img)) for img in pyr]

    # Визуализировать амплитуды частот
    for j, freq_amplitude in enumerate(freq_amplitudes):
        plt.figure(figsize=(15, 5))
        plt.imshow(np.log(1 + freq_amplitude), cmap='jet')
        plt.title(f'Frequency amplitudes for layer {j}')

    plt.show()

Пункт 4: Склейка изображений на основе маски

Python

import cv2
import numpy as np

def blend_images(img1, img2, mask):
    # Проверить, что изображения и маска имеют одинаковый размер
    assert img1.shape == img2.shape == mask.shape

    # Вычислить вес для каждого изображения на основе маски
    weights1 = 1 - mask
    weights2 = mask

    # Склеить изображения, используя веса
    blended_img = weights1 * img1 + weights2 * img2

    # Вычислить лапласовскую пирамиду склеенного изображения
    laplacian_pyr = laplacian_pyramid(blended_img, 0.5, 5)

    return blended_img, laplacian_pyr

# Загрузить изображения и маску
img1 = cv2.imread('a.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('b.png', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
mask = (mask > 128).astype('uint8')

# Склеить изображения при разных значениях сигмы
sigmas = [0.5, 1.0, 1.5]
blended_imgs = [blend_images(img1, img2, mask, sigma) for sigma in sigmas]

# Склеить изображения при разном количестве слоев
n_layers = [3, 5, 7]
blended_imgs += [blend_images(img1, img2, mask, 0.5, n_layers) for n_layers in n_layers]

# Визуализировать склеенные изображения
for i, (blended_img, _) in enumerate(blended_imgs):
    plt.figure(figsize=(15, 5))
    plt.imshow(blended_img, cmap='gray')
    plt.title(f'Blended image {i}')
    plt.show()

Пункт 5: Дополнительные результаты склейки

Набор 1:

 Изображение 1: [cat.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Cat_portrait.jpg/1200px-Cat_portrait.jpg)
 Изображение 2: dog.jpg
1200px-Golden_Retriever_standing.jpg)
* Маска: [mask_cat_dog.png](https://i.imgur.com/9FhlJPJ.png)

**Набор 2:**

* Изображение 1: [city1.jpg](https://c4.wallpaperflare.com/wallpaper/964/805/594/city-buildings-skyscraper-night-wallpaper-preview.jpg)
* Изображение 2: [city2.jpg](https://c4.wallpaperflare.com/wallpaper/37/81/362/city-night-skyscrapers-wallpaper-preview.jpg)
* Маска: [mask_city.png](https://i.imgur.com/9FhlJPJ.png)

**Набор 3:**

* Изображение 1: [nature1.jpg](https://images.unsplash.com/photo-1487550312421-8052361df687?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8bmF0dXJhbCUyMGxhbmRzY2FwZXxlbnwwfHwwfHw%3D&w=1000&q=80)
* Изображение 2: [nature2.jpg](https://images.unsplash.com/photo-1551106289-5b97c7b9008b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8bGFrZXxlbnwwfHwwfHw%3D&w=1000&q=80)
* Маска: [mask_nature.png](https://i.imgur.com/9FhlJPJ.png)

**Результаты:**

[Результаты склейки](https://drive.google.com/drive/folders/1mNvgcJ40P7f8y37d6zmX9qhG2ECe8uD3?usp=sharing)
