import cv2
import numpy as np

# phép lấy nghịch đảo 1 ảnh
def complement(img):
    return abs(np.array(img, copy=True) - 1)

# phép giao 2 ảnh
def and_img(img1, img2):
    result = np.zeros((img1.shape[0],img1.shape[1]))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j] == img2[i,j]:
                result[i,j] = img1[i,j]
    return result

# phép trừ 2 ảnh
def subtract(img1, img2):
    result = np.array(img1, copy=True)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j] == img2[i,j]:
                result[i,j] = 0
    return result

# toán tử erosion
def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    # đếm xem tổng số pixel trùng lắp giữa kernel và image có bằng số pixel 1 của kernel không
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum():
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1

    return eroded_img[:img_shape[0], :img_shape[1]]


'''
TODO: implement morphological operators
'''

#toán tử dilation
def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    dilate = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    # duyệt toàn bộ ảnh, ở những vị trí pixel bằng 1 thì thực hiện in vết kernel lên image
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i, j] == 1.0:
                for m in range(kernel.shape[0]):
                    for n in range(kernel.shape[1]):
                        if kernel[m, n] == 1:
                            dilate[i + m, j + n] = 1

    return dilate[:img_shape[0], :img_shape[1]]


def open(img, kernel):
    # thực hiện erosion rồi dilation
    erode_img = erode(img, kernel)
    open_img = dilate(erode_img, kernel)
    return open_img


def close(img, kernel):
    # thực hiện dilation ori62 erosion
    dilate_img = dilate(img, kernel)
    close_img = erode(dilate_img, kernel)
    return close_img


def hitOrMiss(img, kernel):
    # thực hiện tạo 2 mặt nạ, mặt nạ hit và mặt nạ miss
    kernel_hit = np.zeros((kernel.shape[0], kernel.shape[1]))
    kernel_miss = np.zeros((kernel.shape[0], kernel.shape[1]))
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if kernel[i, j] == 1:
                kernel_hit[i, j] = 1
            if kernel[i, j] == -1:
                kernel_miss[i, j] = 1
    # erode ảnh gốc bởi mặt nạ hit
    dilate_kernel_hit = erode(img, kernel_hit)
    complement_img = complement(img)
    # erode ảnh nghịch đảo ảnh gốc bởi mặt nạ miss
    dilate_kernel_miss = erode(complement_img, kernel_miss)
    # lấy giao 2 ảnh trên thu được kết quả
    hitOrMiss_img = and_img(dilate_kernel_miss, dilate_kernel_hit)
    return hitOrMiss_img


def thinning(img):
    # khai báo 8 mặt nạ
    kernels = [np.array([[-1, -1, -1],
                         [0, 1, 0],
                         [1, 1, 1]]),
               np.array([[0, -1, -1],
                         [1, 1, -1],
                         [1, 1, 0]]),
               np.array([[1, 0, -1],
                         [1, 1, -1],
                         [1, 0, -1]]),
               np.array([[1, 1, 0],
                         [1, 1, -1],
                         [0, -1, -1]]),
               np.array([[1, 1, 1],
                         [0, 1, 0],
                         [-1, -1, -1]]),
               np.array([[0, 1, 1],
                         [-1, 1, 1],
                         [-1, -1, 0]]),
               np.array([[-1, 0, 1],
                         [-1, 1, 1],
                         [-1, 0, 1]]),
               np.array([[-1, -1, 0],
                         [-1, 1, 1],
                         [0, 1, 1]])]
    prev_img = np.array(img, copy=True)
    prev_loop_img = np.array(img, copy=True)
    while_loop = 0
    # Lặp đến khi 2 ảnh trước và sau không còn sự khác biệt
    while True:
        while_loop += 1
        print("loop: ", while_loop)
        for i in range(8):
            current_img = and_img(prev_img, complement(hitOrMiss(prev_img, kernels[i])))
            prev_img = np.array(current_img, copy=True)
        if (current_img == prev_loop_img).all():
            break
        prev_loop_img = current_img

    return current_img

def boundary_extraction(img, kernel):
    # thực hiện erode ảnh
    erode_img = erode(img, kernel)
    # lấy ảnh gốc trừ ảnh erode
    boundary_extraction_img = subtract(img, erode_img)*255
    return boundary_extraction_img