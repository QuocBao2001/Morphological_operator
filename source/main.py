import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import grayScale


def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    if (mor_op[0:2] == "GS"):
        img = img_gray
    else:
        (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('binary image', img)
        cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    img_out = None

    '''
    TODO: implement morphological operators
    '''
    # toán tử dilation
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
    # toán tử erosion
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual
    # toán tử openning
    elif mor_op == 'open':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV open image', img_opening)
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_opening_manual = binary.open(img, kernel)
        cv2.imshow('manual open image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual
    # toán tử closing
    elif mor_op == 'close':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV close image', img_closing)
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_closing_manual = binary.close(img, kernel)
        cv2.imshow('manual close image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
    # toán tử hit or miss
    elif mor_op == 'hitOrMiss':
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        img_hit_or_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('OpenCV hit or miss image', img_hit_or_miss)
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_hit_or_miss_manual = binary.hitOrMiss(img, kernel)
        cv2.imshow('manual hit or miss image', img_hit_or_miss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hit_or_miss_manual
    # toán tử làm mảnh
    elif mor_op == 'thinning':
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        img = img // 255
        img_thinning_manual = binary.thinning(img)
        cv2.imshow('manual close image', img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual
    # toán tử boundary extraction
    elif mor_op == 'boex':

        img = img // 255
        img_boex_manual = binary.boundary_extraction(img, kernel)
        cv2.imshow('manual boundary extraction image', img_boex_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boex_manual
    # toán tử dilation độ xám
    elif mor_op == 'GSdilate':
        img_gs_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV grayscale dilation image', img_gs_dilation)
        cv2.waitKey(wait_key_time)

        img_gs_dilation_manual = grayScale.dilate(img, kernel)
        cv2.imshow('manual grayscale dilation image', img_gs_dilation_manual)
        cv2.waitKey(wait_key_time)

        cv2.imshow('different', img_gs_dilation - img_gs_dilation_manual)
        cv2.waitKey(wait_key_time)
        img_out = img_gs_dilation_manual
    # toán tử erosion độ xám
    elif mor_op == 'GSerode':
        img_gs_dilation = cv2.erode(img, kernel)
        cv2.imshow('OpenCV grayscale erosion image', img_gs_dilation)
        cv2.waitKey(wait_key_time)

        img_gs_erosion_manual = grayScale.erode(img, kernel)
        cv2.imshow('manual grayscale erosion image', img_gs_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_erosion_manual
    # toán tử opening độ xám
    elif mor_op == 'GSopen':
        img_gs_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV grayscale opening image', img_gs_opening)
        cv2.waitKey(wait_key_time)

        img_gs_opening_manual = grayScale.open(img, kernel)
        cv2.imshow('manual grayscale opening image', img_gs_opening_manual)
        cv2.waitKey(wait_key_time)
        cv2.imshow('different', img_gs_opening - img_gs_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_opening_manual
    # toán tử closing độ xám
    elif mor_op == 'GSclose':
        img_gs_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV grayscale closing image', img_gs_closing)
        cv2.waitKey(wait_key_time)

        img_gs_closing_manual = grayScale.close(img, kernel)
        cv2.imshow('manual grayscale closing image', img_gs_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_closing_manual
    # toán tử morphological gradient
    elif mor_op == 'GSgradient':
        img_gs_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)
        cv2.imshow('OpenCV grayscale gradient image', img_gs_gradient)
        cv2.waitKey(wait_key_time)

        img_gs_gradient_manual = grayScale.gradient(img, kernel)
        cv2.imshow('manual grayscale gradient image', img_gs_gradient_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_gradient_manual
    # toán tử đỉnh nón
    elif mor_op == 'GStophat':
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        img_gs_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV grayscale top hat image', img_gs_tophat)
        cv2.waitKey(wait_key_time)

        img_gs_tophat_manual = grayScale.tophat(img, kernel)
        cv2.imshow('manual grayscale top hat image', img_gs_tophat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_tophat_manual
    # toán tử black hat
    elif mor_op == 'GSblackhat':
        img_gs_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV grayscale black hat image', img_gs_blackhat)
        cv2.waitKey(wait_key_time)

        img_gs_blackhat_manual = grayScale.blackhat(img, kernel)
        cv2.imshow('manual grayscale black hat image', img_gs_blackhat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_blackhat_manual
    # toán tử textural segmentation
    elif mor_op == 'GStese':
        # img_gs_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        # cv2.imshow('OpenCV grayscale gradient image', img_gs_gradient)
        # cv2.waitKey(wait_key_time)

        img_gs_tese_manual = grayScale.textual_segmentation(img, kernel)
        cv2.imshow('manual grayscale gradient image', img_gs_tese_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gs_tese_manual
    if img_out is not None:
        cv2.imwrite(out_file, img_out * 255)


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey()


if __name__ == "__main__":
    main(sys.argv[1:])
