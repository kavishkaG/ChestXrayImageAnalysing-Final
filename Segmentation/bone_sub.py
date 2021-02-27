
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt

# In[ ]:


path = 'C:\\Users\\Dell\\Desktop\\Effusion\\00000011_000.png'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

imgplot = plt.imshow(image)

dimension = (1024, 1024)

# resize
resized = cv2.resize(image, dimension)

# globale enhancement
gray1 = cv2.equalizeHist(resized)

# local enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray1 = clahe.apply(gray1)
# median filter
denoised_img = cv2.medianBlur(gray1, 5)

# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width = denoised_img.shape

dimension_2 = (500, 500)
resized_img = cv2.resize(image, dimension_2)

cv2.imshow('resized_img', resized_img)
preprocessed_image = resized_img


# In[ ]:


def deleteRow(array, index):
    array = np.delete(array, index, axis=0)
    return array


def remove_boundary(image):
    thresh1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 401, 6)

    cv2.imshow('thresh1', thresh1)

    # applying erosion
    kernel = np.ones((5, 5), np.uint8)
    thresh1 = cv2.erode(thresh1, kernel, iterations=1)

    cv2.imshow('thresh1ero', thresh1)

    X = image.shape[1]
    Y = image.shape[0]

    # drawing boundary edge
    constant = cv2.line(thresh1, (0, 0), (X, 0), (255, 255, 255), 5)
    constant = cv2.line(constant, (0, 0), (0, Y), (255, 255, 255), 5)
    constant = cv2.line(constant, (X, 0), (X, Y), (255, 255, 255), 5)
    constant = cv2.line(constant, (0, Y), (X, Y), (255, 255, 255), 5)

    # calculate number of pixel clusters
    markers = cv2.connectedComponentsWithStats(constant, 8, cv2.CV_32S)

    # Isolate stats of connected components.
    marker_area_2 = markers[2]

    # build empty image
    lung_mask_2 = np.zeros(image.shape, np.uint8)

    # remove components at boundary
    for index, i in enumerate(marker_area_2):
        if i[0] == 0 and i[1] == 0:
            marker_area_3 = deleteRow(marker_area_2, index)

        else:
            lung_mask_2 = lung_mask_2 + (markers[1] == index)

    # obtain new image by removing edge components

    thresh1[lung_mask_2 == False] = 0

    cv2.imshow('thresh1erofaf', thresh1)

    # dilation and closing to refine mask
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closing', closing)

    dilation = cv2.dilate(closing, kernel, iterations=1)
    cv2.imshow('dilation', dilation)
    # dimension_2 = (500, 500)
    # resized_dilation_img = cv2.resize(dilation, dimension_2)

    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    # photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized_dilation_img))

    # label_pht = Label(image=photo)
    # label_pht.image = photo  # keep a reference!
    # label_pht.grid(row=7, column=1, pady=1)

    # save_img("_stage1.png", dilation)
    return dilation


boundry_removed = remove_boundary(preprocessed_image)


def filter_by_location(image):
    mask = np.zeros(image.shape, np.uint8)
    # cv2.imshow('mask', mask)
    contour, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print('contour', contour[0])
    cnt_sorted = sorted(contour, key=cv2.contourArea, reverse=True)

    print('cnt_sorted', cnt_sorted[0])

    image_width = image.shape[1]
    image_center = image_width / 2

    M = cv2.moments(cnt_sorted[0])
    # print('M[m10]', M)
    cx1 = int(M['m10'] / M['m00'])
    is_largest_left = False

    if cx1 > image_center:
        is_largest_left = True

    considered_cnt_lst2 = [cnt_sorted[0]]

    for i in range(1, len(contour)):
        M2 = cv2.moments(contour[i])
        # print('M2[m10]', M2)
        cx = int(M2['m10'] / M2['m00'])
        if is_largest_left:
            if cx < cx1:
                considered_cnt_lst2.append(contour[i])
        else:
            if cx > cx1:
                considered_cnt_lst2.append(contour[i])

    for cnt in considered_cnt_lst2:
        cv2.drawContours(mask, [cnt], -1, (255, 0, 0), -1)

    cv2.imshow('mask', mask)

    return mask


filter_by_location_image = filter_by_location(boundry_removed)


def filter_by_size(image):
    mask = np.zeros(image.shape, np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    dilation_image = cv2.dilate(closing, kernel, iterations=1)

    contour, hier = cv2.findContours(dilation_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sorted = sorted(contour, key=cv2.contourArea, reverse=True)
    largest_area_1 = cv2.contourArea(cnt_sorted[0])
    largest_area_2 = cv2.contourArea(cnt_sorted[1])

    sensitivity_1 = 0.14
    sensitivity_2 = 2

    area_ratio_1 = (largest_area_1 - largest_area_2) / largest_area_1
    print(area_ratio_1)
    considered_cnt_lst = [cnt_sorted[0], cnt_sorted[1]]

    if area_ratio_1 > sensitivity_1:

        for i in range(2, len(cnt_sorted)):
            if largest_area_2 < (cv2.contourArea(cnt_sorted[i]) * sensitivity_2):
                considered_cnt_lst.append(cnt_sorted[i])
            else:
                break

    for cnt in considered_cnt_lst:
        cv2.drawContours(mask, [cnt], -1, (255, 0, 0), -1)

    cv2.imshow('mask111', mask)
    return mask


# filer_by_size_image = filter_by_size(filter_by_location_image)
filer_by_size_image = filter_by_size(boundry_removed)


def filter_by_distance(image):
    mask = np.zeros(image.shape, np.uint8)

    contour, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)

    M = cv2.moments(contour[0])
    cx1 = int(M['m10'] / M['m00'])

    M1 = cv2.moments(contour[1])
    cx = int(M1['m10'] / M1['m00'])
    shortest_distance = abs(cx1 - cx)
    selected_cnt = contour[1]

    for i in range(2, len(contour)):
        M = cv2.moments(contour[i])
        cx = int(M['m10'] / M['m00'])
        distance = abs(cx1 - cx)
        if shortest_distance > distance:
            shortest_distance = distance
            selected_cnt = contour[i]

    final_contour_lst = [contour[0], selected_cnt]

    for cnt in final_contour_lst:
        cv2.drawContours(mask, [cnt], -1, (255, 0, 0), -1)

    cv2.imshow('maskkvhldh', mask)
    return mask


filter_by_distance_image = filter_by_distance(filer_by_size_image)


def refine_mask(image):
    mask = np.zeros(image.shape, np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    temp_image = image.copy()
    for i in range(2):
        contour, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour) > 1:
            temp_image = image.copy()
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.dilate(image, kernel, iterations=1)
        else:
            image = temp_image.copy()
            break

    contour, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(mask, [cnt], -1, (255, 0, 0), -1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    cv2.imshow('GaussianBlur', mask)

    return mask


refine_mask_image = refine_mask(filter_by_distance_image)


def find_ROI(image, resized_img):

    original = resized_img

    lung_areas = []

    contour, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(len(contour))
    if len(contour) < 2:
        return lung_areas

    else:
        cnt_sorted = sorted(contour, key=cv2.contourArea, reverse=True)

        largest_area_1 = cnt_sorted[0]
        largest_area_2 = cnt_sorted[1]

        M = cv2.moments(largest_area_1)
        cx1 = int(M['m10'] / M['m00'])

        M = cv2.moments(largest_area_2)
        cx2 = int(M['m10'] / M['m00'])

        if cx2 > cx1:
            left_lung = largest_area_2
            right_lung = largest_area_1

        else:
            right_lung = largest_area_2
            left_lung = largest_area_1

        for cnt in right_lung:
            hull_right = cv2.convexHull(cnt)
            cv2.drawContours(original, [cnt], -1, (255, 0, 0), 2)

        hull_right = cv2.convexHull(right_lung)
        cv2.drawContours(original, [hull_right], -1, (255, 0, 0), 3)

        for cnt in left_lung:
            hull_right = cv2.convexHull(cnt)
            cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)

        hull_left = cv2.convexHull(left_lung)
        x = cv2.drawContours(original, [hull_left], -1, (0, 255, 0), 3)
        cv2.imshow('GaussianBlurx', x)
        x, y, w, h = cv2.boundingRect(left_lung)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y + (h + 50))
        fontScale = 1.5
        fontColor = (0, 255, 0)
        lineType = 2

        cv2.putText(original, 'Left Lung',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        x, y, w, h = cv2.boundingRect(right_lung)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y + (h + 50))
        fontScale = 1.5
        fontColor = (255, 0, 0)
        lineType = 2

        y = cv2.putText(original, 'Right Lung',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)


        # cv2.imshow('GaussianBlury', y)

        dimension_2 = (500, 500)
        resized_mask_img = cv2.resize(original, dimension_2)
        cv2.imshow('GaussianBlury', resized_mask_img)
        # plt.imshow(resized_mask_img)

        lung_areas.append(left_lung)
        lung_areas.append(right_lung)
        return lung_areas


lung_areas = find_ROI(refine_mask_image, resized_img)


def get_morph_features(left_lung, right_lung):
    left_lung_area = cv2.contourArea(left_lung)
    right_lung_area = cv2.contourArea(right_lung)

    area_difference = abs(right_lung_area - left_lung_area)
    # as a percentage
    area_difference = (area_difference / (right_lung_area + left_lung_area)) * 100
    area_difference = round(area_difference, 5)

    left_lung_hull = cv2.convexHull(left_lung)
    right_lung_hull = cv2.convexHull(right_lung)

    left_lung_hull_area = cv2.contourArea(left_lung_hull)
    right_lung_hull_area = cv2.contourArea(right_lung_hull)
    hull_area_difference = abs(right_lung_hull_area - left_lung_hull_area)
    # as a percentage
    hull_area_difference = (hull_area_difference / (right_lung_hull_area + left_lung_hull_area)) * 100
    hull_area_difference = round(hull_area_difference, 5)

    left_lung_deviation = abs(left_lung_hull_area - left_lung_area)
    # as a percentage
    left_lung_deviation = (left_lung_deviation / left_lung_hull_area) * 100
    left_lung_deviation = round(left_lung_deviation, 5)

    right_lung_deviation = right_lung_hull_area - right_lung_area
    # as a percentage
    right_lung_deviation = (right_lung_deviation / right_lung_hull_area) * 100
    right_lung_deviation = round(right_lung_deviation, 5)

    features = [area_difference, hull_area_difference, left_lung_deviation, right_lung_deviation]

    return features


morph_features = get_morph_features(lung_areas[0], lung_areas[1])
print(morph_features)


def get_texture_features(left_lung, right_lung, sup_img):
    x, y, w, h = cv2.boundingRect(left_lung)
    cropped_L = sup_img[y:y + h, x:x + w]
    cv2.imshow('coooo', cropped_L)

    g = greycomatrix(cropped_L, [1], [0], levels=256, symmetric=False, normed=True)
    print('g', g[0][0])

    contrast_L = greycoprops(g, 'contrast')[0][0]
    print('contrast_L', contrast_L)
    energy_L = greycoprops(g, 'energy')[0][0]
    # print('energy_L', energy_L)
    homogeneity_L = greycoprops(g, 'homogeneity')[0][0]
    # print('homogeneity_L', homogeneity_L)
    correlation_L = greycoprops(g, 'correlation')[0][0]
    # print('correlation_L', correlation_L)
    asm_L = greycoprops(g, 'ASM')[0][0]
    # print('asm_L', asm_L)
    dissimilarity_L = greycoprops(g, 'dissimilarity')[0][0]
    # print('dissimilarity_L', dissimilarity_L)

    x, y, w, h = cv2.boundingRect(right_lung)
    cropped_R = sup_img[y:y + h, x:x + w]

    g = greycomatrix(cropped_R, [1], [0], levels=256, symmetric=False, normed=True)

    contrast_R = greycoprops(g, 'contrast')[0][0]
    energy_R = greycoprops(g, 'energy')[0][0]
    homogeneity_R = greycoprops(g, 'homogeneity')[0][0]
    correlation_R = greycoprops(g, 'correlation')[0][0]
    asm_R = greycoprops(g, 'ASM')[0][0]
    dissimilarity_R = greycoprops(g, 'dissimilarity')[0][0]

    features = [contrast_L, energy_L, homogeneity_L, correlation_L, asm_L, dissimilarity_L,
                contrast_R, energy_R, homogeneity_R, correlation_R, asm_R, dissimilarity_R]

    return features


texture_features = get_texture_features(lung_areas[0], lung_areas[1], resized_img)
print(texture_features)

cv2.waitKey(0)
