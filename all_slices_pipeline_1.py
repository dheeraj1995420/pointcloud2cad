import open3d as o3d
import numpy as np
import pandas as pd
import math
import os
import cv2 as cv
from PIL import Image
import csv
import image_ai_BB_generator_2
import actual_coords_from_BB_3
import crop2mesh2replace_4
from matplotlib import pyplot as plt
from numpy import loadtxt
def pipeline(pcd_filename):
    pcd = o3d.io.read_point_cloud(pcd_filename + '.pcd')
    pcd_np = np.hstack((np.asarray(pcd.points), np.asarray(pcd.colors)))

    x_values = pcd_np[:, 0];
    y_values = pcd_np[:, 1];
    z_values = pcd_np[:, 2];
    r_values = pcd_np[:, 3];
    g_values = pcd_np[:, 4];
    b_values = pcd_np[:, 5];
    xmax = np.max(x_values);
    xmin = np.min(x_values);
    x_range = xmax - xmin
    ymax = np.max(y_values);
    ymin = np.min(y_values);
    y_range = ymax - ymin
    zmax = np.max(z_values);
    zmin = np.min(z_values);
    z_range = zmax - zmin

    x_sized_20 = [[xmin, 0.2 * x_range + xmin], [0.1 * x_range + xmin, 0.3 * x_range + xmin],
                  [0.2 * x_range + xmin, 0.4 * x_range + xmin],
                  [0.3 * x_range + xmin, 0.5 * x_range + xmin], [0.4 * x_range + xmin, 0.6 * x_range + xmin],
                  [0.5 * x_range + xmin, 0.7 * x_range + xmin], [0.6 * x_range + xmin, 0.8 * x_range + xmin],
                  [0.7 * x_range + xmin, 0.9 * x_range + xmin], [0.8 * x_range + xmin, x_range + xmin]]
    x_sized_30 = [[xmin, 0.3 * x_range + xmin], [0.15 * x_range + xmin, 0.45 * x_range + xmin],
                  [0.3 * x_range + xmin, 0.6 * x_range + xmin]
        , [0.45 * x_range + xmin, 0.75 * x_range + xmin], [0.6 * x_range + xmin, 0.9 * x_range + xmin],
                  [0.7 * x_range + xmin, x_range + xmin]]
    x_sized_40 = [[xmin, 0.4 * x_range + xmin], [0.2 * x_range + xmin, 0.6 * x_range + xmin],
                  [0.4 * x_range + xmin, 0.8 * x_range + xmin]
        , [0.6 * x_range + xmin, x_range + xmin]]
    x_sized_50 = [[xmin, 0.5 * x_range + xmin], [0.25 * x_range + xmin, 0.75 * x_range + xmin],
                  [0.5 * x_range + xmin, x_range + xmin]]

    x_sized_20_for_walls = [[xmin, 0.2 * x_range + xmin], [0.8 * x_range + xmin, x_range + xmin]]
    y_sized_20_for_walls = [[ymin, 0.2 * y_range + ymin], [0.8 * y_range + ymin, y_range + ymin]]

    all_slices = [x_sized_20, x_sized_30, x_sized_40, x_sized_50]
    wall_slices = [x_sized_20_for_walls, y_sized_20_for_walls]
    sized_slice_locs = all_slices

    output = pd.DataFrame()
    execution_path = os.getcwd()
    image_output_loc = execution_path + '\images_without_bb' + "\\"
    detected_object_data_all = []

    for k in range(len(sized_slice_locs)):
        print(str(k + 1) + " of " + str(len(sized_slice_locs)))
        df = pd.DataFrame(detected_object_data_all,
                          columns=['object_name', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'confidence', 'image_name'])
        print(df)
        for j in range(len(sized_slice_locs[k])):
            print(detected_object_data_all)
            something = (sized_slice_locs[k][j][1] - sized_slice_locs[k][j][0]) / x_range
            if something == 0.2:
                name = "x_sized_20"
            elif something == 0.3:
                name = "x_sized_30"
            elif something == 0.4:
                name = "x_sized_40"
            elif something == 0.5:
                name = "x_sized_50"
            print(str(j + 1) + " of " + str(len(sized_slice_locs[k])))
            sliced_pcd = pcd_np
            indexes = []
            print("\t slice location is at : " + str(sized_slice_locs[k][j]))
            for i in range(pcd_np.shape[0]):
                if sized_slice_locs[k][j][0] >= pcd_np[:, 0][i] or pcd_np[:, 0][i] >= sized_slice_locs[k][j][1]:
                    indexes.append(i)
            sliced_pcd = np.delete(sliced_pcd, indexes, axis=0)
            x_min_temp = np.min(sliced_pcd[:, 0]); x_max_temp = np.max(sliced_pcd[:, 0])
            x_range_temp = x_max_temp - x_min_temp

            image_name = name +"_"+ '%s' % str(j)
            print("\t " + str(pcd_np.shape));
            print("\t " + str(sliced_pcd.shape))
            height = 500
            width = 1000
            temp = np.zeros([height, width, 1], dtype=np.uint8)

            image_array = np.zeros([height, width, 3], dtype=np.uint8)  # empty image sized array
            temp = np.full((height, width, 1), xmin)  # used to check if the bin/pixel is already taken
            for i in range(sliced_pcd.shape[0]):  # iterating through all points of sliced pcd->(x,y,z,r,g,b)
                w = int(abs(sliced_pcd[:, 1][i] * (width / y_range)))  # mapping point's Y to image's width
                h = int(abs(sliced_pcd[:, 2][i] * (height / z_range)))  # mapping point's Z to image's height
                if temp[h][w][0] < (sliced_pcd[:, 0][i]):
                    image_array[h][w][0] = 255 * (sliced_pcd[:, 3][i])  # image's R
                    image_array[h][w][1] = 255 * (sliced_pcd[:, 4][i])  # image's G
                    image_array[h][w][2] = 255 * (sliced_pcd[:, 5][i])  # image's B
                    temp[h][w][0] = (sliced_pcd[:, 0][i])
                else:
                    continue
            img = Image.fromarray(image_array)
            img = img.rotate(180)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(image_output_loc + image_name + '.jpg')
            # img = cv.imread(image_output_loc + image_name + '.jpg')
            # img = cv.medianBlur(img, 3)
            # cv.imwrite(image_output_loc + image_name + '.jpg', img)

            objects_detected = image_ai_BB_generator_2.imageai(image_name)
            print("\t " + str(len(objects_detected)) + " objects detected in " + str(image_name + '.jpg'))
            count = 0
            np.savetxt("delete_this" + image_name + ".csv", sliced_pcd, delimiter=",")
            if len(objects_detected) != 0:
                for eachObject in objects_detected:
                    print("\t object_detected_is : " + eachObject["name"])
                    original_coords = actual_coords_from_BB_3.original_coords(eachObject["box_points"], sliced_pcd, height,
                                                                              width, y_range, z_range, image_name)
                    crop2mesh2replace_4.cropping(original_coords, image_name, count)
                    detected_object_data = original_coords
                    detected_object_data.insert(0, eachObject["name"])
                    detected_object_data.append(eachObject["percentage_probability"])
                    detected_object_data.append(image_name + str(count))
                    detected_object_data_all.append(detected_object_data)
                    output = output.append(detected_object_data, ignore_index=True)
                    count += 1
                    print("\t\t " + str(detected_object_data) + "\n")
            else:
                print("\t\t no objects in this one: " + image_name + ".jpg")

    df.to_csv('detected_data_all.csv', index=False)
    print("done")
    return None
