import open3d as o3d
import numpy as np
import pandas as pd
import math
import os
from PIL import Image
import csv
import image_ai_BB_generator_2
import actual_coords_from_BB_3
import crop2mesh2replace_4
from matplotlib import pyplot as plt
from numpy import loadtxt

pcd = o3d.io.read_point_cloud("full_buro_10m.pcd")
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

wall_slices = [[xmin, 0.2 * x_range + xmin],[0.8 * x_range + xmin, x_range + xmin], [ymin, 0.2 * y_range + ymin],[0.8 * y_range + ymin, y_range + ymin]]

output = pd.DataFrame()
execution_path = os.getcwd()
image_output_loc = execution_path + '\images_without_bb' + "\\"
detected_object_data_all = []

for j in range(len(wall_slices)):
    print(str(j + 1) + " of " + str(len(wall_slices)))
    sliced_pcd = pcd_np
    indexes = []
    print("\t slice location is at : " + str(wall_slices[j]))
    if j == 0 or j == 1:
        image_name = 'x_20_' + '%s' % str(j)
        for i in range(pcd_np.shape[0]):
            if wall_slices[j][0] >= pcd_np[:, 0][i] or pcd_np[:, 0][i] >= wall_slices[j][1]:
                indexes.append(i)

    else:
        image_name = 'y_20_' + '%s' % str(j)
        for i in range(pcd_np.shape[0]):
            if wall_slices[j][0] >= pcd_np[:, 1][i] or pcd_np[:, 1][i] >= wall_slices[j][1]:
                indexes.append(i)

    sliced_pcd = np.delete(sliced_pcd, indexes, axis=0)
    print("\t " + str(pcd_np.shape));
    print("\t " + str(sliced_pcd.shape))
    height = int(500)
    # width = 1000
    if j == 0 or j == 1:
        aspect_ratio = (y_range) / (z_range)
    else:
        aspect_ratio = x_range/z_range
    width = int(aspect_ratio * height)
    temp = np.zeros([height, width, 1], dtype=np.uint8)
    image_array = np.zeros([height, width, 3], dtype=np.uint8)  # empty image sized array
    if j == 0 or j == 1:
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
    else:
        temp = np.full((height, width, 1), ymin)  # used to check if the bin/pixel is already taken
        for i in range(sliced_pcd.shape[0]):  # iterating through all points of sliced pcd->(x,y,z,r,g,b)
            w = int(abs(sliced_pcd[:, 0][i] * (width / x_range)))  # mapping point's Y to image's width
            h = int(abs(sliced_pcd[:, 2][i] * (height / z_range)))  # mapping point's Z to image's height
            if temp[h][w][0] < (sliced_pcd[:, 1][i]):
                image_array[h][w][0] = 255 * (sliced_pcd[:, 3][i])  # image's R
                image_array[h][w][1] = 255 * (sliced_pcd[:, 4][i])  # image's G
                image_array[h][w][2] = 255 * (sliced_pcd[:, 5][i])  # image's B
                temp[h][w][0] = (sliced_pcd[:, 1][i])
            else:
                continue

    img = Image.fromarray(image_array)
    img = img.rotate(180)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(image_output_loc + image_name + '.jpg')
    objects_detected = image_ai_BB_generator_2.imageai(image_name)###################################################################################
    print("\t " + str(len(objects_detected)) + " objects detected in " + str(image_name + '.jpg'))
    count = 0
    np.savetxt("delete_this" + image_name + ".csv", sliced_pcd, delimiter=",")
    if len(objects_detected) != 0:
        for eachObject in objects_detected:
            # print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"], ":", image_name)
            print("\t object_detected_is : " + eachObject["name"])
            original_coords = actual_coords_from_BB_3.original_coords(eachObject["box_points"], sliced_pcd, height,
                                                                      width,x_range, y_range, z_range, image_name)#####################################################
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
    # image name =  sliced_x20_'+'%s'%str(j)+'.jpg
    # Object_name, %, BB  = imageAI(''image name') #where BB = (h1,w1,h2,w2)
    ##object_detector(its a list of dictionaries) = imageai(image_name) #image name should be without .jpg
    # x1 y1 z1 x2 y2 z2 = actual_coords(temp_pcd_np, BB,'image name' )
    # json_file = create.json(x1 y1 z1 x2 y2 z2) #use z as orthogonal axis, with respective max and min
    # Cropped = cropped_pcd(json_file, pcd)
    # save cropped pcd in /output with name "objectname_x(20-50)_(1-9)_(1-m)"
df = pd.DataFrame(detected_object_data_all,
                  columns=['object_name', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'confidence', 'image_name'])
print(df)
df.to_csv('detected_data_all.csv', index=False)
# with open("detected_object_data_x_40.csv", 'w', newline='') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     for i, value in enumerate(detected_object_data_all): wr.writerow(value)
# np.savetxt("detected_object_data_"+image_name+".csv", detected_object_data_all, delimiter=",")
print("done")

