import open3d as o3d
import numpy as np
import pandas as pd
import json
import os

# print("4 is running right now ################################")

current_directory_loc = os.getcwd()
pcd = o3d.io.read_point_cloud("full_buro_10m.pcd")
def json_creator(list_cuboid, image_name,i, axis_max, axis_min):
    # list_cuboid =list(np_cuboid)
    dump_this_dict = {
        "axis_max": axis_max,
        "axis_min": axis_min,
        "bounding_polygon": list_cuboid,
        "class_name" : "SelectionPolygonVolume",
        "orthogonal__axis": "Z",
        "version_major": 1,
        "version_minor": 0
     }
    json_object = json.dumps(dump_this_dict, indent = 4)
    json_write_location = current_directory_loc + "\JSONs" + "\\" + image_name + "_" + str(i) + ".json"
    with open(json_write_location, "w") as outfile:
        outfile.write(json_object)
    return None

def cropping(list_corners, image_name,i):
    np_cuboid=bb(list_corners)
    axis_max = max(np_cuboid[:,2])
    axis_min = min(np_cuboid[:,2])
    list_cuboid=np_cuboid.tolist()
    json_creator(list_cuboid, image_name,i,axis_max, axis_min)
    json_location = current_directory_loc + "\JSONs" + "\\" + image_name + "_" + str(i) + '.json'
    vol = o3d.visualization.read_selection_polygon_volume(json_location)
    object = vol.crop_point_cloud(pcd)
    cropped_pcd_location = current_directory_loc+"\cropped_pcds"+"\\"+image_name+"_"+str(i)+".pcd"
    # o3d.io.write_point_cloud(cropped_pcd_location, object)


def bb(list):
    x1 = list[0]; y1 = list[1]; z1 = list[2]; x2 = list[3]; y2 = list[3]; z2 = list[3];
    cuboid = np.array([[x1, y1, z1], [x2, y1, z1],
                       [x1, y2, z1], [x1, y1, z2],
                       [x2, y2, z1], [x1, y2, z2],
                       [x2, y1, z2], [x2, y2, z2],])
    return cuboid


