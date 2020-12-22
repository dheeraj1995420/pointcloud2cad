from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
image_input_loc = execution_path+'\images_without_bb'+"\\"
image_output_loc = execution_path+'\images_with_bb'+"\\"

detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()

detector.setModelPath("yolo.h5")

detector.loadModel()

# print("2 is running right now ################################")

# detections = detector.detectObjectsFromImage(input_image="sliced_x20_4.jpg", output_image_path="sliced_x20_4_dtctd.jpg")

custom = detector.CustomObjects(dining_table=True, chair=True, couch = True, potted_plant=True, laptop=True,
                                keyboard=True, tv=True, backpack=True, bottle=False, book=False)


def imageai(image_name):
    detections_custom = detector.detectCustomObjectsFromImage(custom_objects=custom,
                                                              input_image=os.path.join(image_input_loc,
                                                                                       image_name+'.jpg'),
                                                              output_image_path=os.path.join(image_output_loc,
                                                                                             image_name+'_bb_'+'.jpg'),
                                                              minimum_percentage_probability=50)

    # print(type(detections_custom))
    return detections_custom
    #
# for eachObject in detections_custom:
#     print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
#     print("--------------------------------")
