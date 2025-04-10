#import libraries
import os
import shutil
import cv2
from pathlib import Path
def morph(source, goal, destination):


    #reading images and resizing it
    image_source = cv2.imread(source)
    image_source_name = Path(source).stem
    cv2.imwrite(destination + image_source_name + "00.tif", image_source)

    image_goal = cv2.imread(goal)
    image_goal_name = Path(goal).stem
    # cv2.imwrite(destination + image_goal_name + "00.tif", image_goal)
    # image_goal = cv2.resize(image_goal)

    #morphing in 100 steps
    for i in range(34, 100, 33):
        percentage = float(i / 100)
        print(percentage)
        imgAdd = cv2.addWeighted(image_source, 1 - percentage, image_goal, percentage, 0)
        # cv2.imshow("Morphed", imgAdd)
        # cv2.waitKey(100)
        cv2.imwrite(f"{destination}{image_source_name}{i}.tif", imgAdd)
    #waiting till 0 is pressed
    # cv2.waitKey(0)
    #closing all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # source = "../../montreal_forest_data/nice_cut/tiny/0528.tif"
    # goal = "../../montreal_forest_data/nice_cut/tiny/0617.tif"
    # destination = "../../montreal_forest_data/nice_cut/morph/"
    # morph(source, goal, destination)

    input_folder = "../../montreal_forest_data/nice_cut/realigned/"
    destination_folder = "../../montreal_forest_data/nice_cut/realigned_morph/"

    # input_folder = "../../montreal_forest_data/deadtrees1_warped/2022"
    # destination_folder = "../../montreal_forest_data/deadtrees1_morph/2022"
    input_paths = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    input_paths.sort()
    for i in range(len(input_paths)-1):
        morph(input_folder + input_paths[i], input_folder + input_paths[i+1], destination_folder)
    shutil.copy(input_folder + input_paths[0], destination_folder + input_paths[0].replace('.tif', '00.tif'))
    shutil.copy(input_folder + input_paths[-1], destination_folder + input_paths[-1].replace('.tif', '00.tif'))
