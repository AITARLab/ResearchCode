import json,os,shutil,cv2
import numpy as np

def single_info(json_file):
    with open(json_file, encoding="UTF-8") as f:
        content = json.load(f)

    txt_path = json_file.replace(".json",".txt").replace("json","labels")
    line_mask = []
    for info in content:
        [[x_min, y_max], [x_max, y_min]] = info["rec_area"]
        mask = np.array(info["mask_aera"])
        line_mask.append(mask)
        y,x= mask.shape
        x_center = (x_max+x_min)/x/2
        y_center = (y_max+y_min)/y/2
        nor_width = (x_max-x_min)/x
        nor_height = (y_max-y_min)/y
        content = '0 '+str(x_center)+' '+str(y_center)+' '+str(nor_width)+' '+str(nor_height)+'\n'
        with open(txt_path, 'a') as f:
                f.write(content)
    
    if len(line_mask) >1:
        result_matrix = line_mask[0]
        for matrix in line_mask:
            result_matrix = np.bitwise_or(result_matrix, matrix)
    
    return mask


def new_mask_and_label(file_list):
    for json_path in file_list:
        img_id_type = json_path.split("/")[-1].split("\\")[-1].split(".")[0].split("_")
        img_id = img_id_type[0] + "_" + img_id_type[1]
        type = img_id_type[2]
        mask_path = f"standardization/data/mask/{img_id}_{type}.png"
        mask = single_info(json_path)
        # cv2.imwrite(mask_path, mask)
        

def new_images(json_file_list, source_image_name):
    target_images_list = [json_path.replace(".json",".png").replace("json","images") for json_path in json_file_list]
    mask_images_list = [img.replace("images","mask") for img in target_images_list]
    source_image_path = f"data/images/{source_image_name}.png"
    # source_img = cv2.imread(source_image_path)

    assert len(target_images_list) == len(mask_images_list)
    for i,target_path in enumerate(target_images_list):
        target_img = cv2.imread(source_image_path)
        cv2.imwrite("test.png", target_img)
        for j,mask_path in enumerate(mask_images_list):
            if i != j:
                print(i,j)
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                target_img[mask_img == 255] = 0
        cv2.imwrite(target_path, target_img)
        
        

def new_test_generation(txt_file):
    type_list = ["easy", "normal","difficult"]
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        img_name = line.strip().split(".")[0]
        print(img_name)
        flag_list = []
        for ti in type_list:
            json_file = f"standardization/data/json/{img_name}_{ti}.json"
            if os.path.exists(json_file):
                flag_list.append(json_file)
                
        
        if flag_list == []:
            continue
        else:
            new_mask_and_label(flag_list)
            new_images(flag_list, img_name)
        
        
def test_log_refine(source_log):
    with open(source_log,"r", encoding="UTF-8") as f:
        lines = f.readlines()

    videos = [line.split("_")[0] for line in lines]
    log_name = source_log.split("/")[-1].split(".")[0]

    root = "standardization/data/images"
    for file in os.listdir(root):
        [vid, idx ,type] = file.split(".")[0].split("_")
        # print(file)
        # print(vid)
        if vid in videos:
            # print(vid)
            log_file = f"yaml/{log_name}_{type}.txt"
            with open(log_file, 'a') as f:
                f.write(f"{root}/{file}\n")
            