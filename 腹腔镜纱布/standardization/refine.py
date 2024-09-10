import json,cv2,os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def is_mask_in_rectangle(rect_top_left, rect_bottom_right, mask_points):
    # 矩形区域的坐标
    [rect_left, rect_top] = rect_top_left
    [rect_right, rect_bottom] = rect_bottom_right
    flag = 0
    # 检查每个 mask 点是否在矩形区域内
    for point in mask_points:
        x, y = point
        if not (rect_left <= x <= rect_right and rect_bottom <= y <= rect_top):
            flag += 1

    print((len(mask_points) - flag)/len(mask_points))
    if (len(mask_points) - flag)/len(mask_points) >= 0.7:
        return True  # 所有点都在矩形内
    else:
        return False


def apply_mask_to_image(original_image, mask):
    # 确保 mask 是二值图像（0 和 255）
    mask = mask.astype(bool)  # 将掩码转换为布尔类型
    masked_image = np.zeros_like(original_image)  # 创建与原图相同大小的空图像

    # 保留原图中的 mask 区域
    masked_image[mask] = original_image[mask]
    
    return masked_image

def get_labels(image_path, rectangle_points, mask_points, o_points):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    img_array = np.array(image)
    x_min, y_max = min(rectangle_points[0][0], rectangle_points[1][0]), max(rectangle_points[0][1],rectangle_points[1][1])
    x_max, y_min = max(rectangle_points[0][0], rectangle_points[1][0]), min(rectangle_points[0][1],rectangle_points[1][1])
    
    rect_top_left = [x_min, y_max]
    rect_bottom_right = [x_max, y_min]
    mask = np.zeros([height, width], np.uint8)
    rectangle = np.zeros([height, width], np.uint8)
    rec_points = np.array([rect_top_left, rect_bottom_right], dtype=np.int32)
    rectangle = cv2.rectangle(rectangle, rec_points[0], rec_points[1], 255, thickness=cv2.FILLED)
    
    for point in mask_points:
        # print(is_mask_in_rectangle(rect_top_left, rect_bottom_right, point))
        if is_mask_in_rectangle(rect_top_left, rect_bottom_right, point):
            point_array = np.array(point, dtype=np.int32)
            mask = cv2.fillPoly(mask, [point_array], 255)
    
    if o_points != []:
        for point in o_points:
            # print(is_mask_in_rectangle(rect_top_left, rect_bottom_right, point))
            if is_mask_in_rectangle(rect_top_left, rect_bottom_right, point):
                point_array = np.array(point, dtype=np.int32)
                mask = cv2.fillPoly(mask, [point_array], 0)
    

    mask_area = rectangle & mask 
    rest_area = np.where((rectangle == 255) & (mask == 0), 255, 0).astype(np.uint8)

    return img_array, mask_area, rest_area, [rect_top_left, rect_bottom_right]
   

def Calculate_pixel_difference(img_array, mask_array, rest_array):
    mask_img = apply_mask_to_image(img_array, mask_array)
    rest_img = apply_mask_to_image(img_array, rest_array)
    mask_pixel = np.sum(mask_img[mask_img != 0]) / np.count_nonzero(mask_img) 
    rest_pixel = np.sum(rest_img[rest_img != 0]) / np.count_nonzero(rest_img) 
    pixel_difference = abs(mask_pixel - rest_pixel)

    return pixel_difference


def get_rectangle_id(json_path):
    print(json_path)
    with open(json_path, encoding='UTF-8') as f:
        content = json.load(f)
    
    # with open(new_json_path, 'w', newline='', encoding='UTF-8') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow( ["file","group_id", "pixel_difference","normalized_pixel", "group", "mask_aera", "rest_area","rec_area"])

    
    shapes = content['shapes']
    mask_points = []
    o_points = []
    idx_list = []
    for idx, shape in enumerate(shapes):
        shape_type = shape["shape_type"]
        if shape_type == "rectangle":
            idx_list.append(idx)
        else:
            point = shape["points"]
            if shape["label"] == "0":
                o_points.append(point)
            else:
                mask_points.append(point)
    
    img_path = json_path.replace(".json",".png").replace("json","images")
    for i,idx in enumerate(idx_list):
        img_array, mask_area, rest_area,rec_area = get_labels(img_path, shapes[idx]["points"], mask_points,o_points)
        pixel_difference = Calculate_pixel_difference(img_array, mask_area, rest_area)
        content["shapes"][idx]["group_id"] = i
        content["shapes"][idx]["pixel_difference"] = pixel_difference
        print(pixel_difference)
        if 0<= pixel_difference < 17:
            group = "difficult"
        elif 17<= pixel_difference  < 46.6:
            group = "normal"
        elif 46.6<= pixel_difference <=255:
            group = "easy"

        image_id = img_path.split("/")[-1].split("\\")[-1]
        new_json_path = "standardization/data/json/{}_{}.json".format(image_id.split(".")[0], group)
        new_file_coentent = {"image_id": image_id, "group_id":i, "pixel_difference": pixel_difference,
                             "normalized_pixel": pixel_difference/255, "mask_aera": mask_area.tolist(), "rest_area": rest_area.tolist(),"rec_area": rec_area}
            # writer.writerow([image_id, i, pixel_difference, pixel_difference/255, group, str(mask_area.tolist()),str(rest_area.tolist()), str(rec_area)])

        if os.path.exists(new_json_path):
            with open(new_json_path, 'r', encoding='UTF-8') as file:
                source_coentent = json.load(file)
        else:
                source_coentent = []
        
        source_coentent.append(new_file_coentent)
        with open(new_json_path,"w", encoding='UTF-8') as file:
            json.dump(source_coentent, file, ensure_ascii=False)
    
    with open(json_path, 'w', encoding='UTF-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)