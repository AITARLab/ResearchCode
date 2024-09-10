import json

def get_json(yolo_json,iou):
    with open(yolo_json,"r",encoding='UTF-8') as f:
        content = json.load(f)
    
    # print(len(content))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total = 123
    for info in content:
        if info["score"] >= iou:
            tp += 1
        else:
            fp += 1
    
    fn = total - tp
    print(tp)
    print("precsion {}\trecall {}".format(tp/len(content), tp/total))

