from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import glob
import argparse
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import pytesseract
from numpy import random
from cv2 import getPerspectiveTransform
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.plots import plot_one_box
import math
from numpy.linalg import norm
custom_config = r'--oem 3 --psm 6'

def get_boxes(details, threshold_point):
    total_boxes = len(details['text'])
    ls = []
    for sequence_number in range(total_boxes):

        if int(float(details['conf'][sequence_number])) > threshold_point:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                            details['width'][sequence_number], details['height'][sequence_number])
            ls.append((x, y, x + w, y + h))

    texts = details['text']

    texts = list(filter(('').__ne__, texts))

    refined_texts = []
    refined_boxes = []
    for l in range(len(ls)):
        if ls[l][2] - ls[l][0] < 5 or ls[l][3] - ls[l][1] < 5:
            pass
        else:
            refined_boxes.append(ls[l])
            text_extracted = texts[l]
            final_text = ''.join(filter(str.isdigit, text_extracted))
            refined_texts.append(final_text)

    return refined_boxes, refined_texts

def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

@csrf_exempt
def index(request):
    folder = 'static/input_img/'
    if request.method == "POST" and request.FILES['file']:
        file = request.FILES.get('file')
        input_img = glob.glob('static/input_img/*')
        for f in input_img:
            os.remove(f)
        result = glob.glob('static/result/*')
        for f in result:
            os.remove(f)

        txt_path = glob.glob('static/txt_file/*')
        for f in txt_path:
            os.remove(f)
        location = FileSystemStorage(location=folder)
        fn = location.save(file.name, file)
        path = os.path.join('static/input_img/', fn)
        flag = detect(path)
        # print(flag)
        if flag == None:
            folder = 'static/result/'
            location = FileSystemStorage(location=folder)
            fn = location.save(file.name, file)


        res = glob.glob('static/result/*')
        # print(res)
        flag2 = detect2(res[0])
        # print(flag2)

        if flag2 == None:


            context = {
                "status": "Warning! Send the valid image",
            }
        else:
            context = flag2




        return JsonResponse(context)
    else:
        context = {
            "status": "Warning! Send the valid image",
        }
    return JsonResponse(context)


weights1 = "static/weights/corner_points_95.pt"
# weights1 = "static/weights/four_corner_point_0.82.pt"
weights2 = "static/weights/cheque_boxes_address_amonut_serial_96.pt"
weights3 = "static/weights/micr_99.pt"
# Initialize
set_logging()
device = select_device('cpu')
half = device.type != 'cpu'  # half precision only supported on CUDA

model1 = attempt_load(weights1, map_location=device)  # load FP32 model1
model2 = attempt_load(weights2, map_location=device)  # load FP32 model2
model3 = attempt_load(weights3, map_location=device)  # load FP32 model1

names3 = model3.module.names if hasattr(model3, 'module') else model3.names
colors3 = [[random.randint(0, 255) for _ in range(3)] for _ in names3]


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)  # increases the white region in the image
    img = cv2.erode(img, kernel, iterations=1)  # erodes away the boundaries of foreground object
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img

def distance_formula(coor1, coor2):
    return math.sqrt((coor1[0]-coor2[0])**2+(coor1[1]-coor2[1])**2)

def take_first(tup):
    return tup[0]
def take_second(tup):
    return tup[1]

def sort_boxes(boxes, max_x):
    boxes_x_sort = sorted(boxes, key=take_first)
    full_boxes = []
    box_x_list = []
    for i in range(len(boxes_x_sort)-1):
        if i == len(boxes_x_sort)-2:
            if abs(boxes_x_sort[i][0] - boxes_x_sort[i + 1][0]) < max_x//2:
                box_x_list.append(boxes_x_sort[i])
                box_x_list.append(boxes_x_sort[i+1])
                full_boxes.append(box_x_list)
            else:
                box_x_list.append(boxes_x_sort[i])
                full_boxes.append(box_x_list)
                full_boxes.append([boxes_x_sort[i + 1]])
        else:
            if abs(boxes_x_sort[i][0]-boxes_x_sort[i+1][0]) < max_x//2:
                box_x_list.append(boxes_x_sort[i])
            else:
                box_x_list.append(boxes_x_sort[i])
                full_boxes.append(box_x_list)


                box_x_list=[]
    final_boxes = []
    for j in range(len(full_boxes)):
        boxes = sorted(full_boxes[j], key=take_second)

        final_boxes.append(boxes)

    return final_boxes




def detect(source):
    agnostic_nms = False
    augment = False
    classes = None
    conf_thres = 0.3
    # device = ''
    img_size = 640
    iou_thres = 0.45
    device = ''
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model1
    stride = int(model1.stride.max())  # model1 stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    if half:
        model1.half()  # to FP16
    # Second-stage classifier
    try:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Get names and colors
        names = model1.module.names if hasattr(model1, 'module') else model1.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if device.type != 'cpu':
            model1(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model1.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            # print(img)
            # print(img.shape)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model1(img, augment=augment)[0]
            # print(img)
            # print(pred)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

            try:
                for i, det in enumerate(pred):  # detections per image
                    # else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    nm = p.name.split('.')[0]
                    try:
                        os.mkdir(f'static/result/')
                    except:
                        pass
                    name = p.name.split('.')[0]
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    fresh_image = im0.copy()
                    h, w, _ = fresh_image.shape
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        coors = []
                        coner_flag = False
                        for *xyxy, conf, cls in det:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            lbl = label.split(' ')[0]
                            one_point = ((int(xyxy[0]) + int(xyxy[2])) // 2, (int(xyxy[1]) + int(xyxy[3])) // 2)
                            if len(coors) == 0:
                                coors.append(one_point)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            else:
                                valid = True
                                for c in coors:
                                    dist = distance_formula(c, one_point)
                                    if dist > w / 20:
                                        pass
                                    else:
                                        valid = False
                                        break
                                if valid == True:
                                    coors.append(one_point)
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            if len(coors) == 4:
                                break
                            # else:
                            #     cv2.imwrite(f'static/result/{p.name}', fresh_image)


                        if len(coors) != 4:
                            print('Corners not detected')
                            cv2.imwrite(f'static/result/{p.name}', fresh_image)
                        else:
                            dist_xs = []
                            dist_ys = []
                            for co in range(len(coors) - 1):
                                distance_x = abs(coors[co][0] - coors[co + 1][0])
                                distance_y = abs(coors[co][1] - coors[co + 1][1])
                                dist_xs.append(distance_x)
                                dist_ys.append(distance_y)
                            max_x = max(dist_xs)
                            max_y = max(dist_ys)

                            sorted_boxes = sort_boxes(coors, max_x)

                            final_boxes = []
                            for s in sorted_boxes:
                                final_boxes = final_boxes + s
                            indx = 1

                            src_pts = np.array([final_boxes[0], final_boxes[2], final_boxes[3], final_boxes[1]],
                                               dtype=np.float32)
                            dst_pts = np.array([[0, 0], [max_x, 0], [max_x, max_y], [0, max_y]], dtype=np.float32)

                            perspect = getPerspectiveTransform(src_pts, dst_pts)
                            final_image = cv2.warpPerspective(fresh_image, perspect, (max_x, max_y))
                            # print(final_image)
                            cv2.imwrite(f'static/result/{p.name}', final_image)
                            coner_flag = True
                        return coner_flag

            except:
                pass
    except:
        pass


def detect_for_micr(img0, model, imgsz,names):

    h,w,c = img0.shape
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16



    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    img = letterbox(img0, 640, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]

    pred = non_max_suppression(pred, 0.15, 0.45, classes=None, agnostic=False)
    lbls=[]
    boxes=[]
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            coor_list = []
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                # print(label)
                lbl=label.split(' ')[0]

                x1 = int(xyxy[0].item())

                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                if len(coor_list) == 0:
                    coor_list.append(x1)
                    boxes.append((x1, y1, x2, y2))
                    lbls.append(lbl)
                else:
                    flag_add = True
                    for co in coor_list:
                        if x1 > co+w//100 or x1 < co-w//100:
                            pass
                        else:
                            flag_add = False
                            break
                    if flag_add == True:
                        boxes.append((x1, y1, x2, y2))
                        lbls.append(lbl)
                        coor_list.append(x1)
    return lbls,boxes


def detect2(source):
    agnostic_nms = False
    augment = False
    classes = None
    conf_thres = 0.25
    # conf_thres = 0.50
    # device = ''
    img_size = 640
    # iou_thres = 0.45
    iou_thres = 0.45
    device = ''
    # Initialize
    set_logging()
    save_img=True
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model2
    stride = int(model2.stride.max())  # model2 stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    if half:
        model2.half()  # to FP16
    # Second-stage classifier
    try:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # Get names and colors
        names = model2.module.names if hasattr(model2, 'module') else model2.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if device.type != 'cpu':
            model2(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model2.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            # print(img.shape)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model2(img, augment=augment)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            for i, det in enumerate(pred):  # detections per image
                # else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                nm = p.name.split('.')[0]
                fresh_img = im0.copy()
                try:
                    os.mkdir(f'static/result/')
                except:
                    pass
                name = p.name.split('.')[0]
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                fresh_image = im0.copy()
                h, w, _ = fresh_image.shape
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    address_flag = False
                    serial_flag = False
                    ammount_flag = False
                    my_dict = {}
                    for *xyxy, conf, cls in det:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        lbl = label.split(' ')[0]
                        # print(label)
                        try:
                            if lbl == 'address' and address_flag == False:
                                x1 = int(xyxy[0].item())
                                y1 = int(xyxy[1].item())
                                x2 = int(xyxy[2].item())
                                y2 = int(xyxy[3].item())
                                address_img = fresh_img[y1 - 5 if y1 > 5 else y1:y2 + 5, x1 - 5 if x1 > 5 else x1:x2 + 5]
                                cv2.imwrite(f'static/all_results/{p.name.split(".")[0]}_address.jpg', address_img)
                                pro = preprocess(address_img)
                                txt = pytesseract.image_to_string(pro, config="--psm 6 --oem 3")
                                txt_s = txt.strip()
                                txt_s = txt_s.replace('\n', ' ')
                                print(txt_s)
                                address_flag = True


                                my_dict['address'] = txt_s
                        except:
                            pass

                        try:
                            if lbl == 'serial' and serial_flag == False:
                                serial_flag = True
                                x1 = int(xyxy[0].item())
                                y1 = int(xyxy[1].item())
                                x2 = int(xyxy[2].item())
                                y2 = int(xyxy[3].item())
                                serial_img = fresh_img[y1:y2, x1:x2]
                                cv2.imwrite(f'static/all_results/{p.name.split(".")[0]}_serial_orig.jpg', serial_img)
                                # serial_img = fresh_img[y1 - 15 if y1 > 15 else y1:y2 + 15, x1 - 15 if x1 > 15 else x1:x2 + 45]
                                fresh_img = serial_img.copy()
                                brightness_value = brightness(fresh_img)

                                labels, boxes = detect_for_micr(serial_img, model3, 640, names3)
                                for i in range(len(boxes)):
                                    box = boxes[i]
                                    label = labels[i]
                                    if label == 'a':
                                        color = [0, 0, 255]
                                    else:
                                        color = [255, 0, 0]

                                    cv2.rectangle(serial_img, (box[0], box[1]), (box[2], box[3]), color, 1)
                                    crop_img = fresh_img[box[1]:box[3], box[0]:box[2]]
                                    mask = np.zeros((crop_img.shape[0], crop_img.shape[1], crop_img.shape[2]), dtype=np.uint8)
                                    if brightness_value <= 140:
                                        mask.fill(127)
                                    elif brightness_value > 140 and brightness_value <= 160:
                                        mask.fill(190)
                                    else:
                                        mask.fill(255)

                                    fresh_img[box[1]:box[3], box[0]:box[2]] = mask

                                data = pytesseract.image_to_data(fresh_img, lang='mcr', output_type=pytesseract.Output.DICT,
                                                                 config=custom_config)
                                ls, texts = get_boxes(data, 0.8)
                                for l in ls:
                                    # print(l[0], l[1]), (l[2], l[3])
                                    # if l[0] >=2 and l[1] >=2 or l[2] >=2 and l[3] >=2:
                                    #     pass
                                    # else:
                                    cv2.rectangle(fresh_img, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 1)

                                    cv2.rectangle(serial_img, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 1)
                                cv2.imwrite(f'static/all_results/{p.name.split(".")[0]}_serial_processed.jpg', serial_img)

                                final_list_boxes = ls + boxes
                                final_list_texts = texts + labels
                                coor_list = []
                                for f in final_list_boxes:
                                    coor_list.append(f[0])
                                final_list_texts_sorted = [x for _, x in sorted(zip(coor_list, final_list_texts))]
                                final_micr_number = f""
                                for text in final_list_texts_sorted:
                                    final_micr_number += text


                                my_dict['serial'] = final_micr_number

                        except:
                            pass


                        if lbl == 'amount' and ammount_flag == False:
                            pass
                        # print(my_dict)
                    return my_dict
    except:
        pass
