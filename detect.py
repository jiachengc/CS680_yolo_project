import configparser
import time
from pathlib import Path
import cv2
import torch
import threading
import sys
import multiprocessing as mp
sys.path.append("yolov5")
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from concurrent.futures import ThreadPoolExecutor


Detect_path = 'D:/Data/detect_outputs'  # 检测图片输出路径


def detect(path, model_path, detect_size):
    source = path
    weights = model_path
    imgsz = detect_size
    conf_thres = 0.25
    iou_thres = 0.45
    device = ""
    augment = True
    save_img = True
    save_dir = Path(Detect_path)  # increment run

    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_sizef
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    result_list = []

    for path, img, im0s, vid_cap in dataset:
        # 读取图片传到gpu上
        t1 = time.time()
        img = torch.from_numpy(img).to(device)
        print("read pictures cost time：", time.time() - t1)
        t2 = time.time()
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print("process pictures cost time：", time.time() - t2)
        # Inference
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            # print(s)  # 384x640
            s_result = ''  # 输出检测结果
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    s += f"{n} {names[int(c)]}, "  # add to string
                    s_result += f"{n} {names[int(c)]} "
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:
                        c = int(cls)
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        label = f'{names[int(cls)]}'
                        # print(label)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # print(xyxy)

            print(f'{s}')
            # print(f'{s_result}')
            result_list.append(s_result)

            # 将conf对象中的数据写入到文件中
            conf = configparser.ConfigParser()
            cfg_file = open("glovar.cfg", 'w')
            conf.add_section("default")  # 在配置文件中增加一个段
            # 第一个参数是段名，第二个参数是选项名，第三个参数是选项对应的值
            conf.set("default", "process", str(dataset.img_count))
            conf.set("default", "total", str(dataset.nf))
            conf.write(cfg_file)
            cfg_file.close()
            
            im0 = annotator.result()
            # Save results (image with detections)
            t3 = time.time()
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            print("write pictures cost time：", time.time() - t3)
    print('Done')


def run(path, model_path, detect_size):
    with torch.no_grad():
        detect(path, model_path, detect_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    s_t = time.time()
    path1 = "D:/Data/image"
    path2 = "D:/Data/image2"
    path3 = "D:/Data/image3"
    model_path = "../weights/best.pt"
    detect_size = 1920
    p1 = mp.Process(target=run, args=(path1, model_path, detect_size,))
    p2 = mp.Process(target=run, args=(path2, model_path, detect_size,))
    p3 = mp.Process(target=run, args=(path3, model_path, detect_size,))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    print("Tatal Cost Time:", time.time() - s_t)