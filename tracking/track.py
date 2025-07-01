# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

# 创建一个简单的类来包装检测结果，使其与 ByteTracker 兼容
class DetectionWrapper:
    def __init__(self, det):
        """
        包装检测结果，使其与 ByteTracker 兼容
        
        Args:
            det: numpy array with shape (n, 6), where each row is [x, y, w, h, conf, cls]
        """
        if det.shape[1] >= 6:
            self.xywh = det[:, :4]  # 前4列是 x, y, w, h
            self.conf = det[:, 4]   # 第5列是置信度
            self.cls = det[:, 5]    # 第6列是类别
        else:
            # 如果检测结果格式不符合预期，创建空数组
            self.xywh = np.array([])
            self.conf = np.array([])
            self.cls = np.array([])
            print("Warning: Detection format is not as expected. Expected (n, 6+), got", det.shape)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    
    # Check if custom_args exists
    if not hasattr(predictor, 'custom_args'):
        print("Warning: predictor.custom_args not found. Skipping tracker initialization.")
        return
    
    # Check if tracking_method exists in custom_args
    if not hasattr(predictor.custom_args, 'tracking_method'):
        print("Warning: predictor.custom_args.tracking_method not found. Skipping tracker initialization.")
        return

    try:
        assert predictor.custom_args.tracking_method in TRACKERS, \
            f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

        tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
        trackers = []
        for i in range(predictor.dataset.bs):
            tracker = create_tracker(
                predictor.custom_args.tracking_method,
                tracking_config,
                predictor.custom_args.reid_model,
                predictor.device,
                predictor.custom_args.half,
                predictor.custom_args.per_class
            )
            # motion only modeles do not have
            if hasattr(tracker, 'model'):
                tracker.model.warmup()
            trackers.append(tracker)

        predictor.trackers = trackers
    except Exception as e:
        print(f"Error initializing trackers: {e}")
        predictor.trackers = []


# 自定义的 on_predict_postprocess_end 回调函数，用于替换 ultralytics 中的默认回调函数
def on_predict_postprocess_end(predictor):
    """
    在预测后处理结束时调用，用于更新跟踪器
    
    Args:
        predictor: 预测器对象
    """
    if not hasattr(predictor, 'trackers') or not predictor.trackers:
        print("Warning: No trackers found. Skipping tracking.")
        return
    
    is_stream = predictor.dataset.mode == "stream"
    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        
        # 获取检测结果
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            # 将检测结果转换为 numpy 数组
            det = result.boxes.data.cpu().numpy()
            if len(det) == 0:
                continue
                
            # 使用 DetectionWrapper 包装检测结果
            det_wrapper = DetectionWrapper(det)
            
            try:
                # 更新跟踪器
                tracks = tracker.update(det_wrapper, result.orig_img)
                if len(tracks) == 0:
                    continue
                    
                # 更新结果
                idx = tracks[:, -1].astype(int)
                predictor.results[i] = result[idx]
                
                # 更新边界框
                update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
                predictor.results[i].update(**update_args)
            except Exception as e:
                print(f"Error updating tracker: {e}")

@torch.no_grad()
def run(args):

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )

    # Store custom args in predictor before tracking initialization
    if yolo.predictor is not None:
        yolo.predictor.custom_args = args

    # Register callback before tracking
    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    
    # 替换默认的 on_predict_postprocess_end 回调函数
    yolo.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if not is_ultralytics_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback(
                "on_predict_batch_start",
                lambda p: yolo_model.update_im_paths(p)
            )
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    # Run tracking
    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )
    
    # 初始化视频写入器
    video_writer = None
    
    # 创建保存路径（如果需要保存视频）
    if args.save_vid:
        source_path = Path(args.source) if not args.source.isnumeric() else Path(f"webcam_{args.source}")
        source_name = source_path.stem
        save_path = Path(args.project) / args.name / f"{source_name}_tracked.mp4"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"视频将保存到: {save_path}")
    
    # 帧计数器
    frame_count = 0

    for r in results:
        # 增强可视化效果，显示跟踪框和ID
        img = r.orig_img.copy()
        
        # 使用跟踪器绘制跟踪结果
        if (args.show_track_id or args.save_vid) and hasattr(yolo.predictor, 'trackers') and len(yolo.predictor.trackers) > 0:
            try:
                # 获取当前帧的检测结果
                if hasattr(r, 'boxes') and len(r.boxes) > 0:
                    # 绘制检测框和ID
                    for box in r.boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # 获取类别和置信度
                        cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0
                        # 获取ID（如果有）
                        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
                        
                        # 绘制边界框
                        color = (0, 255, 0)  # 绿色
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # 如果有ID，则显示ID
                        if track_id != -1:
                            label = f'ID:{track_id} C:{cls} {conf:.2f}'
                            # 计算文本大小
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                            # 绘制文本背景
                            cv2.rectangle(img, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
                            # 绘制文本
                            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            except Exception as e:
                print(f"Error plotting results: {e}")
        
        # 添加帧计数信息
        if args.show_frame_count:
            cv2.putText(img, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 初始化视频写入器（仅在第一帧时）
        if args.save_vid and video_writer is None and img is not None:
            h, w = img.shape[:2]
            video_writer = cv2.VideoWriter(
                str(save_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                args.fps,  # 使用指定的帧率
                (w, h)
            )
        
        # 保存帧到视频
        if args.save_vid and video_writer is not None:
            video_writer.write(img)
        
        # 显示处理后的帧
        if args.show:
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
        
        frame_count += 1
    
    # 释放视频写入器
    if video_writer is not None:
        video_writer.release()
        print(f"视频已保存到: {save_path}")
    
    # 关闭所有窗口
    cv2.destroyAllWindows()


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--save-vid', action='store_true',
                        help='save video with tracking visualization')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for saved video')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show trajectories')
    parser.add_argument('--show-track-id', action='store_true',
                        help='show track IDs and bounding boxes')
    parser.add_argument('--show-frame-count', action='store_true',
                        help='show frame count on video')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
