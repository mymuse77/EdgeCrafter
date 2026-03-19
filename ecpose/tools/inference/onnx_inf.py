"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Pose ONNX inference script.
"""

import concurrent.futures
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


@dataclass
class Result:
    label: int
    score: float
    keypoints: np.ndarray


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# COCO keypoint skeleton (1-based in standard definition)
COCO_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]
COCO_SKELETON = [(a - 1, b - 1) for a, b in COCO_SKELETON]


def _decode_keypoints(keypoints: np.ndarray):
    """Decode keypoints into xy and visibility arrays.

    Supports: [K,2], [K,3], [2K], [3K].
    """
    kpts = np.asarray(keypoints)

    if kpts.ndim == 2:
        if kpts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        xy = kpts[:, :2].astype(np.float32)
        vis = kpts[:, 2].astype(np.float32) if kpts.shape[1] >= 3 else np.ones((kpts.shape[0],), dtype=np.float32)
        return xy, vis

    if kpts.ndim == 1:
        if kpts.size % 3 == 0:
            kpts_ = kpts.reshape(-1, 3)
            return kpts_[:, :2].astype(np.float32), kpts_[:, 2].astype(np.float32)
        if kpts.size % 2 == 0:
            kpts_ = kpts.reshape(-1, 2)
            return kpts_.astype(np.float32), np.ones((kpts_.shape[0],), dtype=np.float32)

    return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def draw_pose(image: Image.Image, results, radius=3, line_thickness=2, draw_skeleton=True):
    im_np = np.array(image).copy()

    for res in results:
        kpts, vis = _decode_keypoints(res.keypoints)
        if kpts.shape[0] == 0:
            continue

        kpts = kpts.astype(np.int32)
        vis = np.isfinite(vis) & (vis > 0)

        for idx, (x, y) in enumerate(kpts):
            if vis[idx]:
                cv2.circle(im_np, (int(x), int(y)), radius, (0, 255, 0), -1)

        if draw_skeleton:
            for a, b in COCO_SKELETON:
                if a < len(kpts) and b < len(kpts) and vis[a] and vis[b]:
                    xa, ya = kpts[a]
                    xb, yb = kpts[b]
                    cv2.line(im_np, (int(xa), int(ya)), (int(xb), int(yb)), (255, 128, 0), line_thickness)

        min_xy = np.min(kpts[vis], axis=0) if np.any(vis) else np.min(kpts, axis=0)
        min_xy = np.maximum(min_xy, 0)
        text = f"{res.label} {res.score:.2f}"
        cv2.putText(
            im_np,
            text,
            (int(min_xy[0]), int(max(min_xy[1] - 5, 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return Image.fromarray(im_np.astype(np.uint8))


class ECPoseOnnxInferencer:
    def __init__(self, session, size, thresh):
        self.session = session
        self.size = size
        self.thresh = thresh
        self.transforms = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _parse_outputs(self, outputs):
        # Prefer explicit output names from export_onnx.py
        mapping = {name: arr for name, arr in zip(self.output_names, outputs)}

        if {'scores', 'labels', 'keypoints'}.issubset(mapping.keys()):
            scores = mapping['scores']
            labels = mapping['labels']
            keypoints = mapping['keypoints']
        elif len(outputs) == 3:
            # Fallback to expected order from export_onnx.py
            scores, labels, keypoints = outputs
        else:
            raise RuntimeError(
                f'Unexpected ONNX outputs. names={self.output_names}, count={len(outputs)}; expected scores/labels/keypoints'
            )

        return scores, labels, keypoints

    def infer_batch(self, images):
        batch_results = []
        for img in images:
            w, h = img.size
            orig_size = np.array([[w, h]], dtype=np.int64)
            tensor = self.transforms(img).unsqueeze(0).numpy()

            outputs = self.session.run(
                output_names=None,
                input_feed={'images': tensor, 'orig_target_sizes': orig_size},
            )

            scores, labels, keypoints = self._parse_outputs(outputs)
            scores = scores[0]
            labels = labels[0]
            keypoints = keypoints[0]

            keep = scores > self.thresh
            results = []
            for j in np.where(keep)[0]:
                results.append(Result(
                    label=int(labels[j]),
                    score=float(scores[j]),
                    keypoints=keypoints[j],
                ))

            batch_results.append(results)

        return batch_results

    def infer(self, image):
        return self.infer_batch([image])[0]


class VideoReader(threading.Thread):
    def __init__(self, cap, queue_size=32):
        super().__init__()
        self.cap = cap
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.daemon = True

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        while not self.q.empty():
            self.q.get()


def process_image(inferencer, path: Path, radius=3, line_thickness=2, draw_skeleton=True):
    image = Image.open(path).convert('RGB')
    results = inferencer.infer(image)
    image = draw_pose(image, results, radius=radius, line_thickness=line_thickness, draw_skeleton=draw_skeleton)

    output_path = path.with_stem(f"{path.stem}_onnx_pose_inference")
    image.save(output_path, quality=95, subsampling=0)
    print(f"Saved result to {output_path}")
    print(f"Detected {len(results)} poses")


def process_video(inferencer, path: Path, batch_size=8, num_workers=4, radius=3, line_thickness=2, draw_skeleton=True):
    cap = cv2.VideoCapture(str(path))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = path.with_stem(f"{path.stem}_onnx_pose_inference")
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    reader = VideoReader(cap)
    reader.start()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    futures_queue = queue.Queue()

    def process_and_draw(pil_img, results):
        res_img = draw_pose(
            pil_img,
            results,
            radius=radius,
            line_thickness=line_thickness,
            draw_skeleton=draw_skeleton,
        )
        return cv2.cvtColor(np.array(res_img), cv2.COLOR_RGB2BGR)

    frame_count = 0

    def writer_worker():
        nonlocal frame_count
        while True:
            future = futures_queue.get()
            if future is None:
                break
            frame_out = future.result()
            out.write(frame_out)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frame} frames")

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    buffer_pil = []
    while True:
        frame = reader.read()
        if frame is None:
            break
        buffer_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if len(buffer_pil) == batch_size:
            results_batch = inferencer.infer_batch(buffer_pil)
            for pil_img, results in zip(buffer_pil, results_batch):
                futures_queue.put(executor.submit(process_and_draw, pil_img, results))
            buffer_pil = []

    if buffer_pil:
        results_batch = inferencer.infer_batch(buffer_pil)
        for pil_img, results in zip(buffer_pil, results_batch):
            futures_queue.put(executor.submit(process_and_draw, pil_img, results))

    reader.stop()
    futures_queue.put(None)
    writer_thread.join()
    executor.shutdown()
    cap.release()
    out.release()
    print(f"Saved video result to {output_path}")


def main(args):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.onnx, sess_options, providers=providers)

    input_shape = session.get_inputs()[0].shape
    if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        img_size = (input_shape[2], input_shape[3])
    else:
        img_size = (640, 640)

    print(f"Using device: {args.device}")
    print(f"ONNX Runtime device: {ort.get_device()}")
    print(f"Model input size: {img_size}")

    inferencer = ECPoseOnnxInferencer(
        session=session,
        size=img_size,
        thresh=args.thresh,
    )

    input_path = Path(args.input)
    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        process_image(
            inferencer,
            input_path,
            radius=args.kpt_radius,
            line_thickness=args.kpt_line_thickness,
            draw_skeleton=not args.no_skeleton,
        )
    else:
        process_video(
            inferencer,
            input_path,
            batch_size=args.batch_size,
            radius=args.kpt_radius,
            line_thickness=args.kpt_line_thickness,
            draw_skeleton=not args.no_skeleton,
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ECPose ONNX Inference')
    parser.add_argument('--onnx', '-o', required=True, help='Path to ONNX model file')
    parser.add_argument('--input', '-i', required=True, help='Path to input image or video')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu'], help='Device to run inference on')
    parser.add_argument('--thresh', type=float, default=0.4, help='Score threshold')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for video inference')
    parser.add_argument('--kpt-radius', type=int, default=3, help='Radius of rendered keypoints')
    parser.add_argument('--kpt-line-thickness', type=int, default=2, help='Line thickness for skeleton links')
    parser.add_argument('--no-skeleton', action='store_true', help='Draw keypoints only, no skeleton links')
    main(parser.parse_args())
