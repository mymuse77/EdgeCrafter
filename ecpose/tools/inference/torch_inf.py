"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Pose inference script.
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
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


@dataclass
class Result:
    label: int
    score: float
    keypoints: np.ndarray


# COCO keypoint skeleton (1-based in standard definition)
COCO_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]
COCO_SKELETON = [(a - 1, b - 1) for a, b in COCO_SKELETON]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


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
        color = (0, 255, 0)

        for idx, (x, y) in enumerate(kpts):
            if vis[idx]:
                cv2.circle(im_np, (int(x), int(y)), radius, color, -1)

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


def _parse_pose_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError(f'Unexpected pose model outputs. Expected (scores, labels, keypoints), got type={type(outputs)}')
    scores, labels, keypoints = outputs
    return scores, labels, keypoints


class ECPoseInferencer:
    def __init__(self, model, device, size, thresh, half=True):
        self.model = model
        self.device = device
        self.size = size
        self.thresh = thresh
        self.half = half
        self.transforms = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def infer_batch(self, images):
        orig_sizes = torch.tensor(
            [[img.size[0], img.size[1]] for img in images],
            device=self.device,
            dtype=torch.int64,
        )
        tensors = torch.stack([self.transforms(img) for img in images]).to(self.device)

        device_type = self.device.type if self.device.type != 'cpu' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda' and self.half)):
            outputs = self.model(tensors, orig_sizes)

        scores, labels, keypoints = _parse_pose_outputs(outputs)

        batch_results = []
        for i in range(len(images)):
            keep = scores[i] > self.thresh
            scs = scores[i][keep]
            lbs = labels[i][keep]
            kps = keypoints[i][keep]

            results = []
            for j in range(len(scs)):
                results.append(Result(
                    label=int(lbs[j].item()),
                    score=float(scs[j].item()),
                    keypoints=kps[j].detach().cpu().numpy(),
                ))

            batch_results.append(results)

        return batch_results

    @torch.no_grad()
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

    output_path = path.with_stem(f"{path.stem}_torch_pose_inference")
    image.save(output_path, quality=95, subsampling=0)
    print(f"Saved result to {output_path}")
    print(f"Detected {len(results)} poses")


def process_video(inferencer, path: Path, batch_size=8, num_workers=4, radius=3, line_thickness=2, draw_skeleton=True):
    cap = cv2.VideoCapture(str(path))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = path.with_stem(f"{path.stem}_torch_pose_inference")
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


def build_model(config_path: str, resume_path: str, device: torch.device):
    cfg = YAMLConfig(config_path, resume=resume_path)

    if 'ViTAdapter' in cfg.yaml_cfg:
        cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True

    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()
    return model, tuple(cfg.yaml_cfg['eval_spatial_size']), cfg.yaml_cfg.get('task', '')


def main(args):
    device = torch.device(args.device)
    model, img_size, task = build_model(args.config, args.resume, device)
    if task != 'pose':
        print(f"Warning: config task is '{task}', script is specialized for pose inference.")

    if device.type == 'cuda' and not args.fp32:
        model = model.half()

    inferencer = ECPoseInferencer(
        model=model,
        device=device,
        size=img_size,
        thresh=args.thresh,
        half=(device.type == 'cuda' and not args.fp32),
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

    parser = argparse.ArgumentParser(description='ECPose Torch Inference')
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-r', '--resume', required=True, type=str)
    parser.add_argument('-i', '--input', required=True, type=str)
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument('-t', '--thresh', default=0.4, type=float)
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size for video inference')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 precision instead of FP16')
    parser.add_argument('--kpt-radius', type=int, default=3, help='Radius of rendered keypoints')
    parser.add_argument('--kpt-line-thickness', type=int, default=2, help='Line thickness for skeleton links')
    parser.add_argument('--no-skeleton', action='store_true', help='Draw keypoints only, no skeleton links')
    main(parser.parse_args())
