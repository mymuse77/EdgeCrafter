"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
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
from engine.data.dataset.coco_dataset import mscoco_label2name_remap80


@dataclass
class Result:
    label: int
    score: float
    box: np.ndarray
    mask: np.ndarray = None


# COCO category colors for segmentation visualization
COCO_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128),
    (255, 128, 255), (128, 255, 255), (192, 0, 0), (0, 192, 0), (0, 0, 192),
    (192, 192, 0), (192, 0, 192), (0, 192, 192), (255, 192, 0), (255, 0, 192),
    (0, 255, 192), (192, 255, 0), (255, 192, 128), (192, 255, 128), (128, 192, 255),
    (255, 128, 192), (128, 255, 192), (192, 128, 255), (255, 192, 192), (192, 255, 192),
    (192, 192, 255), (255, 255, 192), (255, 192, 255), (192, 255, 255), (64, 0, 0),
    (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), (128, 64, 0),
    (128, 0, 64), (0, 128, 64), (64, 128, 0), (128, 64, 128), (64, 128, 128), (128, 128, 64),
    (192, 64, 0), (192, 0, 64), (0, 192, 64), (64, 192, 0), (192, 64, 192), (64, 192, 192),
    (192, 192, 64), (255, 64, 0), (255, 0, 64), (0, 255, 64), (64, 255, 0),
    (255, 64, 128), (64, 255, 128), (128, 64, 255), (255, 128, 64), (128, 255, 64),
    (64, 128, 255), (192, 64, 128), (192, 128, 64), (64, 192, 128), (128, 192, 64),
    (64, 128, 192), (128, 64, 192), (192, 128, 192), (128, 192, 192), (192, 192, 128)
]


def apply_mask(image, mask, color, alpha=0.5):
    """Apply segmentation mask to image with transparency."""
    mask = mask.astype(np.uint8)
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + color[c] * alpha,
            image[:, :, c]
        )
    return image


def draw(image, results, alpha=0.5):
    """Draw segmentation results on image."""
    # Convert PIL to numpy array
    im_np = np.array(image).copy()
    
    for res in results:
        color = COCO_COLORS[res.label % len(COCO_COLORS)]

        if res.mask is not None:
            im_np = apply_mask(im_np, res.mask, color, alpha)

        x1, y1, x2, y2 = res.box.astype(int)

        cv2.rectangle(im_np, (x1, y1), (x2, y2), color, 2)

        # replace mscoco_label2name_remap80 with your own label mapping if not using COCO
        text = f"{mscoco_label2name_remap80.get(res.label, str(res.label))} {res.score:.2f}"  
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 1)

        cv2.rectangle(im_np, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(im_np, text, (x1, y1 - 2), font, 0.8, (255, 255, 255), 1)

    return Image.fromarray(im_np.astype(np.uint8))


class ECInferencer:
    def __init__(self, model, task, device, size, thresh, half=True):
        self.model = model
        self.task = task
        self.device = device
        self.size = size
        self.thresh = thresh
        self.half = half
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        return T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def infer_batch(self, images):
        batch_size = len(images)
        orig_sizes = torch.tensor([[img.size[0], img.size[1]] for img in images], device=self.device)

        # Batch transform
        tensors = torch.stack([self.transforms(img) for img in images]).to(self.device)

        # Use AMP for max GPU utilization
        device_type = self.device.type if self.device.type != 'cpu' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda' and self.half)):
            outputs = self.model(tensors, orig_sizes)

        if self.task == "segmentation":
            labels, boxes, scores, masks = outputs
        elif self.task == "detection":
            labels, boxes, scores = outputs
            masks = None
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        batch_results = []
        for i in range(batch_size):
            keep = scores[i] > self.thresh
            lbls = labels[i][keep]
            bxs = boxes[i][keep]
            scs = scores[i][keep]

            res = []
            if masks is not None:
                img_h, img_w = images[i].size[1], images[i].size[0]
                msks = masks[i].unsqueeze(0)  # [1, num_queries, H, W]
                msks = torch.nn.functional.interpolate(
                    msks,
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                msks = msks > 0.0
                msks = msks[keep]
                
                for j in range(len(lbls)):
                    res.append(Result(
                        label=int(lbls[j].item()),
                        score=float(scs[j].item()),
                        box=bxs[j].cpu().numpy(),
                        mask=msks[j].cpu().numpy(),
                    ))
            else:
                for j in range(len(lbls)):
                   res.append(Result(
                        label=int(lbls[j].item()),
                        score=float(scs[j].item()),
                        box=bxs[j].cpu().numpy(),))
            batch_results.append(res)
            
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
        # empty queue to prevent blocking
        while not self.q.empty():
            self.q.get()
            

def process_image(inferencer, path):
    image = Image.open(path).convert("RGB")
    results = inferencer.infer(image)
    image = draw(image, results)
    
    output_path = path.with_stem(f"{path.stem}_torch_inference")
    image.save(output_path, quality=95, subsampling=0)
    print(f"Saved result to {output_path}")
    print(f"Detected {len(results)} instances")


def process_video(inferencer, path, batch_size=8, num_workers=4):
    cap = cv2.VideoCapture(str(path))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = path.with_stem(f"{path.stem}_torch_inference")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    reader = VideoReader(cap)
    reader.start()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    futures_queue = queue.Queue()

    def process_and_draw(pil_img, results):
        res_img = draw(pil_img, results)
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
                future = executor.submit(process_and_draw, pil_img, results)
                futures_queue.put(future)
            buffer_pil = []

    if len(buffer_pil) > 0:
        results_batch = inferencer.infer_batch(buffer_pil)
        for pil_img, results in zip(buffer_pil, results_batch):
            future = executor.submit(process_and_draw, pil_img, results)
            futures_queue.put(future)

    # Cleanup
    reader.stop()
    futures_queue.put(None)
    writer_thread.join()
    executor.shutdown()
    cap.release()
    out.release()
    print(f"Saved video result to {output_path}")


def build_model(config_path: str, resume_path: str, device: torch.device):
    cfg = YAMLConfig(config_path, resume=resume_path)  
    cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
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

    return model, cfg.yaml_cfg["eval_spatial_size"], cfg.yaml_cfg['task']


def main(args):
    device = torch.device(args.device)
    model, img_size, task = build_model(args.config, args.resume, device)

    # Half precision for significant speedup on GPU
    if device.type == 'cuda' and not args.fp32:
        model = model.half()
        
    inferencer = ECInferencer(
        model=model,
        task=task,
        device=device,
        size=img_size,
        thresh=args.thresh,
        half=(device.type == 'cuda' and not args.fp32)
    )

    input_path = Path(args.input)

    if input_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        process_image(inferencer, input_path)
    else:
        process_video(inferencer, input_path, batch_size=args.batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EdgeCrafter Inference")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-d", "--device", default="cuda:0", help="Device to run inference on (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("-t", "--thresh", type=float, default=0.4)
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for video inference")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision instead of FP16 (AMP)")

    main(parser.parse_args())
