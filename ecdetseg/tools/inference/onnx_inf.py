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
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from torch_inf import COCO_COLORS

from engine.data.dataset.coco_dataset import mscoco_label2name_remap80


@dataclass
class Result:
    label: int
    score: float
    box: np.ndarray
    mask: np.ndarray = None


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


class ECOnnxInferencer:
    """ONNX Inference Engine for EdgeCrafter models."""

    def __init__(self, session, task, size, thresh):
        self.session = session
        self.task = task
        self.size = size
        self.thresh = thresh
        self.transforms = self._build_transforms()
        
        # Get input/output info
        self.input_names = [inp.name for inp in session.get_inputs()]
        self.output_names = [out.name for out in session.get_outputs()]

    def _build_transforms(self):
        return T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def infer_batch(self, images):
        batch_results = []
        for img in images:
            w, h = img.size
            orig_size = np.array([[w, h]], dtype=np.int64)
            tensor = self.transforms(img).unsqueeze(0)

            input_feed = {
                'images': tensor.numpy(),
                'orig_target_sizes': orig_size
            }

            outputs = self.session.run(output_names=None, input_feed=input_feed)

            if self.task == "segmentation":
                labels, boxes, scores, masks = outputs
            elif self.task == "detection":
                labels, boxes, scores = outputs
                masks = None
            else:
                raise ValueError(f"Unsupported task: {self.task}")

            # Original script expects scores of shape (1, num_queries)
            # Flattening keep mask to 1D or indexing properly
            keep = scores[0] > self.thresh
            lbls = labels[0][keep]
            bxs = boxes[0][keep]
            scs = scores[0][keep]

            res = []
            if masks is not None:
                img_h, img_w = h, w
                # masks shape is usually (1, num_queries, H, W)
                msks = torch.from_numpy(masks[0]).unsqueeze(0)  # [1, num_queries, H, W]
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
                        box=bxs[j],
                        mask=msks[j].cpu().numpy(),
                    ))
            else:
                for j in range(len(lbls)):
                    res.append(Result(
                        label=int(lbls[j].item()),
                        score=float(scs[j].item()),
                        box=bxs[j],
                    ))
            batch_results.append(res)
            
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


def process_image(inferencer, path):
    image = Image.open(path).convert("RGB")
    results = inferencer.infer(image)
    image = draw(image, results)

    output_path = path.with_stem(f"{path.stem}_onnx_inference")
    image.save(output_path, quality=95, subsampling=0)
    print(f"Saved result to {output_path}")
    print(f"Detected {len(results)} instances")


def process_video(inferencer, path, batch_size=8, num_workers=4):
    cap = cv2.VideoCapture(str(path))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = path.with_stem(f"{path.stem}_onnx_inference")
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


def main(args):
    """Main function."""
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(args.onnx, sess_options, providers=providers)
    num_outputs = len(session.get_outputs())
    if num_outputs == 4:
        args.task = "segmentation"
    else:
        args.task = "detection"

    print(f"Using device: {args.device}")
    print(f"ONNX Runtime device: {ort.get_device()}")
    
    # Get input size from model
    input_shape = session.get_inputs()[0].shape
    
    
    if isinstance(input_shape[2], int):
        img_size = (input_shape[2], input_shape[3])  # (H, W)
    else:
        img_size = (640, 640) # Default fallback if dynamic
        
    print(f"Model input size: {img_size}")

    inferencer = ECOnnxInferencer(
        session=session,
        task=args.task,
        size=img_size,
        thresh=args.thresh,
    )

    input_path = Path(args.input)

    if input_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        process_image(inferencer, input_path)
    else:
        process_video(inferencer, input_path, batch_size=args.batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EdgeCrafter ONNX Inference")
    parser.add_argument("--onnx", "-o", required=True, help="Path to ONNX model file")
    parser.add_argument("--input", "-i", required=True, help="Path to input image or video")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--thresh", type=float, default=0.4, help="Score threshold")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for video inference")

    main(parser.parse_args())
