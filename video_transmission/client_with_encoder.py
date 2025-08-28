#!/usr/bin/env python3
import argparse
import socket
import struct
import json
import time
import cv2
import numpy as np
import torch
from encoder import load_encoder

MAGIC = b'VPKT'
VERSION = 1
TYPE_META  = 1
TYPE_CHUNK = 2

# 头格式: magic(4s) ver(B) type(B) frame_id(u32) total_chunks(u16) chunk_id(u16) payload_len(u16)
HDR = struct.Struct('!4sBBIHHH')
MTU = 1200  # 单个 UDP 负载大小（保守设置，避免分片）

def to_bchw_cuda(frame_bgr: np.ndarray, size_wh=None) -> torch.Tensor:
    if size_wh is not None:
        w, h = size_wh
        frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float() / 255.0          # [H,W,3]
    return t.permute(2, 0, 1).unsqueeze(0).to('cuda')  # [1,3,H,W]

def draw_overlay(img_bgr, text_lines, origin=(10, 28)):
    """在 BGR 图像上叠加多行文本"""
    x, y = origin
    for line in text_lines:
        cv2.putText(img_bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
        y += 28

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest-ip', default="127.0.0.1")
    ap.add_argument('--dest-port', type=int, default=50000)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--size', type=str, default="", help='例如 640x360，不填则用原分辨率')
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--preview', action='store_true', default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    encoder = load_encoder()
    encoder.eval()

    target_wh = None
    if args.size:
        w, h = map(int, args.size.lower().split('x'))
        target_wh = (w, h)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.dest_ip, args.dest_port)
    frame_id = 0
    interval = 1.0 / max(args.fps, 1)

    # 统计相关
    stat_last = time.time()
    frames_in_sec = 0
    bytes_in_sec = 0
    disp_fps = 0.0
    disp_kbps = 0.0

    print(f"[encoder] send to udp://{args.dest_ip}:{args.dest_port}")
    try:
        with torch.no_grad():
            last_send = time.time()
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    continue

                # 控制采样帧率
                now = time.time()
                if now - last_send < interval:
                    # 预览依然可显示摄像头原始画面（可选）
                    if args.preview:
                        overlay = [f"FPS: {disp_fps:.2f}", f"Bitrate: {disp_kbps:.1f} kbps", "waiting..."]
                        show = frame_bgr.copy()
                        draw_overlay(show, overlay)
                        cv2.imshow("source", show)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    continue
                last_send = now

                # 1) BGR -> RGB -> [1,3,H,W] -> CUDA
                image = to_bchw_cuda(frame_bgr, target_wh)

                # 2) 编码得到 indices（忽略 quantized）
                indices, _ = encoder.encode(image)  # 你的 API

                # 3) 准备要发送的原始字节、形状、dtype（把 indices 搬到 CPU）
                idx_cpu = indices.detach().contiguous().to('cpu')
                np_view = idx_cpu.numpy()  # 不复制
                payload = np_view.tobytes(order='C')
                meta = {
                    "shape": list(np_view.shape),
                    "dtype": str(np_view.dtype),   # 例如 'int64'/'uint16'/'float32'
                }
                meta_bytes = json.dumps(meta).encode('utf-8')

                # 4) 发送 META 包
                total_chunks = (len(payload) + MTU - 1) // MTU
                pkt = HDR.pack(MAGIC, VERSION, TYPE_META, frame_id, total_chunks, 0, len(meta_bytes)) + meta_bytes
                sock.sendto(pkt, addr)
                bytes_in_sec += len(pkt)

                # 5) 发送数据分片
                for chunk_id in range(total_chunks):
                    start = chunk_id * MTU
                    end = min(start + MTU, len(payload))
                    chunk = payload[start:end]
                    pkt = HDR.pack(MAGIC, VERSION, TYPE_CHUNK, frame_id, total_chunks, chunk_id, len(chunk)) + chunk
                    sock.sendto(pkt, addr)
                    bytes_in_sec += len(pkt)

                frame_id = (frame_id + 1) & 0xFFFFFFFF
                frames_in_sec += 1

                # 每秒刷新一次统计并叠加到预览
                now2 = time.time()
                if now2 - stat_last >= 1.0:
                    disp_fps = frames_in_sec / (now2 - stat_last)
                    disp_kbps = (bytes_in_sec * 8) / (now2 - stat_last) / 1000.0
                    frames_in_sec = 0
                    bytes_in_sec = 0
                    stat_last = now2

                if args.preview:
                    show = frame_bgr.copy()
                    overlay = [f"FPS: {disp_fps:.2f}", f"Bitrate: {disp_kbps:.1f} kbps"]
                    if target_wh is not None:
                        overlay.append(f"Size: {target_wh[0]}x{target_wh[1]}")
                    draw_overlay(show, overlay)
                    cv2.imshow("source", show)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()

if __name__ == '__main__':
    main()
