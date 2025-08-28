#!/usr/bin/env python3
import argparse
import socket
import struct
import json
import cv2
import numpy as np
import torch
import time
from decoder import load_decoder

MAGIC = b'VPKT'
VERSION = 1
TYPE_META  = 1
TYPE_CHUNK = 2
HDR = struct.Struct('!4sBBIHHH')
TIMEOUT_SEC = 0.5   # 单帧装配超时，避免卡顿

def torch_to_bgr(t: torch.Tensor):
    """
    t: [1,3,H,W] 或 [3,H,W]，范围 [0,1] 或 [-1,1]（按需修改）
    return: BGR uint8 numpy array
    """
    if t.dim() == 4:
        t = t[0]
    # 如果你的 decoder 输出是 [-1,1]，请解除下一行注释
    # t = (t*0.5 + 0.5)
    t = t.clamp(0,1).detach().to(torch.float32).permute(1,2,0).cpu().numpy()  # RGB[H,W,3]
    bgr = cv2.cvtColor((t*255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr

def draw_overlay(img_bgr, text_lines, origin=(10, 28)):
    x, y = origin
    for line in text_lines:
        cv2.putText(img_bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
        y += 28

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen-ip', default='0.0.0.0')
    ap.add_argument('--listen-port', type=int, default=50000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    decoder = load_decoder()
    decoder.eval()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_ip, args.listen_port))
    sock.settimeout(0.05)

    # 多帧装配缓存
    metas = {}              # frame_id -> meta(dict)
    buffers = {}            # frame_id -> bytearray
    got = {}                # frame_id -> set(chunk_id)
    totals = {}             # frame_id -> total_chunks
    first_seen = {}         # frame_id -> timestamp

    # 统计相关：按“解码成功的帧”来计 FPS；按“所有收到的UDP包”来计码率
    stat_last = time.time()
    frames_in_sec = 0
    bytes_in_sec = 0
    disp_fps = 0.0
    disp_kbps = 0.0

    print(f"[decoder] listen on udp://{args.listen_ip}:{args.listen_port}")
    while True:
        data = None
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            pass

        # 清理过期帧
        if first_seen:
            now = time.time()
            dead = [fid for fid,t0 in first_seen.items() if now - t0 > TIMEOUT_SEC]
            for fid in dead:
                metas.pop(fid, None)
                buffers.pop(fid, None)
                got.pop(fid, None)
                totals.pop(fid, None)
                first_seen.pop(fid, None)

        if data is None:
            # 即使这一轮没数据，也定期刷新统计显示（下一帧会带着旧数值）
            now2 = time.time()
            if now2 - stat_last >= 1.0:
                disp_fps = frames_in_sec / (now2 - stat_last)
                disp_kbps = (bytes_in_sec * 8) / (now2 - stat_last) / 1000.0
                frames_in_sec = 0
                bytes_in_sec = 0
                stat_last = now2
            continue

        bytes_in_sec += len(data)

        if len(data) < HDR.size:
            continue
        magic, ver, typ, fid, total, cid, plen = HDR.unpack(data[:HDR.size])
        if magic != MAGIC or ver != VERSION:
            continue
        payload = data[HDR.size:HDR.size+plen]
        if len(payload) != plen:
            continue

        if fid not in first_seen:
            first_seen[fid] = time.time()

        if typ == TYPE_META:
            try:
                meta = json.loads(payload.decode('utf-8'))
            except Exception:
                continue
            metas[fid] = meta
            totals[fid] = total
            buffers.setdefault(fid, bytearray())
            got.setdefault(fid, set())

        elif typ == TYPE_CHUNK:
            if fid not in metas or fid not in totals:
                continue
            # 按需扩容到 (cid+1)*plen
            buf = buffers[fid]
            need_len = (cid + 1) * plen
            if len(buf) < need_len:
                buf.extend(b'\x00' * (need_len - len(buf)))
            start = cid * plen
            buf[start:start+plen] = payload
            got[fid].add(cid)

            # 收齐则解码
            if len(got[fid]) == totals[fid]:
                meta = metas[fid]
                shape = tuple(meta['shape'])
                dtype = np.dtype(meta['dtype'])
                elem_n = int(np.prod(shape))
                true_bytes = elem_n * dtype.itemsize
                raw = bytes(buffers[fid][:true_bytes])

                np_idx = np.frombuffer(raw, dtype=dtype).reshape(shape)
                indices = torch.from_numpy(np_idx).to('cuda', non_blocking=True)

                with torch.no_grad():
                    reconstructed = decoder.decode(indices.to('cuda'))

                # 转 numpy 以便叠加 overlay
                frame_bgr = torch_to_bgr(reconstructed)
                overlay = [f"FPS: {disp_fps:.2f}",
                           f"Bitrate: {disp_kbps:.1f} kbps",
                           f"Shape: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}"]
                draw_overlay(frame_bgr, overlay)
                cv2.imshow('reconstructed', frame_bgr)
                cv2.waitKey(1)

                # 解码成功 -> 帧计数 + 统计刷新
                frames_in_sec += 1
                now2 = time.time()
                if now2 - stat_last >= 1.0:
                    disp_fps = frames_in_sec / (now2 - stat_last)
                    disp_kbps = (bytes_in_sec * 8) / (now2 - stat_last) / 1000.0
                    frames_in_sec = 0
                    bytes_in_sec = 0
                    stat_last = now2

                # 清理该帧
                for d in (metas, buffers, got, totals, first_seen):
                    d.pop(fid, None)

if __name__ == '__main__':
    main()
