#!/usr/bin/env python3
"""Turn a short list of keyframe images into a smooth MP4.

Reads the scenes' `image` entries from a storyboard.json (in order) and fills in the frames
between them up to the storyboard fps. Best for a fixed camera where people or vehicles change
position between keyframes.

Modes
  gpu   GPU cross-fade between keyframes (OpenCL on the GPU, falls back to CPU). Fast; moving
        objects ghost slightly while they cross-fade.
  mci   Motion-compensated interpolation (ffmpeg minterpolate). Looks like real motion but has no
        GPU version, so the keyframe gaps are computed in parallel on all CPU cores.
Both modes encode on the GPU video encoder (NVENC, Quick Sync, AMF) when one works.

    python keyframes_to_video.py D:/VideoCreator/projects/<name>/storyboard.json
    python keyframes_to_video.py <storyboard> --mode gpu --seconds 1.0
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import vc_config
import video_render as vr


def run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise vr.RenderError(f'{what} failed:\n' + '\n'.join(r.stderr.splitlines()[-15:]))


def prepare_images(ffmpeg, images, tmp, W, H):
    """Scale and crop every keyframe once to the output size (lossless PNG)."""
    vf = f'scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1'

    def one(item):
        i, p = item
        out = tmp / f'k{i:04d}.png'
        run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(p), '-vf', vf, '-frames:v', '1',
             '-compression_level', '1', str(out)], f'preparing {p.name}')
        return out

    with ThreadPoolExecutor(max_workers=min(len(images), os.cpu_count() or 4)) as ex:
        return list(ex.map(one, enumerate(images, start=1)))


def post_filters(board, total, first, last, span):
    chain = []
    if board['grain'] > 0:
        chain.append(f"noise=alls={max(1, round(board['grain'] * 120))}:allf=t")
    if first and board['fadeIn'] > 0:
        chain.append(f"fade=t=in:st=0:d={board['fadeIn']:.3f}")
    if last and board['fadeOut'] > 0:
        chain.append(f"fade=t=out:st={max(0.0, span - board['fadeOut']):.3f}:d={board['fadeOut']:.3f}")
    return chain


def render_mci(ffmpeg, pngs, tmp, board, seconds, out_path, encoder):
    fps = board['fps']
    n = max(2, round(seconds * fps))
    pairs = len(pngs) - 1
    workers = max(1, min(pairs, os.cpu_count() or 4))

    pad = 0.1
    gap = n / fps

    def segment(i):
        # minterpolate needs reference frames on both sides of the gap and emits nothing for only two,
        # so each keyframe is repeated briefly before and after the gap, then the padding is trimmed off.
        a, b = pngs[i].as_posix(), pngs[i + 1].as_posix()
        lst = tmp / f'seg{i}.ffconcat'
        lst.write_text(f"ffconcat version 1.0\nfile '{a}'\nduration {pad}\nfile '{a}'\nduration {gap:.6f}\n"
                       f"file '{b}'\nduration {pad}\nfile '{b}'\nduration {pad}\n", encoding='utf-8')
        last = i == pairs - 1
        frames = n + 1 if last else n
        chain = ['format=yuv420p', f'minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:search_param=16',
                 f'trim=start={pad},setpts=PTS-STARTPTS']
        chain += post_filters(board, 0, i == 0, last, frames / fps)
        out = tmp / f'seg{i}.mkv'
        run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(lst),
             '-vf', ','.join(chain), '-frames:v', str(frames), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '8',
             '-pix_fmt', 'yuv420p', '-threads', '2', str(out)], f'interpolating segment {i + 1}/{pairs}')
        if out.stat().st_size < 10_000:
            raise vr.RenderError(f'segment {i + 1}/{pairs} came out empty')
        return out

    print(f'  interpolating {pairs} gaps on {workers} CPU threads', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        segs = list(ex.map(segment, range(pairs)))

    listing = tmp / 'list.txt'
    listing.write_text(''.join(f"file '{s.name}'\n" for s in segs), encoding='utf-8')
    run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-hwaccel', 'auto', '-f', 'concat', '-safe', '0',
         '-i', str(listing), *vr.encoder_args(encoder, board['mbps']), '-r', str(fps), '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', str(out_path)], 'GPU encode')
    return (n * pairs + 1) / fps


def render_gpu(ffmpeg, pngs, tmp, board, seconds, out_path, encoder, hw):
    fps = board['fps']
    pairs = len(pngs) - 1
    total = pairs * seconds
    args = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y']
    if hw:
        args += ['-init_hw_device', 'opencl=ocl', '-filter_hw_device', 'ocl']
    graph = []
    for i, p in enumerate(pngs):
        # Each keyframe is only visible during the fade in and the fade out around it, so its input
        # only needs two gaps of frames; looping it for the whole video would re-decode it every frame.
        span = seconds if i == 0 or i == pairs else 2 * seconds
        args += ['-loop', '1', '-framerate', str(fps), '-t', f'{span + 2 / fps:.4f}', '-i', str(p)]
        graph.append(f'[{i}:v]format={"nv12,hwupload" if hw else "yuv420p"}[v{i}]')
    cur = 'v0'
    xf = 'xfade_opencl' if hw else 'xfade'
    for i in range(1, len(pngs)):
        graph.append(f'[{cur}][v{i}]{xf}=transition=fade:duration={seconds:.4f}:offset={(i - 1) * seconds:.4f}[x{i}]')
        cur = f'x{i}'
    tail = ['hwdownload', 'format=nv12', 'format=yuv420p'] if hw else ['format=yuv420p']
    tail += post_filters(board, total, True, True, total)
    graph.append(f'[{cur}]' + ','.join(tail) + '[out]')
    args += ['-filter_complex', ';'.join(graph), '-map', '[out]', *vr.encoder_args(encoder, board['mbps']),
             '-r', str(fps), '-t', f'{total:.3f}', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out_path)]
    run(args, 'GPU cross-fade')
    return total


def main():
    ap = argparse.ArgumentParser(description='Interpolate keyframes into a smooth MP4.')
    ap.add_argument('storyboard')
    ap.add_argument('--root', help='folder searched for images (default: the storage folder from setup)')
    ap.add_argument('--out', help='output file (default: <storage>/renders/<title>.mp4)')
    ap.add_argument('--quality', choices=list(vr.QUALITY))
    ap.add_argument('--codec', choices=list(vr.ENCODERS), default='h264')
    ap.add_argument('--seconds', type=float, default=1.3, help='seconds between keyframes (default 1.3)')
    ap.add_argument('--mode', choices=['gpu', 'mci'], help='default: the interpolation mode chosen in setup')
    a = ap.parse_args()

    ffmpeg = vr.find_tool('ffmpeg')
    if not ffmpeg:
        raise vr.RenderError('ffmpeg not found')
    cfg = vc_config.ensure_config()
    storage = Path(cfg['storage_dir'])
    mode = a.mode or cfg['interp_mode']
    sb_path = Path(a.storyboard).resolve()
    root = Path(a.root).resolve() if a.root else storage
    raw = json.loads(sb_path.read_text(encoding='utf-8-sig'))
    board = vr.normalize(raw, a.quality or (None if raw.get('quality') or raw.get('width') else cfg['quality']))

    index = vr.index_files(root)
    bases = [sb_path.parent, root]
    images = []
    for s in board['scenes']:
        p = vr.resolve(s['image'], bases, index) if s['image'] else None
        if not p:
            raise vr.RenderError(f"image not found: {s['image']}")
        images.append(p)
    if len(images) < 2:
        raise vr.RenderError('need at least 2 keyframes')

    W, H = board['width'], board['height']
    slug = re.sub(r'[^\w.-]+', '-', board['title']).strip('-.') or 'video'
    out_path = Path(a.out).resolve() if a.out else storage / 'renders' / f'{slug}.mp4'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = vr.pick_encoder(ffmpeg, a.codec, board['mbps'], cfg['render_mode'], cfg['encoder'])
    print(f"{W}x{H} {board['fps']}fps, {len(images)} keyframes, mode {mode}, encoder {encoder}")

    start = time.time()
    with tempfile.TemporaryDirectory(prefix='keyframes-', dir=vc_config.temp_dir(cfg)) as td:
        tmp = Path(td)
        pngs = prepare_images(ffmpeg, images, tmp, W, H)
        if mode == 'mci':
            length = render_mci(ffmpeg, pngs, tmp, board, a.seconds, out_path, encoder)
        elif cfg['render_mode'] == 'cpu':
            length = render_gpu(ffmpeg, pngs, tmp, board, a.seconds, out_path, encoder, hw=False)
        else:
            try:
                length = render_gpu(ffmpeg, pngs, tmp, board, a.seconds, out_path, encoder, hw=True)
                print('  cross-fade ran on the GPU (OpenCL)')
            except vr.RenderError as e:
                print('  GPU cross-fade unavailable, using CPU:', str(e).splitlines()[-1])
                length = render_gpu(ffmpeg, pngs, tmp, board, a.seconds, out_path, encoder, hw=False)
    elapsed = time.time() - start
    print(f'Done: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {length:.1f}s video in {elapsed:.1f}s)')


if __name__ == '__main__':
    try:
        main()
    except vr.RenderError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
