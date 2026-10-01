#!/usr/bin/env python3
"""Render a Video Creator storyboard.json to MP4 with ffmpeg.

Same timeline, camera motion, captions and look as studio.html. Uses the GPU video
encoder when one works (NVENC, Quick Sync, AMF, VideoToolbox) and falls back to software.
Needs Python 3.8+ and ffmpeg/ffprobe; no Python packages.

    python video_render.py video-projects/<slug>/storyboard.json
    python video_render.py <storyboard> --quality 4k --codec hevc --out my.mp4
"""
import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

ASPECTS = {'16:9': (16, 9), '9:16': (9, 16), '1:1': (1, 1), '4:3': (4, 3), '3:4': (3, 4), '21:9': (21, 9)}
QUALITY = {'720p': 720, '1080p': 1080, '1440p': 1440, '4k': 2160}
MOTIONS = ['zoom-in', 'zoom-out', 'pan-left', 'pan-right', 'pan-up', 'pan-down', 'static']
TRANSITIONS = ['crossfade', 'fade-black', 'cut']
SKIP_DIRS = {'node_modules', 'renders'}
NAMED_COLORS = {'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0), 'green': (0, 128, 0),
                'blue': (0, 0, 255), 'gray': (128, 128, 128), 'grey': (128, 128, 128), 'navy': (0, 0, 128),
                'orange': (255, 165, 0), 'purple': (128, 0, 128), 'yellow': (255, 255, 0)}

FONT_CANDIDATES = {
    'bold': ['C:/Windows/Fonts/malgunbd.ttf', '/System/Library/Fonts/AppleSDGothicNeo.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', '/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc',
             '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'],
    'regular': ['C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'],
}

ENCODERS = {
    'h264': ['h264_nvenc', 'h264_qsv', 'h264_amf', 'h264_videotoolbox', 'libx264'],
    'hevc': ['hevc_nvenc', 'hevc_qsv', 'hevc_amf', 'hevc_videotoolbox', 'libx265'],
    'av1': ['av1_nvenc', 'av1_qsv', 'av1_amf', 'libsvtav1', 'libaom-av1'],
}


class RenderError(Exception):
    pass


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def num(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def effect_value(v, default):
    if v is None or v is True:
        return default
    if v is False:
        return 0.0
    return clamp(num(v, default))


def even(v):
    v = int(round(v))
    return v - v % 2


# ---------------- storyboard ----------------

def normalize(raw, quality_override=None):
    scenes_raw = raw.get('scenes') if isinstance(raw, dict) else None
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise RenderError('storyboard has no scenes')
    aspect = raw.get('aspect') if raw.get('aspect') in ASPECTS else '16:9'
    width, height = num(raw.get('width'), 0), num(raw.get('height'), 0)
    if quality_override or not (width > 0 and height > 0):
        short = QUALITY.get(str(quality_override or raw.get('quality') or '1080p').lower(), 1080)
        aw, ah = ASPECTS[aspect]
        if aw >= ah:
            height, width = short, round(short * aw / ah)
        else:
            width, height = short, round(short * ah / aw)
    width, height = even(width), even(height)
    fps = int(round(clamp(num(raw.get('fps'), 30), 12, 60)))

    def_transition = num(raw.get('transition'), 0.8)
    def_type = raw.get('transitionType') if raw.get('transitionType') in TRANSITIONS else 'crossfade'
    scenes = []
    for i, s in enumerate(scenes_raw):
        s = s if isinstance(s, dict) else {}
        ttype = s.get('transitionType') if s.get('transitionType') in TRANSITIONS else def_type
        focus = s.get('focus')
        frames = max(1, round(max(0.5, num(s.get('duration'), 4)) * fps))
        scenes.append({
            'image': s.get('image') or None,
            'background': s.get('background') or '#000',
            'frames': frames,
            'duration': frames / fps,
            'motion': s.get('motion') if s.get('motion') in MOTIONS else (MOTIONS[i % 4] if s.get('image') else 'static'),
            'intensity': clamp(num(s.get('intensity'), 0.15), 0, 0.6),
            'focus': [clamp(num(focus[0], 0.5)), clamp(num(focus[1], 0.5))] if isinstance(focus, list) and len(focus) >= 2 else None,
            'transition': 0.0 if i == 0 or ttype == 'cut' else max(0.0, num(s.get('transition'), def_transition)),
            'transitionType': ttype,
            'title': str(s.get('title') or ''),
            'caption': str(s.get('caption') or ''),
        })

    for i in range(1, len(scenes)):
        s = scenes[i]
        t = min(s['transition'], scenes[i - 1]['duration'] / 2, s['duration'] / 2)
        s['transition'] = round(t * fps) / fps

    effects = raw.get('effects') if isinstance(raw.get('effects'), dict) else {}
    look = raw.get('look') if isinstance(raw.get('look'), dict) else {}
    default_mbps = round(clamp(width * height * fps / 1e6 * 0.22, 6, 80))
    return {
        'title': str(raw.get('title') or 'video'),
        'width': width, 'height': height, 'fps': fps, 'scenes': scenes,
        'vignette': effect_value(effects.get('vignette'), 0.35),
        'grain': effect_value(effects.get('grain'), 0.04),
        'letterbox': bool(effects.get('letterbox')),
        'contrast': num(look.get('contrast'), 1.05),
        'saturation': num(look.get('saturation'), 1.0),
        'warmth': num(look.get('warmth'), 0.0),
        'fadeIn': max(0.0, num(raw.get('fadeIn'), 0.6)),
        'fadeOut': max(0.0, num(raw.get('fadeOut'), 0.8)),
        'music': raw.get('music') or None,
        'musicVolume': clamp(num(raw.get('musicVolume'), 0.8)),
        'mbps': int(num(raw.get('bitrate'), default_mbps)) or default_mbps,
    }


def letterbox_bar(w, h):
    return round((h - w / 2.39) / 2) if w > h and w / h < 2.39 else 0


# ---------------- assets ----------------

def index_files(root, max_depth=4):
    found = {}

    def walk(d, depth):
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                if depth < max_depth and not e.name.startswith('.') and e.name not in SKIP_DIRS:
                    walk(e.path, depth + 1)
            elif e.is_file():
                found.setdefault(e.name.lower(), Path(e.path))

    walk(root, 0)
    return found


def resolve(ref, bases, index):
    p = Path(ref)
    candidates = [p] if p.is_absolute() else [b / p for b in bases]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    hit = index.get(p.name.lower())
    return hit.resolve() if hit else None


def parse_color(c):
    s = str(c).strip()
    if s.lower() in NAMED_COLORS:
        return NAMED_COLORS[s.lower()]
    s = s.lstrip('#')
    if re.fullmatch(r'[0-9a-fA-F]{3}', s):
        s = ''.join(ch * 2 for ch in s)
    if re.fullmatch(r'[0-9a-fA-F]{6}([0-9a-fA-F]{2})?', s):
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    return (0, 0, 0)


def write_png(path, w, h, rows):
    def chunk(kind, data):
        body = kind + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF)

    raw = b''.join(b'\x00' + r for r in rows)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def background_png(path, bg, W, H):
    """Diagonal gradient matching the studio's canvas createLinearGradient(0, 0, w, h)."""
    colors = [parse_color(c) for c in bg] if isinstance(bg, list) and bg else [parse_color(bg)]
    w, h = 512, max(1, round(512 * H / W))
    if len(colors) == 1:
        rows = [bytes(colors[0]) * w] * h
    else:
        stops, denom, steps = len(colors) - 1, w * w + h * h, 1024
        lut = []
        for s in range(steps + 1):
            p = s / steps * stops
            i = min(int(p), stops - 1)
            f, a, b = p - i, colors[i], colors[i + 1]
            lut.append(bytes(round(a[k] + (b[k] - a[k]) * f) for k in range(3)))
        rows = [b''.join(lut[(x * w + y * h) * steps // denom] for x in range(w)) for y in range(h)]
    write_png(path, w, h, rows)
    return w, h


# ---------------- text ----------------

def char_width(ch):
    o = ord(ch)
    if ch == ' ':
        return 0.3
    if 0x1100 <= o <= 0x11FF or 0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF:
        return 1.0
    return 0.65 if ch.isupper() or ch.isdigit() else 0.55


def text_width(s, size):
    return sum(char_width(c) for c in s) * size


def wrap_text(text, size, max_w):
    out = []
    for para in text.split('\n'):
        line = ''
        for word in para.split(' '):
            test = f'{line} {word}' if line else word
            if text_width(test, size) <= max_w:
                line = test
                continue
            if line:
                out.append(line)
            if text_width(word, size) <= max_w:
                line = word
                continue
            line = ''
            for ch in word:
                if line and text_width(line + ch, size) > max_w:
                    out.append(line)
                    line = ch
                else:
                    line += ch
        out.append(line)
    return out


def text_filters(lines, size, centers, font, tmp, prefix, duration, shadow):
    eo = lambda v: f'(1-pow(1-clip({v},0,1),2))'
    alpha = f"min({eo('(t-0.25)/0.6')},{eo(f'({duration:.4f}-t-0.1)/0.5')})"
    out = []
    for j, (line, cy) in enumerate(zip(lines, centers)):
        if not line.strip():
            continue
        name = f'{prefix}_{j}.txt'
        (tmp / name).write_text(line, encoding='utf-8')
        font_opt = f'fontfile={font}:' if font else ''
        out.append(f"drawtext={font_opt}textfile={name}:expansion=none:fontsize={size}:fontcolor=white"
                   f":shadowcolor=black@0.6:shadowx={shadow}:shadowy={shadow}"
                   f":x=(w-text_w)/2:y={cy:.1f}-text_h/2:alpha='{alpha}'")
    return out


# ---------------- ffmpeg ----------------

def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    exe = name + ('.exe' if os.name == 'nt' else '')
    extra = []
    if os.environ.get('LOCALAPPDATA'):
        extra.append(Path(os.environ['LOCALAPPDATA']) / 'Microsoft' / 'WinGet' / 'Links' / exe)
    extra += [Path('/opt/homebrew/bin') / name, Path('/usr/local/bin') / name]
    for c in extra:
        if c.is_file():
            return str(c)
    return None


def probe_size(ffprobe, path):
    r = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                        '-of', 'csv=p=0:s=x', str(path)], capture_output=True, text=True)
    m = re.search(r'(\d+)x(\d+)', r.stdout)
    if r.returncode != 0 or not m:
        raise RenderError(f'cannot read image: {path}')
    return int(m.group(1)), int(m.group(2))


def image_size(ffprobe, path):
    try:
        with open(path, 'rb') as f:
            head = f.read(24)
    except OSError:
        head = b''
    if head[:8] == b'\x89PNG\r\n\x1a\n' and head[12:16] == b'IHDR':
        return struct.unpack('>II', head[16:24])
    return probe_size(ffprobe, path)


def encoder_args(name, mbps):
    b, mx, buf = f'{mbps}M', f'{round(mbps * 1.5)}M', f'{mbps * 2}M'
    if name.endswith('_nvenc'):
        args = ['-c:v', name, '-preset', 'p5', '-rc', 'vbr', '-b:v', b, '-maxrate', mx, '-bufsize', buf]
    elif name.endswith('_qsv'):
        args = ['-c:v', name, '-preset', 'slow', '-b:v', b, '-maxrate', mx, '-bufsize', buf]
    elif name.endswith('_amf'):
        args = ['-c:v', name, '-quality', 'quality', '-b:v', b, '-maxrate', mx]
    elif name.endswith('_videotoolbox'):
        args = ['-c:v', name, '-b:v', b]
    elif name == 'libx264':
        args = ['-c:v', name, '-preset', 'medium', '-crf', '18', '-maxrate', mx, '-bufsize', buf]
    elif name == 'libx265':
        args = ['-c:v', name, '-preset', 'medium', '-crf', '20']
    elif name == 'libsvtav1':
        args = ['-c:v', name, '-preset', '8', '-crf', '30']
    else:
        args = ['-c:v', name, '-cpu-used', '6', '-crf', '30', '-b:v', '0']
    if name.startswith('hevc') or name == 'libx265':
        args += ['-tag:v', 'hvc1']
    return args


def pick_encoder(ffmpeg, codec, mbps):
    listed = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    for name in ENCODERS[codec]:
        if not re.search(rf'\s{re.escape(name)}\s', listed):
            continue
        test = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=black:s=320x240:r=30:d=0.3',
                *encoder_args(name, mbps), '-f', 'null', '-']
        if subprocess.run(test, capture_output=True).returncode == 0:
            return name
    if codec != 'h264':
        print(f'No working {codec} encoder; using h264.')
        return pick_encoder(ffmpeg, 'h264', mbps)
    raise RenderError('ffmpeg has no working H.264 encoder')


def find_font(kind, override):
    if override:
        return Path(override) if Path(override).is_file() else None
    for c in FONT_CANDIDATES[kind]:
        if Path(c).is_file():
            return Path(c)
    return None


def build_command(board, ffmpeg, ffprobe, images, music, tmp, encoder, out_path, fonts):
    W, H, fps = board['width'], board['height'], board['fps']
    unit = min(W, H)
    bar = letterbox_bar(W, H) if board['letterbox'] else 0
    shadow = max(1, round(unit * 0.003))
    args = [ffmpeg, '-hide_banner', '-y', '-nostats', '-progress', 'pipe:1',
            '-filter_complex_threads', str(os.cpu_count() or 4)]
    graph = []

    for i, s in enumerate(board['scenes']):
        n, D, k = s['frames'], s['duration'], s['intensity']
        src = images[i]
        if src:
            iw, ih = image_size(ffprobe, src)
        else:
            src = tmp / f'bg{i}.png'
            iw, ih = background_png(src, s['background'], W, H)
        motion = s['motion'] if src.parent != tmp else 'static'

        if motion in ('zoom-in', 'zoom-out'):
            args += ['-i', str(src)]
            c = max(W / iw, H / ih)
            vw, vh = min(iw, W / c), min(ih, H / c)
            x0, y0 = (iw - vw) / 2, (ih - vh) / 2
            f = max(1.0, min(3.0, 6000 / W))
            zt = f'(on/{n})' if motion == 'zoom-in' else f'(1-on/{n})'
            fx, fy = s['focus'] or (None, None)
            fxc = clamp((fx * iw - x0) / vw) if fx is not None else 0.5
            fyc = clamp((fy * ih - y0) / vh) if fy is not None else 0.5
            chain = (f"[{i}:v]trim=end_frame=1,crop={int(vw)}:{int(vh)}:{int(x0)}:{int(y0)},"
                     f"scale={even(W * f)}:{even(H * f)}:flags=lanczos,setsar=1,"
                     f"zoompan=z='1+{k}*{zt}'"
                     f":x='clip(iw/2+({fxc:.4f}*iw-iw/2)*{zt}-iw/zoom/2,0,iw-iw/zoom)'"
                     f":y='clip(ih/2+({fyc:.4f}*ih-ih/2)*{zt}-ih/zoom/2,0,ih-ih/zoom)'"
                     f":d={n}:s={W}x{H}:fps={fps}")
        else:
            args += ['-i', str(src)]
            ss = 2 if motion != 'static' and W * H <= 2560 * 1440 else 1
            WW, HH = W * ss, H * ss
            z = 1 + k if motion.startswith('pan') else 1
            scale = max(WW / iw, HH / ih) * z
            sw, sh = max(WW, math.ceil(iw * scale)), max(HH, math.ceil(ih * scale))
            u = f'min(t/{D:.6f},1)'
            xs = {'pan-right': f'(in_w-out_w)*{u}', 'pan-left': f'(in_w-out_w)*(1-{u})'}.get(motion, '(in_w-out_w)/2')
            ys = {'pan-down': f'(in_h-out_h)*{u}', 'pan-up': f'(in_h-out_h)*(1-{u})'}.get(motion, '(in_h-out_h)/2')
            # Scale the still once, then repeat that frame; scaling inside the loop would redo it per frame.
            chain = (f"[{i}:v]trim=end_frame=1,scale={sw}:{sh}:flags=lanczos,format=rgb24,"
                     f"loop=loop={n - 1}:size=1:start=0,settb=1/{fps},setpts=N")
            if motion == 'static':
                chain += f",crop={WW}:{HH}"
            else:
                chain += f",crop={WW}:{HH}:x='{xs}':y='{ys}'"
            if ss > 1:
                chain += f',scale={W}:{H}:flags=area'

        parts = [chain, f'fps={fps}', f'trim=end_frame={n}', 'setpts=PTS-STARTPTS', 'format=yuv420p', 'setsar=1', f'settb=1/{fps}']
        if s['title']:
            size = round(unit * 0.075)
            lines = wrap_text(s['title'], size, W * 0.82)
            lh = size * 1.2
            top = (H - len(lines) * lh) / 2
            parts += text_filters(lines, size, [top + lh * (j + 0.5) for j in range(len(lines))],
                                  fonts['bold'], tmp, f't{i}', D, shadow)
        if s['caption']:
            size = round(unit * 0.042)
            lines = wrap_text(s['caption'], size, W * 0.8)
            lh = size * 1.35
            last = H - unit * 0.08 - bar
            parts += text_filters(lines, size, [last - lh * (len(lines) - 1 - j) for j in range(len(lines))],
                                  fonts['regular'], tmp, f'c{i}', D, shadow)
        graph.append(','.join(parts) + f'[s{i}]')

    cur, total = 's0', board['scenes'][0]['duration']
    for i, s in enumerate(board['scenes'][1:], start=1):
        T = s['transition']
        if T > 0:
            kind = 'fadeblack' if s['transitionType'] == 'fade-black' else 'fade'
            graph.append(f'[{cur}][s{i}]xfade=transition={kind}:duration={T:.6f}:offset={total - T:.6f}[x{i}]')
        else:
            graph.append(f'[{cur}][s{i}]concat=n=2:v=1:a=0[x{i}]')
        cur, total = f'x{i}', total + s['duration'] - T

    post = [f"eq=contrast={board['contrast']:.3f}:saturation={board['saturation']:.3f}"]
    if board['warmth']:
        w = clamp(board['warmth'] * 0.12, -0.5, 0.5)
        post.append(f'colorbalance=rs={w / 2:.3f}:rm={w:.3f}:rh={w / 2:.3f}:bs={-w / 2:.3f}:bm={-w:.3f}:bh={-w / 2:.3f}')
    if board['vignette'] > 0:
        post.append(f"vignette=angle={min(1.5, board['vignette'] * 1.8):.3f}")
    if board['grain'] > 0:
        post.append(f"noise=alls={max(1, round(board['grain'] * 120))}:allf=t")
    if bar:
        post.append(f'drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill')
        post.append(f'drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill')
    if board['fadeIn'] > 0:
        post.append(f"fade=t=in:st=0:d={board['fadeIn']:.3f}")
    if board['fadeOut'] > 0:
        post.append(f"fade=t=out:st={max(0.0, total - board['fadeOut']):.3f}:d={board['fadeOut']:.3f}")
    post.append('format=yuv420p')
    graph.append(f'[{cur}]' + ','.join(post) + '[vout]')

    maps = ['-map', '[vout]']
    if music:
        m = len(board['scenes'])
        args += ['-stream_loop', '-1', '-i', str(music)]
        fo = max(board['fadeOut'], 1.5)
        graph.append(f"[{m}:a]atrim=0:{total:.6f},asetpts=PTS-STARTPTS,volume={board['musicVolume']:.3f},"
                     f"afade=t=out:st={max(0.0, total - fo):.6f}:d={fo:.6f},aresample=48000[aout]")
        maps += ['-map', '[aout]', '-c:a', 'aac', '-b:a', '192k']

    args += ['-filter_complex', ';'.join(graph), *maps, *encoder_args(encoder, board['mbps']),
             '-r', str(fps), '-t', f'{total:.6f}', '-movflags', '+faststart', str(out_path)]
    return args, total


def run_ffmpeg(args, total, cwd, log_path):
    start, shown = time.time(), -10
    with open(log_path, 'w', encoding='utf-8', errors='replace') as log:
        proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=log, text=True)
        for line in proc.stdout:
            key, _, val = line.strip().partition('=')
            if key in ('out_time_us', 'out_time_ms') and val.isdigit():
                pct = int(min(100, int(val) / 1e6 / total * 100))
                if pct >= shown + 10:
                    shown = pct - pct % 10
                    print(f'  {shown}%', flush=True)
        proc.wait()
    return proc.returncode, time.time() - start


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    ap = argparse.ArgumentParser(description='Render a storyboard.json to MP4 with ffmpeg.')
    ap.add_argument('storyboard')
    ap.add_argument('--root', default='.', help='folder searched for images and music (default: current folder)')
    ap.add_argument('--out', help='output file (default: <root>/renders/<title>.mp4)')
    ap.add_argument('--quality', choices=list(QUALITY), help='override storyboard quality')
    ap.add_argument('--codec', choices=list(ENCODERS), default='h264')
    ap.add_argument('--font', help='font file for titles and captions')
    ap.add_argument('--dry-run', action='store_true', help='print the ffmpeg command without running it')
    a = ap.parse_args()

    ffmpeg, ffprobe = find_tool('ffmpeg'), find_tool('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RenderError('ffmpeg not found. Install it: Windows "winget install --id Gyan.FFmpeg -e", '
                          'macOS "brew install ffmpeg", Linux "sudo apt install ffmpeg"; then open a new terminal.')

    sb_path = Path(a.storyboard).resolve()
    root = Path(a.root).resolve()
    try:
        raw = json.loads(sb_path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError) as e:
        raise RenderError(f'cannot read storyboard: {e}')
    board = normalize(raw, a.quality)

    index = index_files(root)
    bases = [sb_path.parent, root]
    images, missing = [], []
    for s in board['scenes']:
        p = resolve(s['image'], bases, index) if s['image'] else None
        if s['image'] and not p:
            missing.append(s['image'])
        images.append(p)
    if missing:
        raise RenderError('images not found: ' + ', '.join(missing))
    music = resolve(board['music'], bases, index) if board['music'] else None
    if board['music'] and not music:
        print(f"Music not found, rendering without audio: {board['music']}")

    needs_text = any(s['title'] or s['caption'] for s in board['scenes'])
    slug = re.sub(r'[^\w.-]+', '-', board['title']).strip('-.') or 'video'
    out_path = Path(a.out).resolve() if a.out else root / 'renders' / f'{slug}.mp4'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = pick_encoder(ffmpeg, a.codec, board['mbps'])
    with tempfile.TemporaryDirectory(prefix='video-render-') as td:
        tmp = Path(td)
        fonts = {'bold': None, 'regular': None}
        if needs_text:
            for kind in fonts:
                f = find_font(kind, a.font)
                if f:
                    fonts[kind] = 'font_' + kind + f.suffix.lower()
                    shutil.copyfile(f, tmp / fonts[kind])
            if not fonts['bold'] and not fonts['regular']:
                print('No Korean-capable font found; using the ffmpeg default font (pass --font to choose one).')
            fonts['bold'] = fonts['bold'] or fonts['regular']
            fonts['regular'] = fonts['regular'] or fonts['bold']

        args, total = build_command(board, ffmpeg, ffprobe, images, music, tmp, encoder, out_path, fonts)
        print(f"{board['width']}x{board['height']} {board['fps']}fps, {len(board['scenes'])} scenes, "
              f"{total:.1f}s, encoder {encoder}")
        if a.dry_run:
            print(json.dumps(args, ensure_ascii=False, indent=1))
            return
        code, elapsed = run_ffmpeg(args, total, tmp, tmp / 'ffmpeg.log')
        if code != 0:
            tail = (tmp / 'ffmpeg.log').read_text(encoding='utf-8', errors='replace').splitlines()[-25:]
            raise RenderError('ffmpeg failed:\n' + '\n'.join(tail))

    size_mb = out_path.stat().st_size / 1e6
    print(f'Done: {out_path} ({size_mb:.1f} MB, {total:.1f}s video in {elapsed:.1f}s, {total / max(elapsed, 0.01):.1f}x realtime)')


if __name__ == '__main__':
    try:
        main()
    except RenderError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
