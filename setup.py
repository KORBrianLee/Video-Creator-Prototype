#!/usr/bin/env python3
"""Install check and first-run setup for Video Creator.

Checks Python and ffmpeg, opens the environment window (GPU/CPU, storage folder, quality), saves the
choices, and can copy the Cursor skill into your user skills folder.

    python setup.py
"""
import shutil
import sys
from pathlib import Path

import vc_config
import video_render as vr

HERE = Path(__file__).resolve().parent


def install_skill():
    src = HERE / '.cursor' / 'skills' / 'cursor-video'
    dst = Path.home() / '.cursor' / 'skills' / 'cursor-video'
    if not src.is_dir():
        print('스킬 폴더를 찾지 못했습니다:', src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print('스킬 설치:', dst, '(Cursor를 다시 시작하면 적용됩니다)')


def main():
    if sys.version_info < (3, 8):
        sys.exit('Python 3.8 이상이 필요합니다.')
    print('Video Creator 설치 확인')
    print('  Python', sys.version.split()[0])
    ffmpeg = vr.find_tool('ffmpeg')
    if ffmpeg:
        print('  ffmpeg  ', ffmpeg)
    else:
        print('  ffmpeg 없음. 설치 후 새 터미널에서 다시 실행하세요.')
        print('    Windows: winget install --id Gyan.FFmpeg -e')
        print('    macOS:   brew install ffmpeg')
        print('    Linux:   sudo apt install ffmpeg')

    cfg = vc_config.ensure_config(force=True)
    if cfg.get('install_skill'):
        install_skill()
    print('\n준비 완료. 데모 영상을 만들려면:')
    print('  python video_render.py video-projects/demo/storyboard.json')


if __name__ == '__main__':
    try:
        main()
    except vr.RenderError as e:
        sys.exit(f'Error: {e}')
