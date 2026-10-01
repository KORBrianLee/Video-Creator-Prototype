#!/usr/bin/env python3
"""Settings for Video Creator: where files are stored and which hardware renders them.

On the first run a small window asks the user to choose the environment; the answers are saved in
video-creator.config.json next to the scripts. Run `python setup.py` to change them later.

    python vc_config.py --show     print the current settings as JSON (never opens a window)
"""
import argparse
import json
import os
import sys
from pathlib import Path

CONFIG_PATH = Path(os.environ.get('VIDEO_CREATOR_CONFIG') or Path(__file__).resolve().with_name('video-creator.config.json'))
SUBFOLDERS = ('projects', 'assets', 'renders', 'tmp')
RENDER_MODES = [('auto', '자동: GPU 우선, 안 되면 CPU'), ('gpu', 'GPU만 사용'), ('cpu', 'CPU만 사용')]
QUALITIES = ['720p', '1080p', '1440p', '4k']
INTERP_MODES = [('gpu', 'GPU 크로스페이드 (빠름)'), ('mci', '움직임 보간 (느리지만 더 자연스러움)')]


def default_storage():
    if os.name == 'nt' and Path('D:/').exists():
        return str(Path('D:/VideoCreator'))
    return str(Path.home() / 'VideoCreator')


def defaults():
    return {'version': 1, 'storage_dir': default_storage(), 'render_mode': 'auto', 'encoder': 'auto',
            'quality': '1080p', 'interp_mode': 'gpu', 'install_skill': False}


def load():
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return {**defaults(), **data} if isinstance(data, dict) else None


def save(cfg):
    for sub in SUBFOLDERS:
        (Path(cfg['storage_dir']) / sub).mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


def temp_dir(cfg):
    p = Path(cfg['storage_dir']) / 'tmp'
    p.mkdir(parents=True, exist_ok=True)
    # ffmpeg and other child processes inherit these, so scratch files stay off the system drive too.
    for var in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[var] = str(p)
    return str(p)


def on_system_drive(path):
    system = os.environ.get('SystemDrive', 'C:').rstrip('\\/').upper()
    drive = os.path.splitdrive(str(path))[0].upper()
    return os.name == 'nt' and drive == system


def detect_encoders():
    try:
        import video_render as vr
        ffmpeg = vr.find_tool('ffmpeg')
        return vr.detect_encoders(ffmpeg) if ffmpeg else []
    except Exception:
        return []


def ask_popup(encoders, cfg, timeout=None):
    """Show the setup window. Returns ('ok', cfg), ('cancel', None) or ('nogui', None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        root = tk.Tk()
    except Exception:
        return 'nogui', None

    import video_render as vr
    hw = [e for e in encoders if not vr.is_software(e)]
    state = {'result': None}
    root.title('Video Creator 설정')
    root.attributes('-topmost', True)
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=16)
    frm.grid()

    ttk.Label(frm, text='실행 환경을 선택하세요', font=('Malgun Gothic', 13, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w')
    ttk.Label(frm, text='나중에 setup.py를 다시 실행하면 바꿀 수 있습니다.', foreground='#666').grid(row=1, column=0, columnspan=3, sticky='w', pady=(0, 10))

    ttk.Label(frm, text='렌더링 방식').grid(row=2, column=0, sticky='nw', pady=4)
    mode = tk.StringVar(value=cfg['render_mode'])
    box = ttk.Frame(frm)
    box.grid(row=2, column=1, columnspan=2, sticky='w')
    for value, label in RENDER_MODES:
        ttk.Radiobutton(box, text=label, value=value, variable=mode).pack(anchor='w')
    gpu_text = ('감지된 GPU 인코더: ' + ', '.join(hw)) if hw else '이 PC에서는 사용 가능한 GPU 인코더를 찾지 못했습니다. CPU로 처리됩니다.'
    ttk.Label(frm, text=gpu_text, foreground='#0a7' if hw else '#b60').grid(row=3, column=1, columnspan=2, sticky='w', pady=(0, 6))

    ttk.Label(frm, text='인코더').grid(row=4, column=0, sticky='w', pady=4)
    enc_labels = ['자동'] + encoders
    enc = tk.StringVar(value=cfg['encoder'] if cfg['encoder'] in encoders else '자동')
    ttk.Combobox(frm, textvariable=enc, values=enc_labels, state='readonly', width=26).grid(row=4, column=1, columnspan=2, sticky='w')

    ttk.Label(frm, text='저장 위치').grid(row=5, column=0, sticky='w', pady=4)
    folder = tk.StringVar(value=cfg['storage_dir'])
    ttk.Entry(frm, textvariable=folder, width=34).grid(row=5, column=1, sticky='w')
    ttk.Button(frm, text='찾아보기', command=lambda: folder.set(os.path.normpath(filedialog.askdirectory(initialdir=folder.get() or '/') or folder.get()))).grid(row=5, column=2, padx=(6, 0))
    warn = ttk.Label(frm, text='', foreground='#c00', wraplength=380)
    warn.grid(row=6, column=1, columnspan=2, sticky='w')

    def refresh_warning(*_):
        warn.config(text='C 드라이브입니다. 영상 파일은 용량이 커서 D 드라이브를 권장합니다.' if on_system_drive(folder.get()) else '')

    folder.trace_add('write', refresh_warning)
    refresh_warning()
    ttk.Label(frm, text='이미지·스토리보드·결과 영상·임시 파일이 모두 이 폴더 아래에 저장됩니다.', foreground='#666').grid(row=7, column=1, columnspan=2, sticky='w', pady=(0, 6))

    ttk.Label(frm, text='기본 화질').grid(row=8, column=0, sticky='w', pady=4)
    quality = tk.StringVar(value=cfg['quality'])
    ttk.Combobox(frm, textvariable=quality, values=QUALITIES, state='readonly', width=10).grid(row=8, column=1, sticky='w')

    ttk.Label(frm, text='프레임 보간').grid(row=9, column=0, sticky='w', pady=4)
    interp = tk.StringVar(value=dict(INTERP_MODES)[cfg['interp_mode']])
    ttk.Combobox(frm, textvariable=interp, values=[l for _, l in INTERP_MODES], state='readonly', width=34).grid(row=9, column=1, columnspan=2, sticky='w')

    skill = tk.BooleanVar(value=cfg['install_skill'])
    ttk.Checkbutton(frm, text='Cursor 스킬을 설치해 모든 프로젝트에서 쓰기', variable=skill).grid(row=10, column=0, columnspan=3, sticky='w', pady=(8, 0))

    def collect():
        label_to_interp = {l: v for v, l in INTERP_MODES}
        return {**cfg, 'render_mode': mode.get(), 'encoder': 'auto' if enc.get() == '자동' else enc.get(),
                'storage_dir': os.path.normpath(folder.get().strip() or cfg['storage_dir']), 'quality': quality.get(),
                'interp_mode': label_to_interp.get(interp.get(), 'gpu'), 'install_skill': skill.get()}

    def ok():
        chosen = collect()
        if on_system_drive(chosen['storage_dir']) and not messagebox.askyesno(
                '확인', 'C 드라이브에 저장하시겠습니까?\n영상 작업은 수 GB를 쓸 수 있습니다.', parent=root):
            return
        state['result'] = ('ok', chosen)
        root.destroy()

    def use_defaults():
        state['result'] = ('ok', {**defaults(), 'install_skill': skill.get()})
        root.destroy()

    def cancel():
        state['result'] = ('cancel', None)
        root.destroy()

    buttons = ttk.Frame(frm)
    buttons.grid(row=11, column=0, columnspan=3, sticky='e', pady=(14, 0))
    ttk.Button(buttons, text='저장하고 시작', command=ok).pack(side='right')
    ttk.Button(buttons, text='기본값으로 시작', command=use_defaults).pack(side='right', padx=6)
    ttk.Button(buttons, text='취소', command=cancel).pack(side='right')
    root.protocol('WM_DELETE_WINDOW', cancel)
    if timeout:
        root.after(int(timeout * 1000), use_defaults)

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 3
    root.geometry(f'+{x}+{y}')
    root.mainloop()
    return state['result'] or ('cancel', None)


def ask_console(encoders, cfg):
    def pick(prompt, options, default):
        print(prompt)
        for i, (value, label) in enumerate(options, 1):
            print(f'  {i}) {label}' + ('  [기본]' if value == default else ''))
        raw = input('번호 (Enter = 기본): ').strip()
        return options[int(raw) - 1][0] if raw.isdigit() and 1 <= int(raw) <= len(options) else default

    print('\nVideo Creator 설정 (팝업을 쓸 수 없어 터미널로 묻습니다)')
    print('감지된 인코더:', ', '.join(encoders) or '없음')
    chosen = dict(cfg)
    chosen['render_mode'] = pick('렌더링 방식', RENDER_MODES, cfg['render_mode'])
    folder = input(f"저장 위치 (Enter = {cfg['storage_dir']}): ").strip()
    chosen['storage_dir'] = os.path.normpath(folder) if folder else cfg['storage_dir']
    chosen['quality'] = pick('기본 화질', [(q, q) for q in QUALITIES], cfg['quality'])
    return chosen


def ensure_config(force=False, interactive=True, timeout=None):
    """Return the settings, asking the user the first time (or when force=True)."""
    current = load()
    if current and not force:
        return current
    base = current or defaults()
    if not interactive:
        return base
    encoders = detect_encoders()
    status, chosen = ask_popup(encoders, base, timeout)
    if status == 'nogui':
        if not (sys.stdin and sys.stdin.isatty()):
            for sub in SUBFOLDERS:
                (Path(base['storage_dir']) / sub).mkdir(parents=True, exist_ok=True)
            print('설정 창을 열 수 없는 환경이라 기본값을 사용합니다 (저장 위치: ' + base['storage_dir'] + ').\n'
                  '직접 선택하려면 터미널에서 setup.py를 실행하세요.', file=sys.stderr)
            return base
        chosen = ask_console(encoders, base)
    elif status == 'cancel':
        raise SystemExit('설정이 취소되었습니다. 다시 실행하면 설정 창이 열립니다.')
    save(chosen)
    print(f"설정 저장: {CONFIG_PATH}\n  저장 위치: {chosen['storage_dir']}  /  렌더링: {chosen['render_mode']}  /  인코더: {chosen['encoder']}")
    return chosen


def main():
    ap = argparse.ArgumentParser(description='Show Video Creator settings.')
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    cfg = load()
    print(json.dumps({'configured': cfg is not None, **(cfg or defaults()), 'config_file': str(CONFIG_PATH)},
                     ensure_ascii=False, indent=2))
    if not a.show:
        print('\n설정을 바꾸려면: python setup.py', file=sys.stderr)


if __name__ == '__main__':
    main()
