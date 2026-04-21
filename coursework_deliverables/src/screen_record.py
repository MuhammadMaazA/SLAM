#!/usr/bin/env python3
"""
Screen recorder using xwd window capture.
Runs a display script, finds its window by title, captures at ~15fps → MP4.
Usage: python3 screen_record.py q2c | q3e
"""
import os, sys, struct, time, subprocess
import numpy as np
import cv2

SRC_DIR = os.path.dirname(__file__)

CONFIGS = {
    'q2c': {
        'script':   os.path.join(SRC_DIR, 'live_q2c_display.py'),
        'win_name': 'COMP0222 CW2 - ORB-SLAM2 Monocular SLAM',
        'out':      '/home/mmaaz/SLAM/COMP0222_CW2_GRP_1_Visual_SLAM.mp4',
        'fps':      15,
        'max_sec':  70,
    },
    'q3e': {
        'script':   os.path.join(SRC_DIR, 'live_q3e_display.py'),
        'win_name': 'COMP0222 CW2 - LiDAR SLAM Real-Time Mapping',
        'out':      '/home/mmaaz/SLAM/COMP0222_CW2_GRP_1_LiDAR_SLAM.mp4',
        'fps':      10,
        'max_sec':  50,
    },
}


def xwd_capture_by_id(win_id: str) -> np.ndarray | None:
    env = {**os.environ, 'DISPLAY': ':0'}
    r = subprocess.run(['xwd', '-id', win_id, '-silent'],
                       capture_output=True, env=env)
    d = r.stdout
    if len(d) < 200:
        return None
    def u32(o): return struct.unpack_from('>I', d, o)[0]
    hdr = u32(0); w = u32(16); h = u32(20)
    bpl = u32(48); nc = u32(76)
    if w == 0 or h == 0:
        return None
    px_start = hdr + nc * 12
    needed = bpl * h
    if len(d) < px_start + needed:
        return None
    px = np.frombuffer(d[px_start:px_start + needed], dtype=np.uint8)
    return px.reshape(h, bpl)[:, :w * 4].reshape(h, w, 4)[:, :, :3]


def find_window(name: str, timeout=20) -> str | None:
    env = {**os.environ, 'DISPLAY': ':0'}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ['xwininfo', '-name', name], env=env,
                stderr=subprocess.DEVNULL, text=True)
            for line in out.splitlines():
                if 'Window id:' in line:
                    return line.split()[3]
        except subprocess.CalledProcessError:
            pass
        time.sleep(0.5)
    return None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'q2c'
    cfg  = CONFIGS[mode]
    env  = {**os.environ, 'DISPLAY': ':0'}

    print(f"Starting {mode} display script...")
    proc = subprocess.Popen(
        ['python3', cfg['script']],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("Waiting for window...")
    win_id = find_window(cfg['win_name'])
    if not win_id:
        print("ERROR: window never appeared"); proc.terminate(); return

    print(f"Found window {win_id}. Starting capture at {cfg['fps']}fps...")
    # grab a reference frame for dimensions
    ref = None
    for _ in range(10):
        ref = xwd_capture_by_id(win_id)
        if ref is not None: break
        time.sleep(0.2)
    if ref is None:
        print("ERROR: first capture failed"); proc.terminate(); return

    h, w = ref.shape[:2]
    # Round to nearest multiple of 16 for codec compatibility
    w16 = (w // 16) * 16
    h16 = (h // 16) * 16
    if w16 != w or h16 != h:
        ref = cv2.resize(ref, (w16, h16))
        w, h = w16, h16
    print(f"Capture size: {w}×{h}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(cfg['out'], fourcc, cfg['fps'], (w, h))

    interval = 1.0 / cfg['fps']
    deadline = time.time() + cfg['max_sec']
    nframes  = 0
    vw.write(ref); nframes += 1

    while time.time() < deadline and proc.poll() is None:
        t0  = time.time()
        img = xwd_capture_by_id(win_id)
        if img is None:
            break   # window closed
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        vw.write(img)
        nframes += 1
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)
        if nframes % (cfg['fps'] * 5) == 0:
            print(f"  {nframes} frames captured ({nframes/cfg['fps']:.0f}s)")

    vw.release()
    proc.terminate()
    try: proc.wait(timeout=3)
    except: proc.kill()

    size_mb = os.path.getsize(cfg['out']) / 1e6
    print(f"\nSaved: {cfg['out']}")
    print(f"  {nframes} frames @ {cfg['fps']}fps = {nframes/cfg['fps']:.0f}s | {size_mb:.1f}MB")


if __name__ == '__main__':
    main()
