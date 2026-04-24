#!/usr/bin/env python3
"""
Merge the Q2c visual-SLAM and Q3e LiDAR-SLAM demo MP4s (coursework Q3e asks to
merge with the visual-SLAM video). Requires ffmpeg on PATH.

Defaults match the repo output names next to coursework_deliverables/.

  SLAM_VIDEO_Q2     – input visual SLAM clip (default: ../COMP0222_CW2_GRP_32_Visual_SLAM.mp4)
  SLAM_VIDEO_Q3     – input LiDAR clip      (default: ../COMP0222_CW2_GRP_32_LiDAR_SLAM.mp4)
  SLAM_VIDEO_MERGED – output              (default: ../COMP0222_CW2_GRP_32.mp4)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_DEFAULT_DIR = os.path.abspath(os.path.join(_ROOT, '..'))


def main() -> None:
    q2 = os.environ.get('SLAM_VIDEO_Q2',
                        os.path.join(_DEFAULT_DIR, 'COMP0222_CW2_GRP_32_Visual_SLAM.mp4'))
    q3 = os.environ.get('SLAM_VIDEO_Q3',
                        os.path.join(_DEFAULT_DIR, 'COMP0222_CW2_GRP_32_LiDAR_SLAM.mp4'))
    out = os.environ.get('SLAM_VIDEO_MERGED',
                          os.path.join(_DEFAULT_DIR, 'COMP0222_CW2_GRP_32.mp4'))

    for p, label in ((q2, 'SLAM_VIDEO_Q2'), (q3, 'SLAM_VIDEO_Q3')):
        if not os.path.isfile(p):
            print(f'ERROR: missing {label}: {p}', file=sys.stderr)
            sys.exit(2)

    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as lst:
        # ffmpeg concat demuxer: file list
        lst.write(f"file '{q2}'\n")
        lst.write(f"file '{q3}'\n")
        list_path = lst.name

    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
        '-c', 'copy', out,
    ]
    print(' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(list_path)
    print(f'Merged -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
