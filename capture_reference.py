"""Capture a reference face embedding for identity-gated wake.

Usage: python capture_reference.py [--device 0] [--out reference_face.npy]
"""
import argparse
import sys
import time

import cv2
import face_recognition
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0, help='V4L2 device index (default 0)')
    ap.add_argument('--out', default='reference_face.npy')
    ap.add_argument('--snapshot', default='reference_face.jpg')
    ap.add_argument('--countdown', type=int, default=3)
    ap.add_argument('--attempts', type=int, default=30)
    args = ap.parse_args()

    cam = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cam.isOpened():
        print(f"Failed to open /dev/video{args.device}", file=sys.stderr)
        sys.exit(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Look at the camera. Capturing in {args.countdown}...")
    for n in range(args.countdown, 0, -1):
        print(n)
        time.sleep(1)

    # Drain any stale buffered frames before capturing
    for _ in range(5):
        cam.read()

    for attempt in range(1, args.attempts + 1):
        ok, frame = cam.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model='hog')
        if len(locations) == 1:
            encoding = face_recognition.face_encodings(rgb, locations)[0]
            np.save(args.out, encoding)
            cv2.imwrite(args.snapshot, frame)
            top, right, bottom, left = locations[0]
            print(f"OK: saved encoding ({encoding.shape}) to {args.out}")
            print(f"OK: saved snapshot to {args.snapshot} (face box {left},{top}-{right},{bottom})")
            cam.release()
            return
        if len(locations) > 1:
            print(f"[{attempt}] {len(locations)} faces detected — only you should be in frame")
        else:
            print(f"[{attempt}] no face detected")
        time.sleep(0.2)

    print("Failed to capture a single clear face. Try again with better lighting.", file=sys.stderr)
    cam.release()
    sys.exit(2)


if __name__ == '__main__':
    main()
