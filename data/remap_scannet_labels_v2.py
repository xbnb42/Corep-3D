#!/usr/bin/env python3
import argparse
import numpy as np

BUILTIN = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8,
    10: 9, 11: 10, 12: 11, 14: 12, 16: 13, 24: 14, 28: 15,
    33: 16, 34: 17, 36: 18, 39: 19,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    z = np.load(args.npz)
    data = {k: z[k] for k in z.files}
    raw = z['labels'].astype(np.int64)
    max_id = max(int(raw.max()), max(BUILTIN))
    mapping = np.full(max_id + 1, -1, dtype=np.int64)
    for rid, sid in BUILTIN.items():
        mapping[rid] = sid

    labels20 = np.full_like(raw, -1, dtype=np.int64)
    valid = (raw >= 0) & (raw < len(mapping))
    labels20[valid] = mapping[raw[valid]]
    data['labels_raw'] = raw
    data['labels20'] = labels20
    np.savez_compressed(args.out, **data)

    print('using built-in ScanNet20 mapping')
    print('saved:', args.out)
    print('raw ids:', sorted(np.unique(raw).tolist())[:60])
    print('mapped classes:', sorted(np.unique(labels20[labels20 >= 0]).tolist()))
    print('ignored points:', int((labels20 < 0).sum()), '/', len(labels20))


if __name__ == '__main__':
    main()
