#!/usr/bin/env python3
"""SparseConv (MinkowskiEngine) backbone for unsupervised 3D segmentation.

A standard sparse-conv ResUNet that encodes xyz+rgb into per-point embeddings.
Target env: Python 3.8 + MinkowskiEngine 0.5.x (3090 instance).

Usage sketch:
    coords, feats = voxelize(xyz, rgb, voxel_size=0.05)
    model = MinkUNet(in_channels=3, out_channels=384)
    per_point_feat = per_point_feature(model, sparse, inv)
"""
import numpy as np
import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ImportError:
    raise ImportError("MinkowskiEngine not installed; this backbone needs Python3.8 + MinkowskiEngine 0.5.x")


def voxelize(xyz, rgb=None, voxel_size=0.05, device='cpu'):
    """Point cloud (N,3) + optional rgb (N,3) -> sparse tensor input.
    Returns (sparse_tensor, inverse_map) where inverse_map maps unique voxel
    indices back to the original point indices (for pointwise feature output).
    """
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    # unique voxels
    _, idx, inv = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    # per-voxel feature: mean of rgb (or a constant 1 if no rgb)
    if rgb is not None:
        feat = np.zeros((len(idx), rgb.shape[1]), dtype=np.float32)
        np.add.at(feat, inv, rgb)
        cnt = np.bincount(inv, minlength=len(idx)).astype(np.float32)
        feat /= np.maximum(cnt, 1)[:, None]
    else:
        feat = np.ones((len(idx), 1), dtype=np.float32)

    # MinkowskiEngine 0.5.x: coordinates must be (N, D+1), first column = batch index.
    coords_uniq = coords[idx]
    batch_idx = np.zeros((coords_uniq.shape[0], 1), dtype=np.int32)
    coords_t = torch.from_numpy(np.hstack([batch_idx, coords_uniq])).int().to(device)
    feat_t = torch.from_numpy(feat).to(device)
    sin = ME.SparseTensor(features=feat_t, coordinates=coords_t)
    return sin, inv


class BasicBlock(nn.Module):
    """Two sparse convs with residual connection (ResNet-style)."""
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, dimension=3):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=3, stride=stride,
                                             dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(out_ch)
        self.conv2 = ME.MinkowskiConvolution(out_ch, out_ch, kernel_size=3, stride=1,
                                             dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(out_ch)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=1, stride=stride, dimension=dimension),
                ME.MinkowskiBatchNorm(out_ch),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


class DecoderStage(nn.Module):
    """upsample -> concat skip -> BasicBlock."""
    def __init__(self, in_ch, out_ch, dimension=3):
        super().__init__()
        self.up = ME.MinkowskiConvolutionTranspose(in_ch, out_ch, kernel_size=2, stride=2, dimension=dimension)
        self.block = BasicBlock(out_ch * 2, out_ch, dimension=dimension)

    def forward(self, x, skip):
        x = self.up(x)
        x = ME.cat(x, skip)
        return self.block(x)


class MinkUNet(nn.Module):
    """Sparse-conv ResUNet. channels = list of encoder widths (last is bottleneck)."""
    def __init__(self, in_channels=3, out_channels=384, channels=(32, 64, 128, 256, 512), dimension=3):
        super().__init__()
        self.dimension = dimension
        self.in_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, channels[0], kernel_size=3, stride=1, dimension=dimension),
            ME.MinkowskiBatchNorm(channels[0]),
            ME.MinkowskiReLU(inplace=True),
        )
        # encoder
        self.enc = nn.ModuleList()
        prev = channels[0]
        for c in channels[1:]:
            self.enc.append(BasicBlock(prev, c, stride=2, dimension=dimension))
            prev = c
        # decoder (symmetric)
        self.dec = nn.ModuleList()
        rev = list(reversed(channels))          # [C4, C3, C2, C1, C0]
        for i in range(len(rev) - 1):
            self.dec.append(DecoderStage(rev[i], rev[i + 1], dimension=dimension))
        self.final = ME.MinkowskiConvolution(channels[0], out_channels, kernel_size=1, dimension=dimension)

    def forward(self, sparse):
        skips = []
        x = self.in_conv(sparse)
        skips.append(x)
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
        skips = skips[:-1]                      # drop bottleneck; keep [C0..C3]
        for dec in self.dec:
            x = dec(x, skips.pop())
        return self.final(x)


def per_point_feature(model, sparse, inv_map):
    """Forward + scatter voxel features back to per-point features."""
    out = model(sparse)                         # sparse tensor, features at unique voxels
    feat_voxel = out.F                          # (n_voxel, out_ch)
    return feat_voxel[torch.from_numpy(inv_map).to(feat_voxel.device)]


if __name__ == '__main__':
    # smoke test with random points
    rng = np.random.RandomState(0)
    xyz = rng.rand(1000, 3) * 5
    rgb = rng.rand(1000, 3)
    sparse, inv = voxelize(xyz, rgb, voxel_size=0.05)
    model = MinkUNet(in_channels=3, out_channels=128)
    feat = per_point_feature(model, sparse, inv)
    print('smoke test: per-point feature shape =', tuple(feat.shape))
