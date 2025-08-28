import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 随机数生成器
rng = np.random.default_rng(42)


def add_noise(xy, noise=0.02):
    """给点集加高斯噪声"""
    return xy + rng.normal(scale=noise, size=xy.shape) if noise > 0 else xy


# 一些常见的参数方程

def archimedean_spiral(n=2000, turns=3.5, a=0.0, b=0.20, noise=0.02):
    theta = np.linspace(0, 2 * np.pi * turns, n)
    r = a + b * theta
    x, y = r * np.cos(theta), r * np.sin(theta)
    return add_noise(np.stack([x, y], axis=1), noise)


def logarithmic_spiral(n=2000, turns=3.0, a=0.08, b=0.15, noise=0.02):
    theta = np.linspace(0, 2 * np.pi * turns, n)
    r = a * np.exp(b * theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    return add_noise(np.stack([x, y], axis=1), noise)


def rose_curve(n=2000, k=5, noise=0.02):
    theta = np.linspace(0, 2 * np.pi, n)
    r = np.cos(k * theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    return add_noise(np.stack([x, y], axis=1), noise)


def heart_curve(n=2500, noise=0.03):
    t = np.linspace(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x, y = x / 18.0, y / 18.0  # 缩放一下大小
    return add_noise(np.stack([x, y], axis=1), noise)


def lissajous(n=2000, a=3, b=2, delta=np.pi / 2, noise=0.01):
    t = np.linspace(0, 2 * np.pi, n)
    x, y = np.sin(a * t + delta), np.sin(b * t)
    return add_noise(np.stack([x, y], axis=1), noise)


def lemniscate(n=2000, a=1.0, noise=0.02):
    t = np.linspace(-np.pi / 4 + 1e-3, np.pi / 4 - 1e-3, n // 2)
    r = a * np.sqrt(2 * np.cos(2 * t))
    x1, y1 = r * np.cos(t), r * np.sin(t)
    x2, y2 = -x1, -y1
    x, y = np.concatenate([x1, x2]), np.concatenate([y1, y2])
    return add_noise(np.stack([x, y], axis=1), noise)


def two_moons(n=2000, noise=0.03, gap=0.10, radius=1.0):
    n1, n2 = n // 2, n - n // 2
    angles1, angles2 = rng.uniform(0, np.pi, n1), rng.uniform(0, np.pi, n2)
    x1, y1 = radius * np.cos(angles1), radius * np.sin(angles1)
    x2, y2 = radius * np.cos(angles2) + radius, -radius * np.sin(angles2) - gap
    x, y = np.concatenate([x1, x2]), np.concatenate([y1, y2])
    return add_noise(np.stack([x, y], axis=1), noise)


def circle(n=2000, noise=0.02, radius=1.0):
    t = rng.uniform(0, 2 * np.pi, n)
    x, y = radius * np.cos(t), radius * np.sin(t)
    return add_noise(np.stack([x, y], axis=1), noise)





