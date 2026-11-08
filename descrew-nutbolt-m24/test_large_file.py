"""
Large test file to verify Write tool functionality.
Contains various Python patterns and structures.
"""

import os
import sys
import math
import random
import itertools
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> "Vector2D":
        n = self.norm()
        if n == 0:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / n, self.y / n)

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.4f}, {self.y:.4f})"


@dataclass
class BoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, point: Vector2D) -> bool:
        return (self.min_x <= point.x <= self.max_x and
                self.min_y <= point.y <= self.max_y)

    def intersects(self, other: "BoundingBox") -> bool:
        return not (other.min_x > self.max_x or
                    other.max_x < self.min_x or
                    other.min_y > self.max_y or
                    other.max_y < self.min_y)


# ─── Abstract Base ─────────────────────────────────────────────────────────────

class Shape(ABC):
    def __init__(self, color: str = "white"):
        self.color = color

    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    @abstractmethod
    def bounding_box(self) -> BoundingBox:
        pass

    def describe(self) -> str:
        return (f"{self.__class__.__name__}(color={self.color}, "
                f"area={self.area():.4f}, perimeter={self.perimeter():.4f})")


class Circle(Shape):
    def __init__(self, center: Vector2D, radius: float, color: str = "white"):
        super().__init__(color)
        self.center = center
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.x + self.radius,
            self.center.y + self.radius,
        )


class Rectangle(Shape):
    def __init__(self, origin: Vector2D, width: float, height: float, color: str = "white"):
        super().__init__(color)
        self.origin = origin
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(
            self.origin.x,
            self.origin.y,
            self.origin.x + self.width,
            self.origin.y + self.height,
        )


class Triangle(Shape):
    def __init__(self, a: Vector2D, b: Vector2D, c: Vector2D, color: str = "white"):
        super().__init__(color)
        self.a = a
        self.b = b
        self.c = c

    def area(self) -> float:
        # Shoelace formula
        return abs(
            (self.b.x - self.a.x) * (self.c.y - self.a.y) -
            (self.c.x - self.a.x) * (self.b.y - self.a.y)
        ) / 2

    def perimeter(self) -> float:
        return ((self.b - self.a).norm() +
                (self.c - self.b).norm() +
                (self.a - self.c).norm())

    def bounding_box(self) -> BoundingBox:
        xs = [self.a.x, self.b.x, self.c.x]
        ys = [self.a.y, self.b.y, self.c.y]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))


# ─── Algorithms ────────────────────────────────────────────────────────────────

def quicksort(arr: List[float]) -> List[float]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def binary_search(arr: List[float], target: float) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def fibonacci_gen(n: int) -> Generator[int, None, None]:
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


def sieve_of_eratosthenes(limit: int) -> List[int]:
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "Incompatible matrix dimensions"
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C


# ─── Simple Statistics ─────────────────────────────────────────────────────────

def mean(data: List[float]) -> float:
    return sum(data) / len(data)


def variance(data: List[float]) -> float:
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std_dev(data: List[float]) -> float:
    return math.sqrt(variance(data))


def median(data: List[float]) -> float:
    s = sorted(data)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def percentile(data: List[float], p: float) -> float:
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ─── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Shapes
    c = Circle(Vector2D(0, 0), 5.0, color="red")
    r = Rectangle(Vector2D(1, 1), 4.0, 3.0, color="blue")
    t = Triangle(Vector2D(0, 0), Vector2D(4, 0), Vector2D(2, 3), color="green")

    for shape in [c, r, t]:
        print(shape.describe())
        print("  BBox:", shape.bounding_box())

    # Sorting & search
    data = [random.uniform(0, 100) for _ in range(20)]
    sorted_data = quicksort(data)
    target = sorted_data[10]
    idx = binary_search(sorted_data, target)
    print(f"\nBinary search for {target:.2f}: index {idx}")

    # Primes
    primes = sieve_of_eratosthenes(50)
    print(f"\nPrimes up to 50: {primes}")

    # Fibonacci
    fibs = list(fibonacci_gen(10))
    print(f"First 10 Fibonacci: {fibs}")

    # Stats
    sample = [random.gauss(50, 10) for _ in range(1000)]
    print(f"\nStats on 1000 samples:")
    print(f"  mean={mean(sample):.2f}, std={std_dev(sample):.2f}")
    print(f"  median={median(sample):.2f}")
    print(f"  p25={percentile(sample, 25):.2f}, p75={percentile(sample, 75):.2f}")

