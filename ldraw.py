import math
import part_numbers

from dataclasses import dataclass, field
from typing import Optional

def stud_to_ldu(x):
    return x * 20

def y_rotation(angle):
    return [
            [math.cos(-angle), 0, -math.sin(-angle)],
            [0, 1, 0],
            [math.sin(-angle), 0, math.cos(-angle)],
    ]

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

@dataclass
class Type1Line:
    color: int
    x: float
    y: float
    z: float
    subfile: str
    rotation: list[list[float]] = field(default_factory=lambda: IDENTITY)

    def __str__(self):
        rot_str = ' '.join(f'{x:.2f}' for row in self.rotation for x in row)
        return f'1 {self.color} {stud_to_ldu(self.x)} {stud_to_ldu(self.y)} {stud_to_ldu(self.z)} {rot_str} {self.subfile}'

def gear(x, y, z, g, color=7):
    subfile = f'{part_numbers.gear(g)}.dat'
    return Type1Line(color=color, x=x, y=y, z=z, subfile=subfile)

def axle(x, y, z0, z1, color=0):
    z0, z1 = min(z0, z1), max(z0, z1)
    l = z1 - z0 + 1
    subfile = f'{part_numbers.axle(l)}.dat'
    return Type1Line(color=color, x=x, y=y, z=(z0+z1)/2, subfile=subfile,
                     rotation=y_rotation(math.pi/2))
