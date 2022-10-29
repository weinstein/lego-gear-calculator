import functools
import itertools 
import math

from collections import defaultdict, deque
from dataclasses import dataclass, field
from memoize import memoize_generator


@dataclass(frozen=True)
class Gear:
    n_teeth: int

    def mesh_radius(self):
        return self.n_teeth / 16

    def mesh_distance(self, other):
        return self.mesh_radius() + other.mesh_radius()

    def clearance_radius(self):
        return self.n_teeth / 16 + 0.125

    def intersects(self, other, dx, dy):
        min_clearance = self.clearance_radius() + other.clearance_radius()
        d = math.sqrt(dx*dx + dy*dy)
        return d < min_clearance

    def __truediv__(self, other):
        return self.n_teeth / other.n_teeth

    def __str__(self):
        return f'{self.n_teeth}t'


@dataclass(frozen=True)
class Bushing:
    def clearance_radius(self):
        return 0.375

    def __str__(self):
        return '[B]'


@dataclass
class GearTreeNode:
    edges: list = field(default_factory=list)

    def connect(self, g1, g2):
        next_node = GearTreeNode()
        self.edges.append((g1, g2, next_node))
        return next_node

    def combine(self, other):
        self.edges.extend(other.edges)
        return self

    def traverse(self, node_fn=None, edge_fn=None):
        to_visit = deque()
        to_visit.append(self)
        while to_visit:
            cur = to_visit.popleft()
            if node_fn:
                node_fn(cur)
            for g1, g2, child in cur.edges:
                if edge_fn:
                    edge_fn(g1, g2, child)
                to_visit.append(child)

    def leafs(self):
        for g1, g2, child in self.edges:
            if child.edges:
                yield from child.leafs()
            else:
                yield (g1, g2, child)

    def depth(self):
        ret = 0
        for _, _, child in self.edges:
            ret = max(ret, 1 + child.depth())
        return ret

    def __hash__(self):
        return hash(tuple(self.edges))

    def __str__(self, indent=0):
        ret = ''
        for g1, g2, child in self.edges:
            ret += ' ' * indent + f'{g1} => {g2}\n'
            ret += child.__str__(indent+2)
        return ret

    @classmethod
    def Sequence(cls, *pairs):
        root = cls()
        node = root
        for g1, g2 in pairs:
            node = node.connect(g1, g2)
        return root

@dataclass
class MeshGenerator:
    error: float = 0
    spacing: float = 1

    def _round_up(self, x):
        return math.ceil(x / self.spacing) * self.spacing

    def _spacing_aligned_range(self, lower, upper):
        i = self._round_up(lower)
        while i <= upper:
            yield i
            i += self.spacing

    def __hash__(self):
        return hash((self.error, self.spacing))

    @memoize_generator
    def generate_positions(self, g1, g2):
        target_d = g1.mesh_distance(g2)
        for dy in self._spacing_aligned_range(-target_d - self.error, target_d + self.error):
            for dx in self._spacing_aligned_range(-target_d - self.error, target_d + self.error):
                d = math.sqrt(dx*dx + dy*dy)
                remain = abs(d - target_d)
                if remain <= self.error:
                    yield (dx, dy)

@dataclass(frozen=True)
class Range:
    lower: float = -math.inf
    upper: float = math.inf

    def contains(self, x):
        return x >= self.lower and x <= self.upper

    def __add__(self, x):
        return Range(lower=self.lower+x, upper=self.upper+x)

    def __sub__(self, x):
        return Range(lower=self.lower-x, upper=self.upper-x)

    def __truediv__(self, x):
        return Range(lower=self.lower/x, upper=self.upper/x)

@dataclass
class Axle2D:
    x: int
    y: int
    connections: list = field(default_factory=list)

    def __hash__(self):
        return hash((self.x, self.y, tuple(self.connections)))

    def __str__(self, indent=0):
        ret = ' ' * indent + f'({self.x}, {self.y})\n'
        for g1, g2, child in self.connections:
            ret += ' ' * indent + f'{g1} => {g2}\n'
            ret += child.__str__(indent+2)
        return ret

class GearCalculator2D:
    available_gears = [Gear(n) for n in (8, 16, 24, 40, 12, 20, 36)]
    mesh_gen = MeshGenerator()

    @memoize_generator
    def generate_trains(self, ratio, max_pairs, first_gear=None, last_gear=None):
        ratio_achieved = ratio.contains(1) if isinstance(ratio, Range) else ratio == 1
        if ratio_achieved:
            yield GearTreeNode()
        if max_pairs <= 0:
            return

        g1s = [first_gear] if first_gear is not None else self.available_gears
        for g1 in g1s:
            for g2 in self.available_gears:
                r = g1 / g2
                for tail_node in self.generate_trains(ratio / r, max_pairs - 1, last_gear=last_gear):
                    combined = GearTreeNode()
                    combined.connect(g1, g2).combine(tail_node)
                    if last_gear is None or tail_node.edges or g2 == last_gear:
                        yield combined

    @memoize_generator
    def _arrange_tree(self, root_node, dx_range=Range(), dy_range=Range()):
        if not root_node.edges:
            if dx_range.contains(0) and dy_range.contains(0):
                yield []
            return

        edges = []
        for g1, g2, child in root_node.edges:
            child_arrangements = []
            for dx, dy in self.mesh_gen.generate_positions(g1, g2):
                arrangements = self._arrange_tree(child, dx_range-dx, dy_range-dy)
                for arrangement in arrangements:
                    child_arrangements.append((dx, dy, arrangement))
            edges.append((g1, g2, child_arrangements))
        arrangements = itertools.product(*(x for _, _, x in edges))
        for arrangement in arrangements:
            arranged_edges = list(
                    (g1, g2, dx, dy, t)
                    for (g1, g2, _), (dx, dy, t) in zip(edges, arrangement))
            yield arranged_edges

    def _arranged_to_axle_tree(self, arranged_edges, x=0, y=0):
        ret = Axle2D(x=x, y=y)
        for g1, g2, dx, dy, child in arranged_edges:
            ret.connections.append((g1, g2, self._arranged_to_axle_tree(child, x+dx, y+dy)))
        return ret

    def generate_axle_trees(self, gear_root_node, dx_range=Range(), dy_range=Range()):
        for arranged_edges in self._arrange_tree(gear_root_node, dx_range, dy_range):
            yield self._arranged_to_axle_tree(arranged_edges)

    def generate_flat_axle_trees(self, gear_trees, dx_range=Range(), dy_range=Range()):
        return itertools.chain.from_iterable(
                (self.generate_axle_trees(t, dx_range, dy_range) for t in gear_trees))


@dataclass
class GearLayer3D:
    z_offset: int = 0
    z_gears: defaultdict[list] = field(default_factory=lambda: defaultdict(list))

    def intersects_axle(self, x, y):
        return any(g.intersects(Bushing(), x - gx, y - gy)
                   for gx, gy, g in itertools.chain.from_iterable(self.z_gears.values()))

    def z_upper_bound(self):
        return 1 + max(self.z_gears.keys()) if self.z_gears else self.z_offset

    def find(self, x, y, g):
        for z, xygs in self.z_gears.items():
            if (x, y, g) in xygs:
                return z
        return None

    def z_sorted_gears(self):
        flattened = ((x, y, z, g) for z, xygs in self.z_gears.items() for (x, y, g) in xygs)
        return sorted(flattened, key=lambda t: (t[2], t[0], t[1]))

    def add_gear(self, x, y, g):
        z_upper = self.z_upper_bound()
        for z in range(self.z_offset, 1 + z_upper):
            collides = any(g.intersects(prev_g, x - prev_x, y - prev_y)
                           for prev_x, prev_y, prev_g in self.z_gears[z])
            if not collides:
                self.z_gears[z].append((x, y, g))
                return z
        raise AssertionError(f'no available z index in [{self.z_offset}, {z_upper}]; '
                             'this should never happen!')

    def add_gear_pair(self, x1, y1, g1, x2, y2, g2):
        z_upper = self.z_upper_bound()
        for z in range(self.z_offset, 1 + z_upper):
            collides = any(g.intersects(prev_g, x - prev_x, y - prev_y)
                           for prev_x, prev_y, prev_g in self.z_gears[z]
                           for (x, y, g) in ((x1, y1, g1), (x2, y2, g2)))
            if not collides:
                self.z_gears[z].append((x1, y1, g1))
                self.z_gears[z].append((x2, y2, g2))
                return z
        raise AssertionError(f'no available z index in [{self.z_offset}, {z_upper}]; '
                             'this should never happen!')

    def num_gears(self):
        return sum(len(gs) for gs in self.z_gears.values())

    def stud_size(self):
        min_x = min(x for x, _, _ in itertools.chain.from_iterable(self.z_gears.values()))
        max_x = max(x for x, _, _ in itertools.chain.from_iterable(self.z_gears.values()))
        min_y = min(y for _, y, _ in itertools.chain.from_iterable(self.z_gears.values()))
        max_y = max(y for _, y, _ in itertools.chain.from_iterable(self.z_gears.values()))
        min_z = self.z_offset
        max_z = self.z_upper_bound() - 1
        return (1 + max_x - min_x), (1 + max_y - min_y), (1 + max_z - min_z)

    def clearance_size(self):
        min_x = min(x - g.clearance_radius()
                for x, _, g in itertools.chain.from_iterable(self.z_gears.values()))
        max_x = max(x + g.clearance_radius()
                for x, _, g in itertools.chain.from_iterable(self.z_gears.values()))
        min_y = min(y - g.clearance_radius()
                for _, y, g in itertools.chain.from_iterable(self.z_gears.values()))
        max_y = max(y + g.clearance_radius()
                for _, y, g in itertools.chain.from_iterable(self.z_gears.values()))
        return (max_x - min_x), (max_y - min_y)

    def __str__(self):
        return f'{self.z_sorted_gears()}'


@dataclass
class GearLayers3D:
    layers: list = field(default_factory=lambda: [GearLayer3D()])

    def cur_layer(self):
        return self.layers[-1]

    def start_new_layer(self):
        self.layers.append(GearLayer3D(z_offset=self.cur_layer().z_upper_bound()))

    def add_axle_tree(self, root):
        todo = deque()
        todo.append((root, False))
        while todo:
            node, needs_new_layer = todo.popleft()
            if needs_new_layer:
                self.start_new_layer()
            for g1, g2, child in node.connections:
                child_needs_new_layer = self.cur_layer().intersects_axle(child.x, child.y)
                z = self.cur_layer().add_gear_pair(node.x, node.y, g1, child.x, child.y, g2)
                todo.append((child, child_needs_new_layer))
        return self

    def z_offset(self):
        return self.layers[0].z_offset

    def z_upper_bound(self):
        return self.layers[-1].z_upper_bound()

    def z_sorted_gears(self):
        return list(map(GearLayer3D.z_sorted_gears, self.layers))

    def num_gears(self):
        return sum(layer.num_gears() for layer in self.layers)

    def stud_size(self):
        min_x = min(x for layer in self.layers
                    for x, _, _ in itertools.chain.from_iterable(layer.z_gears.values()))
        max_x = max(x for layer in self.layers
                    for x, _, _ in itertools.chain.from_iterable(layer.z_gears.values()))
        min_y = min(y for layer in self.layers
                    for _, y, _ in itertools.chain.from_iterable(layer.z_gears.values()))
        max_y = max(y for layer in self.layers
                    for _, y, _ in itertools.chain.from_iterable(layer.z_gears.values()))
        min_z = self.z_offset()
        max_z = self.z_upper_bound() - 1
        return (1 + max_x - min_x), (1 + max_y - min_y), (1 + max_z - min_z)

    def clearance_size(self):
        min_x = min(x - g.clearance_radius()
                for layer in self.layers
                for x, _, _, g in layer.z_sorted_gears())
        max_x = max(x + g.clearance_radius()
                for layer in self.layers
                for x, _, _, g in layer.z_sorted_gears())
        min_y = min(y - g.clearance_radius()
                for layer in self.layers
                for _, y, _, g in layer.z_sorted_gears())
        max_y = max(y + g.clearance_radius()
                for layer in self.layers
                for _, y, _, g in layer.z_sorted_gears())
        return (max_x - min_x), (max_y - min_y)

    def __str__(self):
        return f'{self.z_sorted_gears()}'
