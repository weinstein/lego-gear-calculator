from collections import defaultdict
from itertools import chain, permutations
from math import ceil, floor, sqrt
from memoize import memoize, memoize_generator

ALL_GEARS = [
        8, 16, 24, 40,
        12, 20, 36,
]

def pitch_radius(teeth):
    return teeth / 16

# Obtained from physical measurements and experimentation.
# This is the outer radius of the gear's teeth -- the distance required for the
# gear to still spin freely.
def clearance_radius(teeth):
    if teeth is None:
        return BUSHING_RADIUS
    return teeth / 16 + 0.125

def mesh_distance(n1, n2):
    return (n1 + n2) / 16

AXLE_RADIUS = 0.25
BUSHING_RADIUS = 0.375

def round_down(x, spacing):
    return floor(x / spacing) * spacing

def round_up(x, spacing):
    return ceil(x / spacing) * spacing

# Generates a list of numbers in the range [lower, upper] that are multiples of
# spacing.
# Ex: spaced_range(-1, 2.75, 0.5) yields -1, -0.5, 0, 0.5, ... 2.5
def spaced_range(lower, upper, spacing):
    i = round_up(lower, spacing)
    while i < upper:
        yield i
        i += spacing

# Generates all the possible 2D spacings between gears that will mesh.
# Yields (dx, dy) tuples of the possible x/y distances.
@memoize_generator
def meshings(g1, g2, spacing=1, err=0):
    target_d = mesh_distance(g1, g2)
    for dx in spaced_range(-target_d - err, target_d + err, spacing):
        for dy in spaced_range(-target_d - err, target_d + err, spacing):
            d = sqrt(dx*dx + dy*dy)
            remain = abs(d - target_d)
            if remain <= err:
                yield (dx, dy)

# Generates all the gear trains with the given total ratio and the given size.
# Each gear pair in the train is a tuple, and each possible train is a tuple of
# gear pairs.
@memoize_generator
def gear_trains(ratio=1, n_pairs=0):
    if n_pairs <= 0:
        if ratio == 1:
            yield ()
        return

    for g1 in ALL_GEARS:
        for g2 in ALL_GEARS:
            r = g1 / g2
            yield from map(lambda tail: ((g1, g2),) + tail, gear_trains(ratio / r, n_pairs - 1))

# Yields all 2D arrangements of a gear train that will mesh.
# min/max dx/dy if specified applies to the distance between the first and last
# axle (but intermediate axles can still be outside this range).
@memoize_generator
def arrange_gears(gear_pairs,
                  min_dx=None, max_dx=None,
                  min_dy=None, max_dy=None,
                  spacing=1, err=0):
    if not gear_pairs:
        if ((min_dx is None or min_dx <= 0) and
            (max_dx is None or max_dx >= 0) and
            (min_dy is None or min_dy <= 0) and
            (max_dy is None or max_dy >= 0)):
            yield ()
        return

    g1, g2 = gear_pairs[0]
    tail = gear_pairs[1:]
    for dx, dy in meshings(g1, g2, spacing=spacing, err=err):
        yield from map(
                lambda x: ((g1, g2, dx, dy),) + x, 
                arrange_gears(
                        tail,
                        min_dx=min_dx-dx if min_dx is not None else None,
                        max_dx=max_dx-dx if max_dx is not None else None,
                        min_dy=min_dy-dy if min_dy is not None else None,
                        max_dy=max_dy-dy if max_dy is not None else None,
                        spacing=spacing, err=err))

# Re-arranges a sequence of (gear pair, relative position) into a sequence of
# axles, each with gears and an absolute (x,y) position.
# The first and last axle will only have 1 gear each.
def position_axles(arrangement):
    prev = None
    x = 0
    y = 0
    for (g1, g2, dx, dy) in arrangement:
        yield (prev, g1, x, y)
        prev = g2
        x += dx
        y += dy
    yield (prev, None, x, y)

# True if, on the same z-plane, a gear g1 at (x1,y1) would intersect with gear
# g2 at (x2,y2).
def intersects(x1, y1, g1, x2, y2, g2):
    clearance = clearance_radius(g1) + clearance_radius(g2)
    dx = x2 - x1
    dy = y2 - y1
    d = sqrt(dx*dx + dy*dy)
    return d < clearance

# Determine how a train of axles with gears should be layered in z-planes.
# The result is a sequence of layers, and each layer is a sequence of axles.
# Each axle has an (x,y) position and up to 2 gears, each with a z index
# relative to that layer.
# 
# This is a greedy placement algorithm that works as follows.
# To place an axle:
#   1) check if the axle would intersect with anything that was already placed
#      in the current layer.
#      * If it would, start a new layer with the axle. Go back to step (1) with
#        the next axle.
#   2) pick the lowest z index possible without creating any intersections
# There are some caveats around allowing z-reuse for gear duplicates
# (physically, using a single gear instead of 2 of the same).
#
# One problem with this approach is that it doesn't account for the fact that
# adjacent layers need to share an axle, so new layers do not begin entirely
# unconstrained.
# Another problem is that, although it provides useful placements and is fast,
# this approach is not guaranteed to find the most efficient placement.
def layer_axles(axles):
    layers = [[]]
    prev_z = None
    for g1, g2, x, y in axles:
        # If the new axle needs to go through previous gears (or vice versa)
        # then the new axle needs to go in a new layer.
        for g1_, g2_, x_, y_, _, _ in layers[-1]:
            if any(intersects(x, y, g, x_, y_, None) for g in (g1, g2, g1_, g2_)):
                layers.append([])
                break
        cur_layer = layers[-1]
        if g2 is None:
            cur_layer.append((g1, g2, x, y, prev_z, None))
            prev_z = None
            continue
        # Greedy: pick lowest available z
        # max_z is higher than anything else in the current layer
        max_z = (1 + max(z for _, _, _, _, z1_, z2_ in cur_layer for z in (z1_, z2_) if z is not None)
                 if cur_layer else (
                     prev_z + 1
                     if prev_z is not None else 0))
        z_to_gears = defaultdict(list)
        for g1_, g2_, x_, y_, z1_, z2_ in cur_layer:
            z_to_gears[z1_].append((x_, y_, g1_))
            z_to_gears[z2_].append((x_, y_, g2_))
        next_z = max_z
        for z in range(max_z + 1):
            if z == prev_z:
                if g1 != g2:
                    continue
                else:
                    next_z = z
                    break
            z_collides = any(intersects(x, y, g2, x_, y_, g_) for x_, y_, g_ in z_to_gears[z])
            if not z_collides:
                next_z = z
                break
        cur_layer.append((g1, g2, x, y, prev_z, next_z))
        prev_z = z
    return layers

def count_half_spacings(layers):
    count = 0
    for axles in layers:
        for (g1, g2, x, y, _, _) in axles:
            if (x * 2) % 2 == 1 or (y * 2) % 2 == 1:
                count += 1
    return count

def clearance_size(layers):
    min_x = min(x - clearance_radius(g) for axles in layers for g1, g2, x, _, _, _ in axles for g in (g1, g2))
    max_x = max(x + clearance_radius(g) for axles in layers for g1, g2, x, _, _, _ in axles for g in (g1, g2))
    min_y = min(y - clearance_radius(g) for axles in layers for g1, g2, _, y, _, _ in axles for g in (g1, g2))
    max_y = max(y + clearance_radius(g) for axles in layers for g1, g2, _, y, _, _ in axles for g in (g1, g2))
    return (max_x - min_x), (max_y - min_y)

def z_size(layers):
    size = 1
    for layer in layers:
        size += max(z for _, _, _, _, z1, z2 in layer for z in (z1, z2) if z is not None)
    return size

def stud_size(layers):
    min_x = min(x for axles in layers for _, _, x, _, _, _ in axles)
    max_x = max(x for axles in layers for _, _, x, _, _, _ in axles)
    min_y = min(y for axles in layers for _, _, _, y, _, _ in axles)
    max_y = max(y for axles in layers for _, _, _, y, _, _ in axles)
    return (1 + max_x - min_x), (1 + max_y - min_y), z_size(layers)

# Scoring function used for sorting. In order of importance:
#   * number of axles
#   * number of half spacings
#   * length along the z-axis
#   * total volume of bounding box
def placement_score(layers):
    n_axles = sum(len(axles) for axles in layers)
    sx, sy = clearance_size(layers)
    _, _, sz = stud_size(layers)
    vol = sx * sy * sz 
    # return n_axles, count_half_spacings(layers), vol 
    return n_axles, count_half_spacings(layers), sz, vol

def gears_required(layers):
    return sorted(list(g for axles in layers for g1, g2, _, _, z1, z2 in axles for g, _ in {(g1, z1), (g2, z2)} if g is not None))

# True if an axle can be placed at (x,y) without intersecting any gears in the
# axles.
def clear_for_axle(axles, x, y):
    for g1, g2, x_, y_ in axles:
        for g in (g1, g2):
            x0 = x_ - clearance_radius(g)
            x1 = x_ + clearance_radius(g)
            y0 = y_ - clearance_radius(g)
            y1 = y_ + clearance_radius(g)
            is_clear = (x0 >= x + BUSHING_RADIUS or
                        x1 <= x - BUSHING_RADIUS or
                        y0 >= y + BUSHING_RADIUS or
                        y1 <= y - BUSHING_RADIUS)
            if not is_clear:
                return False
    return True

# Calculates possible 2D arrangements of gear trains.
def solve(ratio=1,
          min_dx=None, max_dx=None,
          min_dy=None, max_dy=None,
          max_depth=0, spacing=1, err=0,
          train_filter=None,
          train_mapper=None):
    for i in range(max_depth):
        trains = gear_trains(ratio=ratio, n_pairs=i)
        if train_filter:
            trains = filter(train_filter, trains)
        if train_mapper:
            trains = chain.from_iterable(map(train_mapper, trains))
        yield from chain.from_iterable(
                arrange_gears(
                        train,
                        min_dx=min_dx, max_dx=max_dx,
                        min_dy=min_dy, max_dy=max_dy,
                        spacing=spacing, err=err)
                for train in trains)


# Calculates all the 3D arrangements of gear trains for clock minute/hour hands.
# The constraints are:
#   * gear ratio must be 12: 12 minute hand rotations = 1 hour hand rotation
#   * must be coaxial: first and last axle must have the same 0,0 position AND
#     that position must not intersect with any other axles/gears in the train
#   * must begin with a 16t clutch gear to attach the hour hand on the minute
#     hand's axle
#   * must have an even number of gear pairs so that input and output rotate in
#     the same direction
def clock_hands():
    arrangements = solve(
            ratio=12,
            min_dx=0, max_dx=0,
            min_dy=0, max_dy=0,
            max_depth=5,
            spacing=0.5,
            err=0.05,
            # Coaxial minute/hour hands:
            #  must start with 16t clutch,
            #  and must have the same direction of rotation for input/output
            train_filter=lambda t: t[0][0] == 16 and len(t) % 2 == 0)
    axle_placements = (list(position_axles(a)) for a in arrangements)
    # Coaxial minute/hour hands constraint:
    # minute hand must be able to route through the entire arrangement at 0,0
    axle_placements = filter(lambda x: clear_for_axle(x[1:-1], 0, 0), axle_placements)
    axle_placements = list(map(layer_axles, axle_placements))
    axle_placements.sort(key=placement_score)
    return axle_placements


# Calculates all the 3D arrangements of gear trains between minute hands and an
# escapement wheel.
# The constraints are:
#   * with a 40t escapement wheel and a 1s pendulum, total ratio must be 90
#     (escapement rotation = 40s, minute hand rotation = 3600s, 3600/40=90)
#   * for a horizontally symmetric appearance, gear train must end at the same
#     x-position as it started. That puts the escapement wheel in line
#     vertically with the hands.
def escapement_gear_trains():
    arrangements = solve(
            ratio=90,
            min_dx=0, max_dx=0,
            max_depth=6,
            spacing=1,
            err=0.05)
    axle_placements = (list(position_axles(a)) for a in arrangements)
    axle_placements = list(map(layer_axles, axle_placements))
    axle_placements.sort(key=placement_score)
    return axle_placements

# Calculates all the 3D arrangements of the entire gear train for a clock.
# There are 2 sub trains:
#   * hour hand -> minute hand
#   * minute hand -> escapement
# This calculates gear trains for the subtrains individually, then glues
# them together and calculates z placement for the glued trains.
def total_clock():
    hands = solve(
            ratio=12,
            min_dx=0, max_dx=0,
            min_dy=0, max_dy=0,
            max_depth=5,
            spacing=1,
            err=0.05,
            # Coaxial minute/hour hands:
            #  must start with 16t clutch,
            #  and must have the same direction of rotation for input/output
            train_filter=lambda t: t[0][0] == 16 and len(t) % 2 == 0)
    hands_axles = (list(position_axles(a)) for a in hands)
    # Coaxial minute/hour hands constraint:
    # minute hand must be able to route through the entire arrangement
    hands_axles = filter(lambda x: clear_for_axle(x[1:-1], 0, 0), hands_axles)

    def _insert_clutchable(t):
        # A 40t or 36t gear can be used to make the first axle into a friction axle.
        # A 16t clutch gear linkage in the first pair can be used as a friction
        # clutch connecting to the minute hand.
        if t[1][0] == 40 or t[1][0] == 36 or t[0][1] == 16:
            yield t
        if t[0][1] != 16:
            yield ((16, 16),) + t
    escapement = solve(
            ratio=90,
            min_dx=0, max_dx=0,
            min_dy=-5, max_dy=-3,
            max_depth=6,
            spacing=1,
            err=0.05,
            train_mapper=_insert_clutchable)
    escapement_axles = (list(position_axles(a)) for a in escapement)
    glued = []
    for h in hands_axles:
        for e in escapement_axles:
            # glue the trains together
            middle = h[-1][0:1] + e[0][1:]
            # add the 40t escapement gear at the end of the escapement train
            end = e[-1][0:1] + (40,) + e[-1][2:]
            glued.append(h[:-1] + [middle] + e[1:-1] + [end])
    glued = list(map(layer_axles, glued))
    glued.sort(key=placement_score)
    return glued

def pretty_coords(x, y, z, elem_width=4):
    xy = f'{x:{elem_width}},{y:{elem_width}}'
    zs = f'{z:{elem_width}}' if z is not None else '*'.rjust(elem_width)
    return f'({xy},{zs})'

def pretty_gear(g, width=2):
    return f'{g:{width}}' if g is not None else '_'.rjust(width)

def axle_debug_str(g1, g2, x, y, z1, z2):
    return ' -- '.join(f'{pretty_coords(x,y,z)} {pretty_gear(g)}' for (g, z) in ((g1, z1), (g2, z2)))

def pretty_axle(g1, g2, x, y, z1, z2, stud_width=32):
    ret = ''
    if z1 is None or (z2 is not None and z1 > z2):
        (g1, z1), (g2, z2) = (g2, z2), (g1, z1)
    for (g, z) in ((g1, z1), (g2, z2)):
        if g is not None and z is not None and (z+1) * stud_width >= len(ret):
            ret += f'{pretty_coords(x, y, z)} {pretty_gear(g)}'.rjust((z+1) * stud_width)[len(ret):]
    return ret

if __name__ == '__main__':
    axle_placements = total_clock()
    for placement in axle_placements:
        print(f'{sum(len(axles) for axles in placement)} axles '
              f'{count_half_spacings(placement)} half-spaced '
              f'in an {clearance_size(placement)} area; '
              f'gears: {gears_required(placement)} '
              f'studs: {stud_size(placement)} '
              f'layers: {len(placement)}')
        for i in range(len(placement)):
            layer = placement[i]
            print(f'  layer {i+1}:')
            for axle in layer:
                print(f'    {pretty_axle(*axle)}')
