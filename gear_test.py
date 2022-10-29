import math
import unittest

from gear import *

g40 = Gear(40)
g24 = Gear(24)
g20 = Gear(20)
g16 = Gear(16)
g12 = Gear(12)
g8 = Gear(8)


def axle_seq(*pairs, x0=0, y0=0):
    ret = Axle2D(x=x0,y=y0)
    if not pairs:
        return ret
    g1, g2, x, y = pairs[0]
    axle_for_tail = axle_seq(*pairs[1:], x0=x, y0=y)
    ret.connections.append((g1, g2, axle_for_tail))
    return ret


class GearTest(unittest.TestCase):
    def test_mesh(self):
        self.assertEqual(g40.mesh_distance(g8), 3)
        self.assertEqual(g8.mesh_distance(g16), 1.5)

    def test_intersects(self):
        self.assertTrue(g40.intersects(g40, dx=4, dy=0))
        self.assertFalse(g8.intersects(g8, dx=1, dy=1))

    def test_ratio(self):
        self.assertEqual(g40 / g8, 5)
        self.assertEqual(g8 / g16, 0.5)


class MeshGeneratorTest(unittest.TestCase):
    def test_1d_meshings(self):
        gen = MeshGenerator()
        self.assertEqual(set(gen.generate_positions(g40, g8)), {
            (-3,  0),
            ( 3,  0),
            ( 0, -3),
            ( 0,  3),
        })

    def test_2d_meshings(self):
        gen = MeshGenerator(error=0.05)
        self.assertEqual(set(gen.generate_positions(g24, g12)), {
            (-2, -1),
            (-1, -2),
            ( 1, -2),
            ( 2, -1),
            ( 2,  1),
            ( 1,  2),
            (-1,  2),
            (-2,  1),
        })


class GearTreeTest(unittest.TestCase):
    def test_eq(self):
        t1 = GearTreeNode()
        t1.connect(g40, g8)
        t2 = GearTreeNode()
        t2.connect(g40, g8)
        self.assertEqual(t1, t2)

        t1.x, t1.y = 1, 2
        t2.x, t2.y = 1, 2
        self.assertEqual(t1, t2)

    def test_neq(self):
        t1 = GearTreeNode()
        t1.connect(g40, g8)
        t2 = GearTreeNode()
        t2.connect(g40, g16)
        self.assertNotEqual(t1, t2)


class GearCalculatorTest(unittest.TestCase):
    def test_gen_trains(self):
        calc2d = GearCalculator2D()
        calc2d.mesh_gen.error = 0.05
        calc2d.available_gears = [g40, g24, g16, g12, g8]
        trains = set(calc2d.generate_trains(ratio=2, max_pairs=1))
        self.assertEqual(trains, {
            GearTreeNode.Sequence((g24, g12)),
            GearTreeNode.Sequence((g16, g8)),
        })

    def test_arrange_axles(self):
        calc2d = GearCalculator2D()
        calc2d.mesh_gen.error = 0.05
        root = (GearTreeNode()
                .combine(GearTreeNode.Sequence((g40, g8)))
                .combine(GearTreeNode.Sequence((g16, g16))))

        axle_trees = set(calc2d.generate_axle_trees(root))
        expected_sample = Axle2D(x=0, y=0, connections=[
            (g40, g8, Axle2D(x=3, y=0)),
            (g16, g16, Axle2D(x=2, y=0)),
        ])
        self.assertIn(expected_sample, axle_trees)

    def test_arrange_axles_constrained(self):
        calc2d = GearCalculator2D()
        calc2d.mesh_gen.error = 0.05
        seq = GearTreeNode.Sequence((g16, g16), (g24, g12), (g24, g12))
        axle_trees = set(calc2d.generate_axle_trees(seq, dx_range=Range(0,0), dy_range=Range(0,0)))
        expected_sample = axle_seq(
                (g16, g16, 2, 0),
                (g24, g12, 1, 2),
                (g24, g12, 0, 0),
        )
        self.assertIn(expected_sample, axle_trees)

    def test_clock_hands(self):
        calc2d = GearCalculator2D()
        calc2d.mesh_gen.error = 0.05
        trains = calc2d.generate_trains(ratio=12, max_pairs=4, first_gear=g16)
        axle_trees = set(calc2d.generate_flat_axle_trees(
            trains, dx_range=Range(0,0), dy_range=Range(0,0)))
        self.assertIn(
            axle_seq(
                (g16, g16, 2, 0),
                (g16, g20, 3, 2),
                (g24, g8,  3, 0),
                (g40, g8,  0, 0),
            ),
            axle_trees)
        self.assertIn(
            axle_seq(
                (g16, g16, 2, 0),
                (g24, g12, 3, 2),
                (g24, g12, 2, 0),
                (g24, g8,  0, 0),
            ),
            axle_trees)


class GearLayerTest(unittest.TestCase):
    def test_z_ub(self):
        layer = GearLayer3D()
        self.assertEqual(layer.z_upper_bound(), 0)
        layer.add_gear(0, 0, g40) # z=0
        self.assertEqual(layer.z_upper_bound(), 1)
        layer.add_gear(1, 0, g40) # z=1
        self.assertEqual(layer.z_upper_bound(), 2)
        layer.add_gear(2, 0, g40) # z=2
        self.assertEqual(layer.z_upper_bound(), 3)

    def test_add_gears(self):
        layer = GearLayer3D()
        self.assertEqual(layer.add_gear(0, 0, g8), 0)
        self.assertEqual(layer.add_gear_pair(0, 0, g40, 3, 0, g8), 1)
        self.assertEqual(layer.add_gear(3, 0, g12), 0)

    def test_add_gears_with_offset(self):
        layer = GearLayer3D(z_offset=12)
        self.assertEqual(layer.add_gear(0, 0, g8), 12)
        self.assertEqual(layer.add_gear_pair(0, 0, g40, 3, 0, g8), 13)
        self.assertEqual(layer.add_gear(3, 0, g12), 12)

    def test_intersection(self):
        layer = GearLayer3D()
        layer.add_gear(0, 0, g40)
        layer.add_gear(5, 0, g40)
        layer.add_gear(1, 1, g24)
        self.assertTrue(layer.intersects_axle(-1, -1))
        self.assertTrue(layer.intersects_axle(-2, -2))
        self.assertFalse(layer.intersects_axle(-2, -3))
        self.assertTrue(layer.intersects_axle(2, 2))
        self.assertTrue(layer.intersects_axle(4, 2))
        self.assertFalse(layer.intersects_axle(5, 3))

    def test_sort(self):
        layer = GearLayer3D()
        self.assertEqual(layer.add_gear(0, 0, g8), 0)
        self.assertEqual(layer.add_gear_pair(0, 0, g40, 3, 0, g8), 1)
        self.assertEqual(layer.add_gear(3, 0, g12), 0)
        self.assertEqual(layer.z_sorted_gears(), [
            (0, 0, 0, g8),
            (3, 0, 0, g12),
            (0, 0, 1, g40),
            (3, 0, 1, g8),
        ])

    def test_layer_sequence(self):
        axle_tree = axle_seq(
                (g16, g16, 2, 0), (g16, g20, 3, 2),
                (g24, g8, 3, 0), (g40, g8, 0, 0),
        )
        layers = GearLayers3D()
        layers.add_axle_tree(axle_tree)
        for layer in layers.layers:
            print('==layer==')
            print(f'{layer.z_sorted_gears()}')
        self.assertEqual(len(layers.layers), 2)

    def test_layers_with_branches(self):
        axle_tree = Axle2D(x=0, y=0, connections=[
            (g40, g8, Axle2D(x=3, y=0)),
            (g16, g16, Axle2D(x=2, y=0)),
        ])
        layers = GearLayers3D()
        layers.add_axle_tree(axle_tree)
        self.assertEqual(len(layers.layers), 2)
        self.assertEqual(layers.layers[0].z_sorted_gears(), [
            (0, 0, 0, g40),
            (3, 0, 0, g8),
            (0, 0, 1, g16),
            (2, 0, 1, g16),
            ], msg=f'actual: {layers.layers[0].z_sorted_gears()}')
        self.assertEqual(layers.layers[1].z_sorted_gears(), [])

    def test_clock_hands(self):
        calc2d = GearCalculator2D()
        calc2d.mesh_gen.error = 0.05
        trains = calc2d.generate_trains(ratio=12, max_pairs=4, first_gear=g16)
        axle_trees = calc2d.generate_flat_axle_trees(
            trains, dx_range=Range(0,0), dy_range=Range(0,0))
        arrangements = list(GearLayers3D().add_axle_tree(t) for t in axle_trees)
        def _sort_key(layers):
            return (layers.z_upper_bound(),
                    math.prod(layers.clearance_size()))
        arrangements.sort(key=_sort_key)
        self.assertGreaterEqual(len(arrangements), 1)
        self.assertEqual(len(arrangements[0].layers), 2)



if __name__ == '__main__':
    unittest.main()
