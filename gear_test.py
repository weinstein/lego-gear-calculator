import unittest

from gear import *

g40 = Gear(40)
g24 = Gear(24)
g16 = Gear(16)
g12 = Gear(12)
g8 = Gear(8)


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
        self.assertTrue(
                expected_sample in axle_trees,
                msg=f'expected to find:\n{expected_sample}\n'
                    f'actual:\n' + '\n========\n'.join(str(t) for t in axle_trees))


if __name__ == '__main__':
    unittest.main()
