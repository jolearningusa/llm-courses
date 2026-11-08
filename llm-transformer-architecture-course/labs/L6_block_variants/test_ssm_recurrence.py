import math
import random
import unittest

from ssm_recurrence import coefficients, mimo_read, mimo_update, reference, stream


class RecurrenceTests(unittest.TestCase):
    def close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertLessEqual(abs(a-b), 1e-11 + 1e-11 * abs(b))

    def test_literal_trapezoidal_states(self):
        params = [coefficients(1.0, 0.0, 0.5)] * 2
        actual, carry = stream([2.0, 4.0], params)
        self.assertEqual(actual, [1.0, 4.0])
        self.assertEqual(carry, (4.0, 4.0))
        self.assertEqual(reference([2.0, 4.0], params), [1.0, 4.0])

    def test_euler_endpoint_ignores_previous_drive(self):
        params = [coefficients(0.5, -0.7, 1.0)] * 3
        self.assertEqual(params[0][1], 0)
        a, _ = stream([1, 2, 3], params, (2j, 1000))
        b, _ = stream([1, 2, 3], params, (2j, -999))
        self.close(a, b)

    def test_complex_step_matches_unrolled_sum(self):
        rng = random.Random(17)
        values = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(128)]
        params = [coefficients(rng.uniform(0.01, 0.2), -0.3, rng.random(),
                               rng.uniform(-0.5, 0.5)) for _ in values]
        actual, _ = stream(values, params, (0.2 + 0.4j, -0.7j))
        self.close(actual, reference(values, params, (0.2 + 0.4j, -0.7j)))

    def test_chunk_boundary_needs_state_and_previous_drive(self):
        values = [math.sin(i / 7) for i in range(257)]
        params = [coefficients(0.1, -0.2, 0.4, 0.1)] * len(values)
        whole, _ = stream(values, params)
        chunks, carry = [], (0j, 0j)
        for start in range(0, len(values), 17):
            part, carry = stream(values[start:start+17], params[start:start+17], carry)
            chunks.extend(part)
        self.close(chunks, whole)
        self.assertEqual(stream([], [], carry), ([], carry))
        _, boundary = stream(values[:17], params[:17])
        broken, _ = stream(values[17:], params[17:], (boundary[0], 0j))
        self.assertGreater(abs(broken[0] - whole[17]), 1e-4)

    def test_future_inputs_do_not_change_the_prefix(self):
        params = [coefficients(0.1, -0.5, 0.5)] * 4
        a, _ = stream([1, 2, 3, 4], params)
        b, _ = stream([1, 2, 100, -200], params)
        self.assertEqual(a[:2], b[:2])

    def test_complex_multiply_is_two_real_coordinate_rotation(self):
        phase, radius, z = 0.7, 0.8, complex(2, -3)
        alpha, _, _ = coefficients(1.0, math.log(radius), 1.0, phase)
        actual = alpha * z
        expected = radius * complex(math.cos(phase)*2 + math.sin(phase)*3,
                                    math.sin(phase)*2 - math.cos(phase)*3)
        self.close([actual], [expected])
        self.assertAlmostEqual(abs(actual), radius * abs(z))

    def test_literal_mimo_update_read_and_fixed_core_shape(self):
        state = [[1, 2], [3, 4]]
        result = mimo_update(state, 0.5, [[1, 2], [3, 4]], [[5, 6], [7, 8]])
        self.assertEqual(result, [[17.5, 24], [40.5, 55]])
        self.assertEqual(mimo_read(result, [[1, 0], [0, 1]]), [[17.5, 40.5], [24, 55]])
        rank1 = mimo_update(state, 0.5, [[1], [3]], [[5], [7]])
        self.assertEqual(rank1, [[5.5, 8], [16.5, 23]])
        self.assertEqual(sum(map(len, result)), sum(map(len, rank1)))

    def test_invalid_coefficients_and_shapes(self):
        for args in [(0, -1, 0.5), (1, 1, 0.5), (1, -1, 1.1), (1, -1, float('nan'))]:
            with self.assertRaises(ValueError):
                coefficients(*args)
        with self.assertRaises(ValueError):
            stream([1], [])
        with self.assertRaises(ValueError):
            reference([1], [])
        with self.assertRaises(ValueError):
            mimo_update([[0]], 1, [[1, 2]], [[3]])
        with self.assertRaises(ValueError):
            mimo_read([[0]], [[1], [2]])


if __name__ == '__main__':
    unittest.main()
