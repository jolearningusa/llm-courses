import unittest

import numpy as np

import cnn


def numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def naive_conv(x, W, b, stride, pad):
    """Direct loops over every output position: the definition im2col must match."""
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = cnn.conv_out_size(H, k, stride, pad), cnn.conv_out_size(Wd, k, stride, pad)
    out = np.zeros((N, F, Ho, Wo))
    for n in range(N):
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    out[n, f, i, j] = (xp[n, :, i * stride:i * stride + k, j * stride:j * stride + k] * W[f]).sum() + b[f]
    return out


class Shapes(unittest.TestCase):
    def test_output_size(self):
        self.assertEqual(cnn.conv_out_size(224, 11, 4, 2), 55)   # AlexNet's first layer
        self.assertEqual(cnn.conv_out_size(32, 5), 28)           # LeNet-5's first layer
        self.assertEqual(cnn.conv_out_size(56, 3, 1, 1), 56)     # "same" padding

    def test_receptive_field(self):
        self.assertEqual(cnn.receptive_field([(3, 1)] * 2), 5)
        self.assertEqual(cnn.receptive_field([(3, 1)] * 3), 7)
        self.assertEqual(cnn.receptive_field([(3, 1), (3, 1), (2, 2), (3, 1)]), 10)

    def test_three_3x3_cheaper_than_one_7x7(self):
        self.assertLess(3 * cnn.conv_params(64, 64, 3, False), cnn.conv_params(64, 64, 7, False))


class Layers(unittest.TestCase):
    def test_conv_matches_definition(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 3, 7, 6))
        for stride, pad in [(1, 0), (1, 1), (2, 1)]:
            conv = cnn.Conv2d(3, 4, 3, stride, pad, rng)
            conv.b[:] = rng.normal(size=4)
            np.testing.assert_allclose(conv.forward(x), naive_conv(x, conv.W, conv.b, stride, pad), atol=1e-12)

    def test_conv_gradients(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 2, 5, 5))
        conv = cnn.Conv2d(2, 3, 3, stride=2, pad=1, rng=rng)
        up = rng.normal(size=conv.forward(x).shape)
        f = lambda: float((conv.forward(x) * up).sum())
        f()
        gx = conv.backward(up)
        np.testing.assert_allclose(gx, numerical_grad(f, x), rtol=1e-6, atol=1e-8)
        f()
        conv.backward(up)
        np.testing.assert_allclose(conv.grads[0], numerical_grad(f, conv.W), rtol=1e-6, atol=1e-8)

    def test_maxpool_routes_gradient_to_the_maximum(self):
        x = np.array([[[[1., 5., 0., 0.], [2., 3., 0., 9.], [0., 0., 4., 0.], [0., 7., 0., 0.]]]])
        pool = cnn.MaxPool2d(2)
        np.testing.assert_array_equal(pool.forward(x)[0, 0], [[5, 9], [7, 4]])
        g = pool.backward(np.ones((1, 1, 2, 2)))[0, 0]
        self.assertEqual(g.sum(), 4)
        self.assertEqual(g[0, 1], 1)
        self.assertEqual(g[1, 3], 1)

    def test_convolution_is_translation_equivariant(self):
        rng = np.random.default_rng(2)
        conv = cnn.Conv2d(1, 2, 3, pad=1, rng=rng)
        x = np.zeros((1, 1, 10, 10))
        x[0, 0, 2:5, 2:5] = rng.normal(size=(3, 3))
        shifted = np.roll(x, (3, 2), axis=(2, 3))
        a = conv.forward(x)
        b = conv.forward(shifted)
        np.testing.assert_allclose(np.roll(a, (3, 2), axis=(2, 3)), b, atol=1e-12)

    def test_whole_cnn_gradients(self):
        rng = np.random.default_rng(3)
        model = cnn.small_cnn(seed=4)
        X, y = cnn.shapes(3, seed=5)
        loss = lambda: cnn.softmax_xent(model.forward(X), y)[0]
        _, g = cnn.softmax_xent(model.forward(X), y)
        model.backward(g)
        for p, gp in list(model.params_and_grads())[:2]:   # the first conv: deepest in the backward pass
            np.testing.assert_allclose(gp, numerical_grad(loss, p), rtol=1e-5, atol=1e-8)


class Experiments(unittest.TestCase):
    def test_cnn_generalizes_to_moved_shapes_and_mlp_does_not(self):
        Xtr, ytr = cnn.shapes(2000, region=(0, 3), seed=0)
        Xte, yte = cnn.shapes(1000, region=(4, 7), seed=1)
        c, m = cnn.small_cnn(), cnn.small_mlp()
        self.assertLess(abs(c.num_params() - m.num_params()), 200, "compare at about the same size")
        cnn.sgd_train(c, Xtr, ytr, lr=0.02, steps=600)
        cnn.sgd_train(m, Xtr, ytr, lr=0.02, steps=600)
        self.assertGreater(cnn.accuracy(c, Xte, yte), 0.85)
        self.assertLess(cnn.accuracy(m, Xte, yte), 0.5)

    def test_residual_connections_rescue_a_deep_network(self):
        plain = cnn.depth_experiment(64, False)
        res = cnn.depth_experiment(64, True)
        self.assertGreater(res[0], 10 * plain[0], "much more gradient reaches the first layer")
        self.assertGreater(plain[1], 0.8, "the 64-layer plain net does not train")
        self.assertLess(res[1], 0.1)


if __name__ == "__main__":
    unittest.main()
