import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from main import lorenz_system, integrate_lorenz


class TestLorenzSystem:
    def test_lorenz_system_zero_point(self):
        point = np.array([0.0, 0.0, 0.0])
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        result = lorenz_system(point, sigma, rho, beta)
        expected = np.array([0.0, 0.0, 0.0])
        assert_array_equal(result, expected)

    def test_lorenz_system_known_point(self):
        point = np.array([1.0, 2.0, 3.0])
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        result = lorenz_system(point, sigma, rho, beta)
        x, y, z = point
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        expected = np.array([dx_dt, dy_dt, dz_dt])
        assert_array_almost_equal(result, expected)

    def test_lorenz_system_different_params(self):
        point = np.array([1.0, 1.0, 1.0])
        sigma, rho, beta = 5.0, 15.0, 2.0
        result = lorenz_system(point, sigma, rho, beta)
        expected = np.array([0.0, 14.0 - 1.0, 1.0 - 2.0])
        assert_array_almost_equal(result, expected)

    def test_lorenz_system_negative_values(self):
        point = np.array([-1.0, -2.0, -3.0])
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        result = lorenz_system(point, sigma, rho, beta)
        x, y, z = point
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        expected = np.array([dx_dt, dy_dt, dz_dt])
        assert_array_almost_equal(result, expected)

    def test_lorenz_system_return_type(self):
        point = np.array([1.0, 2.0, 3.0])
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        result = lorenz_system(point, sigma, rho, beta)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


class TestIntegrateLorenz:
    def test_integrate_single_step(self):
        initial_point = np.array([1.0, 1.0, 1.0])
        params = (10.0, 28.0, 8.0 / 3.0)
        dt = 0.01
        steps = 1
        result = integrate_lorenz(initial_point, params, dt, steps)
        assert result.shape == (steps, 3)
        assert_array_equal(result[0], initial_point)

    def test_integrate_multiple_steps(self):
        initial_point = np.array([1.0, 1.0, 1.0])
        params = (10.0, 28.0, 8.0 / 3.0)
        dt = 0.01
        steps = 10
        result = integrate_lorenz(initial_point, params, dt, steps)
        assert result.shape == (steps, 3)
        assert_array_equal(result[0], initial_point)
        for i in range(1, steps):
            assert not np.array_equal(result[i], result[i - 1])

    def test_integrate_zero_initial(self):
        initial_point = np.array([0.0, 0.0, 0.0])
        params = (10.0, 28.0, 8.0 / 3.0)
        dt = 0.01
        steps = 10
        result = integrate_lorenz(initial_point, params, dt, steps)
        for i in range(steps):
            assert_array_almost_equal(result[i], initial_point, decimal=6)

    def test_integrate_different_dt(self):
        initial_point = np.array([1.0, 1.0, 1.0])
        params = (10.0, 28.0, 8.0 / 3.0)
        steps = 100
        result_small_dt = integrate_lorenz(initial_point, params, 0.001, steps)
        result_large_dt = integrate_lorenz(initial_point, params, 0.1, steps)
        assert not np.array_equal(result_small_dt[-1], result_large_dt[-1])

    def test_integrate_return_type(self):
        initial_point = np.array([1.0, 1.0, 1.0])
        params = (10.0, 28.0, 8.0 / 3.0)
        dt = 0.01
        steps = 10
        result = integrate_lorenz(initial_point, params, dt, steps)
        assert isinstance(result, np.ndarray)
        assert result.shape == (steps, 3)

    def test_integrate_deterministic(self):
        initial_point = np.array([1.0, 1.0, 1.0])
        params = (10.0, 28.0, 8.0 / 3.0)
        dt = 0.01
        steps = 100
        result1 = integrate_lorenz(initial_point, params, dt, steps)
        result2 = integrate_lorenz(initial_point, params, dt, steps)
        assert_array_equal(result1, result2)


if __name__ == "__main__":
    pytest.main()
