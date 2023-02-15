import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous


class ashton_gen(rv_continuous):
    r"""An Ashton continuous random variable.

    The Ashton distribution divides [0, 1] into quartiles that are uniform
    distributions defined by the shape parameters $a$ (the asymmetry) and
    $h$ (high fraction).

    .. math::

        f(x, a, h) =
        \begin{cases}
            a \cdot h & 0 \lt x \leq \frac{1}{4} \\
            a \cdot (1 - h) & \frac{1}{4} \lt x \leq \frac{1}{2} \\
            (1 - a) \cdot (1 - h) & \frac{1}{2} \lt x \leq \frac{3}{4} \\
            (1 - a) \cdot h & \frac{3}{4} \lt x \lt 1 \\
        \end{cases}
    """

    def _argcheck(self, a, h):
        return np.all((a >= 0.0) & (a <= 1.0) & (h >= 0.0) & (h <= 1.0))

    def _pdf(self, x, a, h):
        a, h = np.broadcast_arrays(a, h)
        if a.size == 1:
            return self._scalar_pdf(x, a.item(), h.item())
        else:
            return [
                self._scalar_pdf(_x, _a, _h)
                for _x, _a, _h in zip(*np.broadcast_arrays(x, a, h))
            ]

    def _scalar_pdf(self, x, a, h):
        pdf = [
            a * h,
            a * (1.0 - h),
            (1.0 - a) * (1.0 - h),
            (1.0 - a) * h,
            (1.0 - a) * h,
        ]
        return np.array(pdf)[(x * 4).astype(int)] * 4
        # return pdf[int(x * 4)] * 4
        # return interp1d([0.0, 0.25, 0.5, 0.75, 1.0], pdf, kind="previous")(x) * 4.0

    def _cdf(self, x, a, h):
        a, h = np.broadcast_arrays(a, h)
        if a.size == 1:
            return self._scalar_cdf(x, a.item(), h.item())
        else:
            return [
                self._scalar_cdf(_x, _a, _h)
                for _x, _a, _h in zip(*np.broadcast_arrays(x, a, h))
            ]

    def _scalar_cdf(self, x, a, h):
        cdf = np.array([0.0, a * h, a, 1 - h * (1 - a), 1])
        return np.interp(x, [0.0, 0.25, 0.5, 0.75, 1.0], cdf)

    def _ppf(self, q, a, h):
        a, h = np.broadcast_arrays(a, h)
        if a.size == 1:
            return self._scalar_ppf(q, a.item(), h.item())
        else:
            return [
                self._scalar_ppf(_q, _a, _h)
                for _q, _a, _h in zip(*np.broadcast_arrays(q, a, h))
            ]

    def _scalar_ppf(self, q, a, h):
        ppf = np.array([0.0, a * h, a, 1 - h * (1 - a), 1])
        return np.interp(q, ppf, [0.0, 0.25, 0.5, 0.75, 1.0])


ashton = ashton_gen(a=0.0, b=1.0, name="ashton")


class WaveAngleGenerator:
    def __init__(self, asymmetry=0.8, high_fraction=0.2, rng=None):
        # def __init__(self, asymmetry=0.8, high_fraction=0.2, wave_climl=180, rng=None):
        """Generate incoming wave angles.

        Parameters
        ----------
        asymmetry: float, optional
            Fraction of waves approaching from the left, looking offshore (Ashton & Murray, 2006). Value typically
            varied in BRIE.
        high_fraction: float, optional
            Fraction of waves approaching at angles higher than 45 degrees from shore normal (Ashton & Murray, 2006).
            Value typically 0.2 in BRIE.

        Examples
        --------
        >>> import numpy as np
        >>> from AST.waves import WaveAngleGenerator
        >>> angles = WaveAngleGenerator()
        >>> angles.next()  # doctest: +SKIP
        array([1.0975858])

        >>> angles.next(samples=4)  # doctest: +SKIP
        array([-0.27482598,  1.25014028,  0.32893483, -0.87115524])

        Define an angle distribution where there are no high angle (i.e.
        no angles outside of -45 to 45 degrees).

        >>> angles = WaveAngleGenerator(asymmetry=0.5, high_fraction=0.0)
        >>> step = np.pi / 4.0
        >>> angles.pdf(np.deg2rad([-67.5, -22.5, 22.5, 67.5])) * step
        array([0. , 0.5, 0.5, 0. ])

        >>> angles.cdf(np.deg2rad([-90, -45, 0, 45, 90]))
        array([0. , 0. , 0.5, 1. , 1. ])
        """
        if asymmetry < 0.0 or asymmetry > 1.0:
            raise ValueError("wave angle asymmetry must be between 0 and 1")
        if high_fraction < 0.0 or high_fraction > 1.0:
            raise ValueError("fraction of high angles must be between 0 and 1")

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        # KA: using radians instead of degrees
        x = np.deg2rad(np.array([-90.0, -45.0, 0.0, 45.0, 90]))

        f = (
            np.array(
                [
                    0.0,
                    asymmetry * high_fraction,
                    asymmetry * (1.0 - high_fraction),
                    (1.0 - asymmetry) * (1.0 - high_fraction),
                    (1.0 - asymmetry) * high_fraction,
                ]
            )
            * 4.0
            / np.pi
        )

        # f = (
        #         np.array(
        #             [
        #                 asymmetry * high_fraction,
        #                 asymmetry * (1.0 - high_fraction),
        #                 (1.0 - asymmetry) * (1.0 - high_fraction),
        #                 (1.0 - asymmetry) * high_fraction,
        #                 (1.0 - asymmetry) * high_fraction,
        #             ]
        #         )
        #         * 4.0
        #         / np.pi
        # )

        self._wave_pdf = interp1d(x, f, kind="next", bounds_error=False, fill_value=0.0)
        self._wave_cdf = interp1d(
            x, np.cumsum(f) * np.pi / 4.0, bounds_error=False, fill_value=(0.0, 1.0)
        )
        self._wave_inv_cdf = interp1d(
            np.cumsum(f) * np.pi / 4.0, x, bounds_error=False, fill_value=np.nan
        )

    def pdf(self, angle):
        """Cumulative distribution function for wave angle.

        Parameters
        ----------
        angle: number or ndarray
            Angle(s) at which to evaluate the cdf [rad].

        Returns
        -------
        ndarray of float
            This is the normalized angular distribution of wave energy (Eq 39 in BRIE, from AM06).
        """
        return self._wave_pdf(angle)

    def cdf(self, angle):
        """Cumulative distribution function for wave angle.

        Parameters
        ----------
        angle: number or ndarray
            Angle(s) at which to evaluate the cdf [degree].

        Returns
        -------
        ndarray of float
            This is the normalized cumulative distribution of wave energy (Eq 25 in BRIE, from AM06).
        """
        return self._wave_cdf(angle)

    def next(self, samples=1):
        """Next wave angles from the distribution.

        Parameters
        ----------
        samples : int
            Number of wave angles to return.

        Returns
        -------
        ndarray of float
            Waves angles [radians].
        """

        # I don't want to extrapolate, so instead if the rng is below the interpolation bounds, I pick a new number
        # x = self._rng.random(samples)

        # while x < self._lower_bnd:
        #     x = self._rng.random(samples)

        return self._wave_inv_cdf(self._rng.random(samples))
        # return np.floor(self._wave_inv_cdf(x))
