"""Alongshore sediment transport

This module provides functions for diffusing sediment along a straight (non-complex) coast.
Formulations described in Nienhuis and Lorenzo-Trueba, 2019 [4]_ and stem from Ashton and Murray, 2006 [2]_,
Nienhuis et al., 2015 [1]_, and Komar, 1998 [3]_.

References
----------

.. [1] Jaap H. Nienhuis, Andrew D. Ashton, Liviu Giosan; What makes a delta wave-dominated?.
    Geology ; 43 (6): 511–514. doi: https://doi.org/10.1130/G36518.1
.. [2] Andrew D. Ashton, A. Brad Murray. High‐angle wave instability and emergent shoreline shapes:
    1. Modeling of sand waves, flying spits, and capes. Journal of Geophysical Research: Earth Surface 111.F4 (2006).
.. [3] P.D. Komar, 1998, Beach processes and sedimentation: Upper Saddle River, New Jersey, Prentice Hall , 544 p.
.. [4] Jaap H. Nienhuis, Jorge Lorenzo Trueba; Simulating barrier island response to sea level rise with the barrier
    island and inlet environment (BRIE) model v1.0 ; Geosci. Model Dev., 12, 4013–4030, 2019; https://doi.org/10.5194/gmd-12-4013-2019


Notes
---------
All calculations are performed with the domain perspective of looking onshore to offshore

"""
import numpy as np
import scipy.constants
import scipy.sparse

SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.0


def calc_alongshore_transport_k(
    gravity=scipy.constants.g,
    n=1.0,
    rho_water=1050.0,
    gamma_b=0.78,
):
    r"""Calculate alongshore transport diffusion coefficient.

    The diffusion coefficient is calculated from Nienhuis, Ashton, Giosan, 2015 [1]_ .
    Note that the Ashton, 2006 value for *k* is incorrect.

    Parameters
    ----------
    gravity : float, optional
        Acceleration due to gravity [m/s^2].
    n : float, optional
        Ratio of group velocity to phase velocity of the breaking waves
        (1 in shallow water).
    rho_water: float, optional
        Density of water [kg / m^3].
    gamma_b: float, optional
        Ratio of breaking wave height and water depth.

    Returns
    -------
    float
        Empirical constant for alongshore sediment transport.

    Notes
    -----

    The sediment transport constant, :math:`K_1`, is calculated as follows,

    .. math::

        K_1 = 5.3 \cdot 10^{-6} K \left( \frac{1}{2n} \right)^{6 \over 5} \left( \frac{\sqrt{g \gamma_b}} {2 \pi} \right)^{1 \over 5}

    where:

    .. math::

        K = 0.46 \rho g^{3 \over 2}

    """
    return (
        5.3e-6
        # * 0.46  # I'm not sure about this factor
        * rho_water
        * gravity ** 1.5
        * (1 / (2 * n)) ** 1.2
        * (np.sqrt(gravity * gamma_b) / (2 * np.pi)) ** 0.2
    )


def calc_shoreline_angles(y, spacing=1.0, out=None):
    r"""Calculate shoreline angles.

    Given a series of coastline positions, `y`, with equal spacing
    of points, calculate coastline angles with the *x*-axis. Angles
    at first and last points are calculated using wrap-around
    boundaries. This angle array corresponds to theta in Figure 1a of
    Ashton and Murray 2006. Note in BRIE, x_s is oriented so we look
    from offshore to onshore (positive y is landward mvmt).

    Parameters
    ----------
    y : array of float
        Y-positions of the shoreline [m].
    spacing : float
        Spacing between shoreline segments [m].
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned.

    Returns
    -------
    angles : array of float
        Angle of shoreline [rads].

    Examples
    --------
    >>> import numpy as np
    >>> from AST.alongshore_transporter import calc_shoreline_angles

    >>> y = [0.0, 0.0, 0.0, 0.0, 0.0]
    >>> calc_shoreline_angles(y)
    array([0., 0., 0., 0., 0.])

    Angles are measures with respect to the x-axis.

    >>> y = [0.0, 0.0, 1.0, 0.0, 0.0]
    >>> np.rad2deg(calc_shoreline_angles(y))
    array([  0.,  45., -45.,   0.,   0.])

    Angles at the ends are calculated as wrap-around boundaries.

    >>> y = [1.0, 0.0, 0.0, 0.0, 0.0]
    >>> np.rad2deg(calc_shoreline_angles(y))
    array([-45.,   0.,   0.,   0.,  45.])
    """
    return np.arctan2(np.diff(y, append=y[0]), spacing, out=out)  # dx/dy


def calc_coast_qs(wave_angle, wave_height=1.0, wave_period=10.0):
    r"""Calculate coastal alongshore sediment transport for a given incoming wave angle.

    Parameters
    ----------
    wave_angle: float or array of float
        Incoming wave angle relative to local shoreline normals [rad]. That is, a
        value of 0 means approaching waves are normal to the coast, negative
        values means waves approaching from the right, and positive from
        the left.
    wave_height: float, optional
        Height of incoming waves [m].
    wave_period: float, optional
        Period of incoming waves [s].

    Returns
    -------
    float or array of float
        Coastal qs [m3 / yr]

    Notes
    -----

    Alongshore sediment transport is computed using the CERC or Komar (Komar, 1998 [2]_ ) formula, reformulated into deep-water wave properties (Ashton and Murray, 2006 [3]_ ) by back-refracting the waves over shore-parallel contours, which yields:

    .. math::

        Q_s = K_1 \cdot H_s^{12/5} T^{1/5} \cos^{6/5}\left( \Delta \theta \right) \sin \left(\Delta \theta\right)

    where :math:`H_s` is the offshore deep-water significant wave height (in meters), :math:`T` is the wave period (in seconds), and :math:`\Delta \theta` is the deep-water wave approach angle relative to the local shoreline orientation (rads).

    References
    ----------
    .. [2] Komar P.D., 1998, Beach processes and sedimentation: Upper Saddle River, New Jersey, Prentice Hall , 544 p.

    .. [3] Ashton A.D. Murray A.B., 2006, High-angle wave instability and emergent shoreline shapes: 1. Modeling of sand waves, flying spits, and capes: Journal of Geophysical Research , v. 111, F04011, doi:10.1029/2005JF000422.
    """
    return (
        wave_height ** 2.4
        * (wave_period ** 0.2)
        * SECONDS_PER_YEAR
        * AlongshoreTransporter.K
        * (np.cos(wave_angle) ** 1.2)
        * np.sin(wave_angle)
    )  # [m3/yr]


def calc_inlet_alongshore_transport(
    wave_angle, shoreline_angle=0.0, wave_height=1.0, wave_period=10.0
):
    r"""Calculate alongshore transport along a coastline for a single wave angle. Only used in inlet calculations within
    BRIE.

    Parameters
    ----------
    wave_angle: float
        Incoming wave angle as measured counter-clockwise from the
        positive x-axis [rads].
    shoreline_angle: float or array of float, optional
        Angle of shoreline with respect to the positive x-axis [rads].
    wave_height: float, optional
        Incoming wave height [m].
    wave_period: float, optional
        Incoming wave period [s].

    Returns
    -------
    float or array of float
        Alongshore transport along the shoreline.
    """
    wave_angle_wrt_shoreline = np.clip(
        # np.pi / 2.0 + shoreline_angle - wave_angle,
        wave_angle - shoreline_angle,
        a_min=-np.pi / 2.0,
        a_max=np.pi / 2.0,
    )

    return calc_coast_qs(
        wave_angle_wrt_shoreline, wave_height=wave_height, wave_period=wave_period
    )


def calc_coast_diffusivity(
    wave_pdf,
    shoreline_angles,
    wave_height=1.0,
    wave_period=10.0,
    berm_ele=2.0,
    n_bins=181,
):
    r"""Calculate sediment diffusion along a coastline. Corresponds to Equations 37-39 in NLT19 [1]_, with formulations from
    AM06 [2]_.

    .. math::

        D \left( \theta \right) = k/(H_b+D_T) \cdot H_0 ^{12/5} T^{1/5} \cdot [E \left( \phi_0 \right) * \psi \left( \phi_0 - \theta \right)]

    where :math:`E\left( \phi_0 \right)` is the normalized angular distribution of wave energy, and :math:`\psi \left( \phi_0 - \theta \right)` is the angle depdendence of diffusivity.


    References
    ----------
    .. [1] Jaap H. Nienhuis, Jorge Lorenzo Trueba; Simulating barrier island response to sea level rise with the barrier
    island and inlet environment (BRIE) model v1.0 ; Geosci. Model Dev., 12, 4013–4030, 2019; https://doi.org/10.5194/gmd-12-4013-2019

    .. [2] Ashton A.D. Murray A.B., 2006, High-angle wave instability and emergent shoreline shapes: 1. Modeling of sand waves, flying spits, and capes: Journal of Geophysical Research , v. 111, F04011, doi:10.1029/2005JF000422.


    Parameters
    ----------
    wave_pdf: func
        Probability density function of incoming waves defined for wave
        angles from -pi / 2 to pi / 2.
    shoreline_angles: float
        Angle of shoreline with respect to the positive x-axis [rads].
    wave_height: float, optional
        Height of incoming waves [m].
    wave_period: float, optional
        Period of incoming waves [s].
    berm_ele: float, optional
        Berm elevation [m]
    n_bins: float, optional
        The number of bins used for the wave resolution: if 181 and [-90,90] in angle array below,
        the wave angles are in the middle of the bins,
        symmetrical about zero, spaced by 1 degree
    """

    # all_angles, step = np.linspace(-89.5, 89.5, n_bins, retstep=True)
    # all_angles = np.deg2rad(all_angles)
    all_angles, step = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_bins, retstep=True)

    d_sf = 8.9 * wave_height

    # e_phi_0 = wave_pdf(all_angles) * np.deg2rad(step)
    e_phi_0 = wave_pdf(all_angles) * step

    # KA: don't understand the negative here, but it works
    diff_phi0_theta = (
        -(
            AlongshoreTransporter.K
            / (berm_ele + d_sf)
            * wave_height ** 2.4
            * wave_period ** 0.2
        )
        * SECONDS_PER_YEAR
        # * (np.cos(delta_angles) ** 0.2)
        # * (1.2 * np.sin(delta_angles) ** 2 - np.cos(delta_angles) ** 2),
        * (np.cos(all_angles) ** 0.2)
        * (1.2 * np.sin(all_angles) ** 2 - np.cos(all_angles) ** 2)
    )

    # we convolve the normalized angular distribution of wave energy with the (relative wave) angle dependence
    # of the diffusivity
    # coast_diff = np.convolve(e_phi_0, diff_phi_0, mode="same")
    y = np.convolve(e_phi_0, diff_phi0_theta, mode="full")

    # KA: the "same" method differs in Matlab and Numpy; here we pad and slice out the "same" equivalent
    npad = len(diff_phi0_theta) - 1
    first = npad - npad // 2
    coast_diff_phi0_theta = y[
        first : first + len(e_phi_0)
    ]  # this is D above, for all relative wave angles

    # KA: why minus shoreline angles? I think because coast_diff_phi0_theta assumes a straight coastline (theta = 0) and we need to
    # evaluate at phi_0 - theta (i.e., the relative wave angle array for a non-straight shoreline)
    coast_diff = np.interp(
        -shoreline_angles, all_angles, coast_diff_phi0_theta
    )  # this is D above, evaluated at theta
    # return np.interp(shoreline_angles, all_angles, y) * np.sign(-wave_angle)
    # return np.interp(-wave_angle, all_angles, y)  # * np.sign(-wave_angle)

    return coast_diff, coast_diff_phi0_theta  # [m^2/s]


def _build_tridiagonal_matrix(diagonal, lower=None, upper=None):
    """Build a tridiagonal matrix with wrap-around boundaries.

    Parameters
    ----------
    values_at_node: array of float
        Values to place along the diagonals.

    Examples
    --------
    >>> from AST.alongshore_transporter import _build_tridiagonal_matrix
    >>> _build_tridiagonal_matrix([1.0, 2.0, 3.0, 4.0]).toarray()
    array([[1., 1., 0., 1.],
           [2., 2., 2., 0.],
           [0., 3., 3., 3.],
           [4., 0., 4., 4.]])

    >>> _build_tridiagonal_matrix(
    ...     [1.0, 2.0, 3.0, 4.0],
    ...     lower=[11.0, 12.0, 13.0, 14.0],
    ...     upper=[21.0, 22.0, 23.0, 24.0],
    ... ).toarray()
    array([[ 1., 21.,  0., 11.],
           [12.,  2., 22.,  0.],
           [ 0., 13.,  3., 23.],
           [24.,  0., 14.,  4.]])
    """
    if lower is None:
        lower = diagonal
    if upper is None:
        upper = diagonal
    n_rows = n_cols = len(diagonal)

    mat = scipy.sparse.spdiags(
        [np.r_[lower[1:], 0], diagonal, np.r_[0, upper[:-1]]],
        [-1, 0, 1],
        n_rows,
        n_cols,
    ).tolil()

    mat[0, -1] = lower[0]
    mat[-1, 0] = upper[-1]

    return mat


def _build_matrix(
    shoreline_x,
    wave_distribution,
    dy=1.0,
    wave_height=1.0,
    wave_period=10.0,
    dt=1.0,
    dx_dt=0,
):
    r"""UPDATE THIS

    Parameters
    ----------
    wave_angle: float
        Incoming wave angle as measured counter-clockwise from the
        positive x-axis [rads].
    shoreline_angle: float or array of float, optional
        Angle of shoreline with respect to the positive x-axis [rads].
    wave_height: float, optional
        Incoming wave height [m].
    wave_period: float, optional
        Incoming wave period [s].

    Returns
    -------
    float or array of float
        Alongshore transport along the shoreline.
    """

    shoreline_angles = calc_shoreline_angles(shoreline_x, spacing=dy)

    coast_diff, _ = calc_coast_diffusivity(
        wave_distribution.pdf,
        # np.pi / 2.0 - shoreline_angles, # Use shoreline angles???
        # -shoreline_angles,  # Use shoreline angles??? # KA: I don't think this should be negative
        shoreline_angles,
        wave_height=wave_height,
        wave_period=wave_period,
    )

    # this is beta in Equation 41 of NLT19
    # NOTE: Jaap updated on May 27, 2020 to force shoreline diffusivity to be greater than zero. Not sure I understand
    # why diffusivity needs to be greater than zero (it doesn't have to be theoretically).
    # r_ipl = np.clip(
    #     coast_diff
    #     * dt
    #     / (2.0 * dy ** 2),
    #     a_min=0.0,
    #     a_max=None,
    # )

    r_ipl = coast_diff * dt / (2.0 * dy ** 2)

    # Set non-periodic boundary conditions
    r_ipl[0] = 0
    r_ipl[-1] = 0

    mat = _build_tridiagonal_matrix(1.0 + 2.0 * r_ipl, lower=-r_ipl, upper=-r_ipl)

    rhs = (
        shoreline_x
        + r_ipl
        * np.diff(
            shoreline_x,
            n=2,
            prepend=shoreline_x[-1:],
            append=shoreline_x[:1],
        )
        + dx_dt
    )

    return mat.tocsc(), rhs, r_ipl


class AlongshoreTransporter:

    """Transport sediment along a coast.

    Examples
    --------
    >>> from AST.alongshore_transporter import AlongshoreTransporter
    >>> transporter = AlongshoreTransporter([0.0, 0.0, 1.0, 0.0, 0.0])
    >>> transporter.update()
    """

    K = calc_alongshore_transport_k()

    def __init__(
        self,
        shoreline_x,
        alongshore_section_length=1.0,
        time_step=1.0,
        change_in_shoreline_x=0.0,
        wave_height=1.0,
        wave_period=10.0,
        wave_angle=0.0,
        wave_distribution=None,
    ):
        """The AlongshoreTransporter module.

        Parameters
        ----------
        shoreline_x: array of float
            A shoreline position [m].
        alongshore_section_length: float, optional
            Length of each alongshore section [m].
        time_step: float, optional
            Time step of the numerical model [y].
        change_in_shoreline_x: float or array of float, optional
            Change in shoreline x position (accretion/erosion) [m].
        wave_height: float, optional
            Mean offshore significant wave height [m].
        wave_period: float, optional
            Mean wave period [s].
        wave_angle: float or array of float, optional
            Incoming wave angle relative to local shoreline normals [rad]. That is, a
            value of 0 means approaching waves are normal to the coast, negative
            values means waves approaching from the right, and positive from
            the left [deg]
        wave_distribution: a scipy distribution
        """

        self._shoreline_x = np.asarray(shoreline_x, dtype=float)
        self._dy = alongshore_section_length
        self._dt = time_step
        self._dx_dt = change_in_shoreline_x
        self._wave_height = wave_height
        self._wave_period = wave_period
        self._wave_angle = wave_angle

        if wave_distribution is None:
            wave_distribution = scipy.stats.uniform(loc=-np.pi / 2.0, scale=np.pi)
        self._wave_distribution = wave_distribution

        self._shoreline_angles = calc_shoreline_angles(
            self._shoreline_x, spacing=self._dy
        )

        self._time = 0.0

        # self._q_s = np.empty_like(shoreline_x)

    # def _build_matrix(self, dt=1.0):
    #     shoreline_angles = self._shoreline_angles
    #
    #
    #     r_ipl = calc_coast_diffusivity(
    #         self._wave_distribution.pdf,
    #         # np.pi / 2.0 - shoreline_angles, # Use shoreline angles???
    #         #- shoreline_angles,  # Use shoreline angles???
    #         shoreline_angles, # Use shoreline angles???
    #         wave_height=self._wave_height,
    #         wave_period=self._wave_period,
    #     ) * dt / (2.0 * self._dy ** 2)
    #
    #     mat = _build_tridiagonal_matrix(1.0 + 2.0 * r_ipl, lower=-r_ipl, upper=-r_ipl)
    #
    #     rhs = (
    #             self._shoreline_x
    #             + r_ipl
    #             * np.diff(
    #         self._shoreline_x,
    #         n=2,
    #         prepend=self._shoreline_x[-1:],
    #         append=self._shoreline_x[:1],
    #     )
    #         # + self._x_s_dt
    #     )
    #
    #     return mat.tocsc(), rhs

    def update(self):

        self._time += self._dt

        # self._wave_angle = self._wave_distribution.rvs(size=1)

        # self._q_s[:] = (
        #     calc_inlet_alongshore_transport(
        #         self._wave_angle,
        #         shoreline_angle=self._shoreline_angles,
        #         wave_height=self._wave_height,
        #         wave_period=self._wave_period,
        #     )
        #     * dt
        # )

        # calculates diffusivity and then returns the tridiagonal matrix and right-hand-side of Equation 41 in NLT19
        mat, rhs, _ = _build_matrix(
            self._shoreline_x,
            self._wave_distribution,
            dy=self._dy,
            wave_height=self._wave_height,
            wave_period=self._wave_period,
            dt=self._dt,
            dx_dt=self._dx_dt,
        )

        # invert the tridiagonal matrix to solve for the new shoreline position
        self._shoreline_x[:] = scipy.sparse.linalg.spsolve(mat, rhs)

        # calc_shoreline_angles(self._shoreline_x, self._dy, out=self._shoreline_angles)

    @property
    def shoreline_x(self):
        return self._shoreline_x

    @property
    def wave_height(self):
        return self._wave_height

    @wave_height.setter
    def wave_height(self, new_val):
        if new_val < 0.0:
            raise ValueError("wave height must be non-negative")
        self._wave_height = new_val

    @property
    def wave_period(self):
        return self._wave_period

    @wave_period.setter
    def wave_period(self, new_val):
        if new_val <= 0.0:
            raise ValueError("wave period must be positive")
        self._wave_period = new_val

    @property
    def wave_angle(self):
        return self._wave_angle

    @property
    def wave_pdf(self):
        return self._wave_distribution.pdf

    @property
    def shoreline_angles(self):
        return self._shoreline_angles