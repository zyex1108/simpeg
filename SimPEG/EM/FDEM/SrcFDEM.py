from SimPEG import Survey, Problem, Utils, np, sp
from scipy.constants import mu_0
from SimPEG.EM.Utils import *
from SimPEG.Utils import Zero

class BaseSrc(Survey.BaseSrc):
    """
    Base source class for FDEM Survey
    """

    freq = None
    # rxPair = RxFDEM
    integrate = True

    def eval(self, prob):
        """
        Evaluate the source terms.
        - :math:`s_m` : magnetic source term
        - :math:`s_e` : electric source term

        :param Problem prob: FDEM Problem
        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: tuple with magnetic source term and electric source term
        """
        s_m = self.s_m(prob)
        s_e = self.s_e(prob)
        return s_m, s_e

    def evalDeriv(self, prob, v=None, adjoint=False):
        """
        Derivatives of the source terms with respect to the inversion model
        - :code:`s_mDeriv` : derivative of the magnetic source term
        - :code:`s_eDeriv` : derivative of the electric source term

        :param Problem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: tuple with magnetic source term and electric source term derivatives times a vector
        """
        if v is not None:
            return self.s_mDeriv(prob, v, adjoint), self.s_eDeriv(prob, v, adjoint)
        else:
            return lambda v: self.s_mDeriv(prob, v, adjoint), lambda v: self.s_eDeriv(prob, v, adjoint)

    def bPrimary(self, prob):
        """
        Primary magnetic flux density

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def hPrimary(self, prob):
        """
        Primary magnetic field

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        return Zero()

    def ePrimary(self, prob):
        """
        Primary electric field

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary electric field
        """
        return Zero()

    def jPrimary(self, prob):
        """
        Primary current density

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary current density
        """
        return Zero()

    def s_m(self, prob):
        """
        Magnetic source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        return Zero()

    def s_e(self, prob):
        """
        Electric source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        return Zero()

    def s_mDeriv(self, prob, v, adjoint=False):
        """
        Derivative of magnetic source term with respect to the inversion model

        :param Problem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of magnetic source term derivative with a vector
        """

        return Zero()

    def s_eDeriv(self, prob, v, adjoint=False):
        """
        Derivative of electric source term with respect to the inversion model

        :param Problem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of electric source term derivative with a vector
        """
        return Zero()


class RawVec_e(BaseSrc):
    """
    RawVec electric source. It is defined by the user provided vector s_e

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.array s_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [True]
    """

    def __init__(self, rxList, freq, s_e, integrate=True): #, ePrimary=None, bPrimary=None, hPrimary=None, jPrimary=None):
        self._s_e = np.array(s_e, dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate

        BaseSrc.__init__(self, rxList)

    def s_e(self, prob):
        """
        Electric source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if prob._formulation is 'EB' and self.integrate is True:
            return prob.Me * self._s_e
        return self._s_e


class RawVec_m(BaseSrc):
    """
    RawVec magnetic source. It is defined by the user provided vector s_m

    :param float freq: frequency
    :param rxList: receiver list
    :param numpy.array s_m: magnetic source term
    :param bool integrate: Integrate the source term (multiply by Me) [True]
    """

    def __init__(self, rxList, freq, s_m, integrate=True):  #ePrimary=Zero(), bPrimary=Zero(), hPrimary=Zero(), jPrimary=Zero()):
        self._s_m = np.array(s_m, dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate

        BaseSrc.__init__(self, rxList)

    def s_m(self, prob):
        """
        Magnetic source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if prob._formulation is 'HJ' and self.integrate is True:
            return prob.Me * self._s_m
        return self._s_m


class RawVec(BaseSrc):
    """
    RawVec source. It is defined by the user provided vectors s_m, s_e

    :param rxList: receiver list
    :param float freq: frequency
    :param numpy.array s_m: magnetic source term
    :param numpy.array s_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [True]
    """
    def __init__(self, rxList, freq, s_m, s_e, integrate=True):
        self._s_m = np.array(s_m, dtype=complex)
        self._s_e = np.array(s_e, dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate
        BaseSrc.__init__(self, rxList)

    def s_m(self, prob):
        """
        Magnetic source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if prob._formulation is 'HJ' and self.integrate is True:
            return prob.Me * self._s_m
        return self._s_m

    def s_e(self, prob):
        """
        Electric source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if prob._formulation is 'EB' and self.integrate is True:
            return prob.Me * self._s_e
        return self._s_e


class MagDipole(BaseSrc):
    """
    Point magnetic dipole source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    This approach uses a primary-secondary in frequency. Here we show the
    derivation for E-B formulation noting that similar steps are followed for
    the H-J formulation.

    .. math::
        \mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\\\
            {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}}

    We split up the fields and :math:`\mu^{-1}` into primary (:math:`\mathbf{P}`) and secondary (:math:`\mathbf{S}`) components

    - :math:`\mathbf{e} = \mathbf{e^P} + \mathbf{e^S}`
    - :math:`\mathbf{b} = \mathbf{b^P} + \mathbf{b^S}`
    - :math:`\\boldsymbol{\mu}^{\mathbf{-1}} = \\boldsymbol{\mu}^{\mathbf{-1}^\mathbf{P}} + \\boldsymbol{\mu}^{\mathbf{-1}^\mathbf{S}}`

    and define a zero-frequency primary problem, noting that the source is
    generated by a divergence free electric current

    .. math::
        \mathbf{C} \mathbf{e^P} = \mathbf{s_m^P} = 0 \\\\
            {\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^P} \mathbf{b^P} - \mathbf{M_{\sigma}^e} \mathbf{e^P} = \mathbf{M^e} \mathbf{s_e^P}}

    Since :math:`\mathbf{e^P}` is curl-free, divergence-free, we assume that there is no constant field background, the :math:`\mathbf{e^P} = 0`, so our primary problem is

    .. math::
        \mathbf{e^P} =  0 \\\\
            {\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^P} \mathbf{b^P} = \mathbf{s_e^P}}

    Our secondary problem is then

    .. math::
        \mathbf{C} \mathbf{e^S} + i \omega \mathbf{b^S} = - i \omega \mathbf{b^P} \\\\
            {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b^S} - \mathbf{M_{\sigma}^e} \mathbf{e^S} = -\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^S} \mathbf{b^P}}

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu=mu_0):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.moment = moment
        self.mu = mu
        self.integrate = False
        BaseSrc.__init__(self, rxList)

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        formulation = prob._formulation

        if formulation is 'EB':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif formulation is 'HJ':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T


        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            a = MagneticDipoleVectorPotential(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)

        else:
            srcfct = MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            ay = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            az = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            a = np.concatenate((ax, ay, az))

        return C*a

    def hPrimary(self, prob):
        """
        The primary magnetic field from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        return 1./self.mu * b

    def s_m(self, prob):
        """
        The magnetic source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        b_p = self.bPrimary(prob)
        if prob._formulation is 'HJ':
            b_p = prob.Me * b_p
        return -1j*omega(self.freq)*b_p

    def s_e(self, prob):
        """
        The electric source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return Zero()
        else:
            formulation = prob._formulation

            if formulation is 'EB':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif formulation is 'HJ':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class MagDipole_Bfield(BaseSrc):

    """
    Point magnetic dipole source calculated with the analytic solution for the
    fields from a magnetic dipole. No discrete curl is taken, so the magnetic
    flux density may not be strictly divergence free.

    This approach uses a primary-secondary in frequency in the same fashion as the MagDipole.

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu = mu_0):
        self.freq = float(freq)
        self.loc = loc
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.orientation = orientation
        self.moment = moment
        self.mu = mu
        BaseSrc.__init__(self, rxList)

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from the analytic solution for magnetic fields from a dipole

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        formulation = prob._formulation

        if formulation is 'EB':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl

        elif formulation is 'HJ':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl.T

        srcfct = MagneticDipoleFields
        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx,bz))
        else:
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            by = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx,by,bz))

        return b

    def hPrimary(self, prob):
        """
        The primary magnetic field from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        return 1/self.mu * b

    def s_m(self, prob):
        """
        The magnetic source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        if prob._formulation is 'HJ':
            b = prob.Me * b
        return -1j*omega(self.freq)*b

    def s_e(self, prob):
        """
        The electric source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return Zero()
        else:
            formulation = prob._formulation

            if formulation is 'EB':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif formulation is 'HJ':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class CircularLoop(BaseSrc):
    """
    Circular loop magnetic source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    This approach uses a primary-secondary in frequency in the same fashion as the MagDipole.

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', radius=1., mu=mu_0):
        self.freq = float(freq)
        self.orientation = orientation
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.radius = radius
        self.mu = mu
        self.loc = loc
        self.integrate = False
        BaseSrc.__init__(self, rxList)

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        formulation = prob._formulation

        if formulation is 'EB':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif formulation is 'HJ':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T

        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            a = MagneticLoopVectorPotential(self.loc, gridY, 'y', moment=self.radius, mu=self.mu)

        else:
            srcfct = MagneticLoopVectorPotential
            ax = srcfct(self.loc, gridX, 'x', self.radius, mu=self.mu)
            ay = srcfct(self.loc, gridY, 'y', self.radius, mu=self.mu)
            az = srcfct(self.loc, gridZ, 'z', self.radius, mu=self.mu)
            a = np.concatenate((ax, ay, az))

        return C*a

    def hPrimary(self, prob):
        """
        The primary magnetic field from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        return 1./self.mu*b

    def s_m(self, prob):
        """
        The magnetic source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        if prob._formulation is 'HJ':
            b =  prob.Me *  b
        return -1j*omega(self.freq)*b

    def s_e(self, prob):
        """
        The electric source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return Zero()
        else:
            formulation = prob._formulation

            if formulation is 'EB':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl


            elif formulation is 'HJ':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class PrimSec(BaseSrc):
    """
    Primary-Secondary source in the physical properties. A primary problem is
    first solved, and the fields from this problem are used to construct a
    source term for the secondary problem. Either a mesh and
    fields need to be provided or a prob and a survey.

    For the EB formulation, we start the derivation from Maxwell's equations:

    .. math::
        \\nabla \\times \\vec{E} + i \omega \\vec{B} = \\vec{s_m} \\\\
        \\nabla \\times \\mu^{-1} \\vec{B} - \sigma \\vec{E} = \\vec{s_e}

    we consider the physical properties, fields, and fluxes to be composed of
    two parts, a primary and a secondary:

        - :math:`\sigma   = \sigma_p + \sigma_s`
        - :math:`\mu^{-1} = \mu^{-1}_p + \mu^{-1}_s`
        - :math:`\\vec{E} = \\vec{E_p} + \\vec{E_s}`
        - :math:`\\vec{B} = \\vec{B_p} + \\vec{B_s}`

    and choose our primary such that

    .. math::
        \\nabla \\times \\vec{E}_p + i \omega \\vec{B}_p = \\vec{s_m} \\\\
        \\nabla \\times \\mu^{-1}_p \\vec{B}_p - \sigma_p \\vec{E}_p = \\vec{s_e}_p

    so the secondary problem is then

    .. math::
        \\nabla \\times \\vec{E}_s + i \omega \\vec{B}_s = 0 \\\\
        \\nabla \\times \\mu^{-1} \\vec{B}_s - \sigma \\vec{E}_s = - \\nabla \\times \\mu^{-1}_s \\vec{B}_p + \sigma_s \\vec{E}_p


    If instead, HJ formulation is considered, then we start off with

    .. math::
        \\nabla \\times \\rho \\vec{J} + i \omega \\mu \\vec{H} = \\vec{s_m} \\\\
        \\nabla \\times \\vec{H} - \\vec{J} = \\vec{s_e}

    and we define the primary secondary problem in terms of

        - :math:`\\rho  = \\rho_p + \\rho_s`
        - :math:`\mu = \mu_p + \mu_s`
        - :math:`\\vec{J} = \\vec{J_p} + \\vec{J_s}`
        - :math:`\\vec{H} = \\vec{H_p} + \\vec{H_s}`

    with the primary being defined by

    .. math::
        \\nabla \\times \\rho_p \\vec{J}_p + i \omega \\mu_p \\vec{H}_p = \\vec{s_m} \\\\
        \\nabla \\times \\vec{H}_p - \\vec{J}_p = \\vec{s_e}

    so the secondary problem is given by

    .. math::
        \\nabla \\times \\rho \\vec{J}_s + i \omega \\mu \\vec{H} = - \\nabla \\times \\rho_s \\vec{J}_p - i \omega \\mu_s \\vec{H}_p \\
        \\nabla \\times \\vec{H}_p - \\vec{J}_p = 0

    Note: if different meshes are employed for the primary and secondary
    problems, then we need to interpolate the fields from the primary mesh to
    the secondary mesh. We do this by always interpolating the field and
    computing a flux if need be in order to ensure that fluxes remain
    numerically divergence free.

    :param list rxList: Receiver list
    :param float freq: frequency
    :param numpy.array m: primary model
    :param Problem prob: primary problem
    :param Survey survey: primary survey
    """


    def __init__(self, rxList, freq, m, prob, survey):
        self.freq = float(freq)
        self.m = m
        self.prob = prob
        self.survey = survey
        self.fields = None

        if self.survey.ispaired:
            if self.survey.prob is not self.prob:
                raise Exception('The survey object is already paired to a problem. Use survey.unpair()')
        else:
            self.prob.pair(self.survey)

        self.mesh = self.prob.mesh
        self.prob.curModel = self.m

        BaseSrc.__init__(self, rxList)

    def MeSigma(self, prob):
        if getattr(self, '_MeSigma', None) is None:
            sigmaprimary = self.prob.curModel.sigma
            if self.mesh != prob.mesh:
                P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType='CC')
                sigmaprimary = P * sigmaprimary
            self._MeSigma = prob.mesh.getEdgeInnerProduct(sigmaprimary)
        return self._MeSigma

    def MfMui(self, prob):
        if getattr(self, '_MfMui', None) is None:
            muiprimary = self.prob.curModel.mui
            if self.mesh != prob.mesh and not isinstance(muiprimary,float): # if different meshes and mu is a vector --> need to interpolate
                P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType='CC')
                muiprimary = P * muiprimary
            self._MfMui = prob.mesh.getFaceInnerProduct(muiprimary)
        return self._MfMui

    def MfRho(self, prob):
        if getattr(self, '_MfRho', None) is None:
            rhoprimary = self.prob.curModel.rho
            if self.mesh != prob.mesh:
                P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType='CC')
                rhoprimary = P * rhoprimary
            self._MfRho = prob.mesh.getFaceInnerProduct(rhoprimary)
        return self._MfRho

    def MeMu(self, prob):
        if getattr(self, '_MeMu', None) is None:
            muprimary = self.prob.curModel.mu
            if self.mesh != prob.mesh and not isinstance(muiprimary,float): # if different meshes and mu is a vector --> need to interpolate
                P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType='CC')
                muprimary = P * muprimary
            self._MeMu = prob.mesh.getEdgeInnerProduct(muprimary)
        return self._MeMu

    # note if you switch from one formulation to another, but are using the same mesh, this will break
    def ePrimary(self,prob):
        if getattr(self, '_ePrimary', None) is None:
            if self.fields is None:
                self.fields = self.prob.fields(self.m)

            ePrimary = self.fields[:,'e']

            if self.mesh != prob.mesh:
                if self.prob._formulation == 'HJ':
                    P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType=prob._GLoc('e'), locTypeFrom='CCV')
                else:
                    P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType=prob._GLoc('e'))
                ePrimary = Utils.mkvc(P * ePrimary)
            self._ePrimary = Utils.mkvc(ePrimary)

        return self._ePrimary

    # note if you switch from one formulation to another, but are using the same mesh, this will break
    def bPrimary(self, prob):
        if getattr(self, '_bPrimary', None) is None:
            if self.fields is None:
                self.fields = self.prob.fields(self.m)

            if self.mesh == prob.mesh:
                bPrimary = self.fields[:,'b']
            else:
                bPrimary = prob.mesh.edgeCurl * self.ePrimary(prob)

            self._bPrimary = Utils.mkvc(bPrimary)

        return self._bPrimary

    # note if you switch from one formulation to another, but are using the same mesh, this will break
    def hPrimary(self, prob):
        if getattr(self, '_hPrimary', None) is None:
            if self.fields is None:
                self.fields = self.prob.fields(self.m)

            hPrimary = self.fields[:,'h']

            if self.mesh != prob.mesh:
                if self.prob._formulation == 'EB':
                    P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType=prob._GLoc('h'), locTypeFrom='CCV')
                else:
                    P = self.mesh.getInterpolationMatMesh2Mesh(prob.mesh, locType=prob._GLoc('h'))
                print P.shape, hPrimary.shape, prob._GLoc('h')
                hPrimary = Utils.mkvc(P * hPrimary)
            self._hPrimary = Utils.mkvc(hPrimary)

        return self._hPrimary

    # note if you switch from one formulation to another, but are using the same mesh, this will break
    def jPrimary(self, prob):
        if getattr(self, '_jPrimary', None) is None:
            if self.fields is None:
                self.fields = self.prob.fields(self.m)

            if self.mesh == prob.mesh:
                jPrimary = self.fields[:,'j']
            else:
                jPrimary = prob.mesh.edgeCurl * self.hPrimary(prob)

            self._jPrimary = Utils.mkvc(jPrimary)

        return self._jPrimary

    def s_e(self,prob):
        if prob._formulation == 'EB':
            # - \\nabla \\times \\mu^{-1}_s \\vec{B}_p + \sigma_s \\vec{E}_p
            s_e =  -prob.mesh.edgeCurl.T * ((prob.MfMui - self.MfMui(prob)) * self.bPrimary(prob)) +  (prob.MeSigma - self.MeSigma(prob)) * self.ePrimary(prob)
            return Utils.mkvc(s_e)
        else:
            return Zero()

    def s_eDeriv(self, prob, v, adjoint=False):
        if prob._formulation == 'EB':
            if adjoint is True:
                return prob.MeSigmaDeriv(self.ePrimary(prob)).T * v
            return prob.MeSigmaDeriv(self.ePrimary(prob)) * v
        else:
            return Zero()

    def s_m(self,prob):
        if prob._formulation == 'HJ':
            # - \\nabla \\times \\rho_s \\vec{J}_p - i \omega \\mu_s \\vec{H}_p
            s_m = - prob.mesh.edgeCurl.T * (prob.MfRho - self.MfRho(prob)) * self.jPrimary(prob) - 1j * omega(self.freq) * ((prob.MeMu - self.MeMu(prob)) * self.hPrimary(prob))
            return s_m
        else:
            return Zero()

    def s_mDeriv(self, prob, v, adjoint=False):
        if prob._formulation == 'HJ':
            if adjoint is True:
                return - prob.MfRhoDeriv(self.jPrimary(prob)).T * (prob.mesh.edgeCurl * v)
            return - prob.mesh.edgeCurl.T * (prob.MfRhoDeriv(self.jPrimary(prob)) * v)
        else:
            return Zero()



