"""BE: 2-halo template for the electron density, the same halo-model form with halo exclusion as the BD pressure template.

The BD pressure template (bd_twohalo.TwoHaloExcl.template_excl, read-only) is
  t(r | M, z) = b(M) int dM' n(M') b(M') S(M', M) int_{s >= f_ex (R200c + R200c')} d^3s xi_lin(s) F'(|r - s| | M'),
with Sheth-Mo-Tormen bias b, Sheth-Tormen mass function n, Eisenstein & Hu linear xi, neighbour profile F' truncated at
4 R200c', neighbours 10^10-10^16 Msun, f_ex = exclusion factor, and S = the ratio of the neighbour's to the host's profile
normalisation. For pressure F' = Battaglia+12 P_e/P200c and S = P200c(M')/P200c(M) = (M'/M)^(2/3).
For electrons (this module): F' = the Battaglia+16 AGN gas profile in the production electron units
E = n_e mu_XGP/(rho_crit f_b) (fullsnap_production/physics.py uses the B16 values as the electron starting point),
  E' = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a]^(-(b - 0.2)/a),  n0 = 4e3 m^0.29 (1+z)^-0.66,  a = 0.88 m^-0.03 (1+z)^0.19,
  b = 3.83 m^0.04 (1+z)^-0.025,  m = M'/1e14 Msun,
and S = 1, because E is normalised by rho_crit(z) f_b, which does not depend on the halo mass.
mean_term=True replaces b b' xi by (1 + b b' xi), adding the uncorrelated neighbour gas with the same exclusion hole.
The method below is a copy of template_excl with the normalisation ratio S made a property of the field; field='P'
reproduces the BD template exactly (checked in be_etemplate.py).
"""
import numpy as np

import bd_twohalo as TH

B16 = np.array([[4e3, .29, -.66], [.88, -.03, .19], [3.83, .04, -.025]])


class TwoHaloField(TH.TwoHaloExcl):
    def __init__(self, cosmo, log10m_range, field='E', **kw):
        self.field = field
        super().__init__(cosmo, log10m_range, **kw)

    def b12_pe(self, x, M, z):
        if getattr(self, 'field', 'P') == 'P':
            return super().b12_pe(x, M, z)
        lm, lz = np.log(M/1e14), np.log1p(z)
        n0, a, b = (B16[i, 0]*np.exp(B16[i, 1]*lm+B16[i, 2]*lz) for i in range(3))
        q = x/.5
        return n0*q**-.2*(1+q**a)**(-(b-.2)/a)

    def scale(self, Mp, M):
        return (Mp/M)**(2/3) if self.field == 'P' else np.ones_like(Mp)

    def template_excl(self, M, z, lo, hi, integral, exclusion=True, nr=6, ns=64, rex_factor=1.5, mean_term=False):
        lo, hi, integral = np.atleast_1d(lo), np.atleast_1d(hi), np.atleast_1d(integral)
        Mp = self.mass
        R = self.r200c(M, z)
        Rp = self.r200c(Mp, z)
        S = self.scale(Mp, M)
        weight = self.dndlnm(Mp, z)*self.bias(Mp, z)*S
        lnm = np.log(Mp)
        dl = np.gradient(lnm)
        dl[0] *= .5
        dl[-1] *= .5
        F = self.b12_pe(np.maximum(self.xq[None, :], 1e-12), Mp[:, None], z)
        q = np.concatenate([np.zeros((len(Mp), 1)),
                            np.cumsum(.5*np.diff(self.xq)[None, :]*(F[:, 1:]*self.xq[None, 1:]+F[:, :-1]*self.xq[None, :-1]), axis=1)],
                           axis=1)
        qmax = q[:, -1]

        def Q(u):
            x = u/Rp.reshape((-1,)+(1,)*(u.ndim-1))
            out = np.empty_like(u)
            for i in range(len(Mp)):
                out[i] = np.interp(x[i], self.xq, q[i], right=qmax[i])
            return out*(Rp**2).reshape((-1,)+(1,)*(u.ndim-1))

        bM = self.bias(np.array([M]), z)[0]
        tr, wr = TH._gl(nr)
        ts, wsg = TH._gl(ns)
        out = np.empty(len(lo))
        for j, (a, b, it) in enumerate(zip(lo, hi, integral)):
            a = 0. if it else a
            na = 24 if it else nr
            tr_, wr_ = (tr, wr) if na == nr else TH._gl(na)
            r = (a+(b-a)*tr_)*R
            vw = wr_*r**2
            vw /= vw.sum()
            smin = np.maximum(rex_factor*(R+Rp) if exclusion else 0., np.maximum(r.min()-self.x_trunc*Rp, 0.))
            smax = r.max()+self.x_trunc*Rp
            s = smin[:, None]+(smax-smin)[:, None]*ts[None, :]
            ws = (smax-smin)[:, None]*wsg[None, :]
            ss = s[:, :, None]
            rr = r[None, None, :]
            K = (Q(ss+rr)-Q(np.abs(ss-rr)))/(2*ss*rr)
            Kbar = (K*vw[None, None, :]).sum(axis=-1)
            I = (ws*4*np.pi*s**2*self.xi(s, z)*Kbar).sum(axis=1)
            mean = bM*np.sum(dl*weight*I)
            if mean_term:
                I0 = (ws*4*np.pi*s**2*Kbar).sum(axis=1)
                mean = mean+np.sum(dl*self.dndlnm(Mp, z)*S*I0)
            out[j] = mean*(4*np.pi/3*b**3 if it else 1.)
        return out
