'''Phil Hopkins thought experiment which shows the volume approximation
property of SPH Kernel vs Sheperd filtered Kernel.
'''
import numpy as np
from compyle.api import declare
from compyle.utils import ArgumentParser
import matplotlib.pyplot as plt

from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group
from pysph.tools.sph_evaluator import SPHEvaluator


def create_domain(length):
    return DomainManager(
        xmin=0, xmax=length, ymin=0, ymax=length, periodic_in_x=True,
        periodic_in_y=True
    )


class KernelMomentsDaughter(Equation):
    def __init__(self, dest, sources, no_of_daughters):
        self.nod = no_of_daughters
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_wij_D):
        i = declare('int')
        for i in range(self.nod):
            d_wij_D[d_idx*self.nod + i] = 0.0

    def loop(self, d_idx, s_gid, s_idx, d_wij_D, XIJ, RIJ, WJ):
        d_wij_D[d_idx*self.nod + s_gid[s_idx]] += WJ


def create_particles(length, dx, h, dim, no_of_daughters, seed=123, test=False):
    _x = np.arange(dx/2, length, dx)

    np.random.seed(seed)
    if dim == 2:
        x, y = np.meshgrid(_x, _x)
        xd, yd, zd = np.random.rand(3, no_of_daughters)
        xd = dx/2 + xd * (length - dx/2)
        yd = dx/2 + yd * (length - dx/2)
        zd = dx/2 + zd * (length - dx/2)
        if test:
            xd = [0.2, 0.6, 0.25]
            yd = [0.75, 0.45, 0.15]
            zd = [0, 0, 0]
        m = h**dim
        dummy = get_particle_array(name='dummy', x=x, y=y, h=h)
        sources = get_particle_array(name='sources', x=xd, y=yd, h=h, m=m)
    else:
        raise NotImplementedError("1-D and 3-D is not implemented.")

    np.random.seed(1)
    factor = dx * 2
    dummy.x += np.random.random(dummy.x.shape) * factor
    dummy.y += np.random.random(dummy.x.shape) * factor

    dummy.add_property('wij_D', stride=no_of_daughters)
    sources.add_property('gid', type='int')
    sources.gid[:] = np.arange(no_of_daughters, dtype=int)

    dummy.add_property('sum_wij')

    props = ['ravg', 'vmax', 'awhat', 'n_nbrs', 'auhat', 'mT', 'avhat']
    for prop in props:
        sources.add_property(prop)

    # Uncomment this to check the interpolation and SPH particle locations.
    # plt.scatter(dummy.x, dummy.y, s=1)
    # plt.scatter(sources.x, sources.y, s=40, c='r')
    # plt.show()
    return dummy, sources


def create_equations(length, dummy, sources, dim, no_of_daughters, kernel,
                     shift=False):
    eqns = []
    eq = []
    eq.append(
        KernelMomentsDaughter(dummy.name, [sources.name], no_of_daughters)
    )
    eqns.append(Group(equations=eq))

    domain_mng = create_domain(length)
    sph_eval = SPHEvaluator(
        arrays=[sources, dummy], equations=eqns, dim=dim,
        kernel=kernel, domain_manager=domain_mng
    )
    return sph_eval


def compute_factors(length=1.0, pixels=100, hdx=30, dim=2, no_of_daughters=3,
                    kernel=None, plot=False, shepard=False,
                    test=False, seed=123):
    if kernel is None:
        from pysph.base.kernels import QuinticSpline as Kernel
        kernel = Kernel(dim=dim)
    dx = 1.0/pixels
    h = hdx * dx

    msg = "Please make sure no. of SPH points is a multiple of 3."
    assert no_of_daughters%3 == 0, msg

    dummy, sources = create_particles(length, dx, h, dim, no_of_daughters,
                                      test=test, seed=seed)

    sph_eval = create_equations(length, dummy, sources, dim,
                                no_of_daughters, kernel)
    sph_eval.update()
    sph_eval.evaluate()

    if plot:
        x, y = dummy.x.reshape(pixels, pixels), dummy.y.reshape(pixels, pixels)
        wij_D = dummy.wij_D.reshape(pixels, pixels, no_of_daughters)
        if shepard:
            sum_wij = wij_D.sum(axis=2) + 1e-12
            wij_D /= sum_wij[:, :, None]

        # Normalize so the colors are in the range of [0, 1]
        wij_D /= wij_D.max() 

        # Un-comment to plot values due to each SPH on the grid.
        # for i in range(no_of_daughters):
        #     plt.clf()
        #     c = plt.contourf(x, y, wij_D[:, :, i])
        #     plt.scatter(sources.x, sources.y, c='r')
        #     plt.colorbar()
        #     plt.show()

        k = np.zeros([pixels, pixels, 3])
        for i in range(no_of_daughters//3):
            for j in range(3):
                k[:, :, j] += wij_D[:, :, i*3 + j]
        k /= k.max()
        plt.imshow(k, extent=[0, length, length, 0])
        plt.scatter(sources.x, sources.y, c='w', s=40)
        plt.savefig("test.png")
        plt.show()
    return dummy


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--pixels', action='store', type=int, dest='pixels',
                   default=200,
                   help='Number of interpolation points/pixels for the image.')
    p.add_argument('-n', action='store', type=int, dest='no_of_daughters',
                   default=3, help='Number of daughter particles.')
    p.add_argument('--hdx', action='store', type=float, dest='hdx',
                   default=500, help='Particle smoothing length factor.')
    p.add_argument('--dim', action='store', type=int, dest='dim',
                   default=2, help='Dimension of the simulation.')
    p.add_argument('--seed', action='store', type=int, dest='seed',
                   default=2, help='Seed for the random no. generator.')
    p.add_argument(
        '--plot', action='store_true', dest='plot',
        default=False, help='Show plots at the end of simulation.'
    )
    p.add_argument(
        '--test', action='store_true', dest='test',
        default=False,
        help='Use the same SPH particle positions as Hopkins, else SPH \
        particle positions are randomly generated..'
    )
    p.add_argument(
        '--shepard', action='store_true', dest='shepard',
        default=False, help='Do Shepard correction for the kernel.'
    )
    o = p.parse_args()

    compute_factors(length=1.0, pixels=o.pixels, shepard=o.shepard,
                    no_of_daughters=o.no_of_daughters, hdx=o.hdx, dim=o.dim,
                    plot=o.plot, test=o.test, seed=o.seed)
