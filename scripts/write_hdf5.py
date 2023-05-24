import argparse
import pathlib
import typing

import h5py
import numpy


DEFAULTS = {
    'nx': 7,
    'ny': 7,
    'nz': 7,
    'x0': 0.5,
    'y0': 0.5,
    'z0': 0.5,
    'Kx': 0.0,
    'Ky': 0.0,
    'Kz': 0.0,
    'Sx': 0.0,
    'Sy': 0.0,
    'Sz': 0.0,
    'Mx': 0,
    'My': 0,
    'Mz': 0,
    'n0': 1.0,
    'dn': 0.0,
    'Vx': 0.0,
    'Vy': 0.0,
    'Vz': 0.0,
}


class Grid:
    """Representation of the 3D logical grid."""

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        lx: float=None,
        ly: float=None,
        lz: float=None,
    ) -> None:
        x = numpy.linspace(0.0, (lx or 1.0), nx)
        y = numpy.linspace(0.0, (ly or 1.0), ny)
        z = numpy.linspace(0.0, (lz or 1.0), nz)
        xc, yc, zc = numpy.meshgrid(x, y, z)
        self._x = xc
        self._y = yc
        self._z = zc

    def sinusoidal(
        self,
        mx: int=None,
        my: int=None,
        mz: int=None,
    ) -> numpy.ndarray:
        """Project a sinusoidal function onto this grid."""
        fx = numpy.cos((mx or 0)*numpy.pi*self.x)
        fy = numpy.cos((my or 0)*numpy.pi*self.y)
        fz = numpy.cos((mz or 0)*numpy.pi*self.z)
        return fx * fy * fz

    def gaussian(
        self,
        sx: float=None,
        sy: float=None,
        sz: float=None,
        x0: float=None,
        y0: float=None,
        z0: float=None,
    ) -> numpy.ndarray:
        """Project a Gaussian function onto this grid."""
        gx = (self.x - (x0 or 0.0)) / sx if sx else 0.0
        gy = (self.y - (y0 or 0.0)) / sy if sy else 0.0
        gz = (self.z - (z0 or 0.0)) / sz if sz else 0.0
        return numpy.exp(-0.5 * (gx**2 + gy**2 + gz**2))

    @property
    def x(self):
        """The x coordinates."""
        return self._x

    @property
    def y(self):
        """The y coordinates."""
        return self._y

    @property
    def z(self):
        """The z coordinates."""
        return self._z


def main(filepath=None, verbose: bool=False, **user):
    """Create 3-D density and flux arrays from an analytic form.
    
    All axis lengths are 1.0.
    """
    opts = DEFAULTS.copy()
    opts.update({k: v for k, v in user.items() if v})
    vlasov = compute_vlasov_quantities(opts)
    path = pathlib.Path(filepath or 'vlasov').resolve().expanduser()
    write_arrays(path, vlasov, opts, verbose=verbose)


def compute_vlasov_quantities(opts: dict):
    """Compute density and fluxes from options."""
    grid = Grid(nx=opts['nx'], ny=opts['ny'], nz=opts['nz'])
    sinusoids = grid.sinusoidal(mx=opts['Mx'], my=opts['My'], mz=opts['Mz'])
    gaussian = grid.gaussian(
        sx=opts['Sx'],
        sy=opts['Sy'],
        sz=opts['Sz'],
        x0=opts['x0'],
        y0=opts['y0'],
        z0=opts['z0'],
    )
    density = opts['n0'] + opts['dn']*sinusoids*gaussian
    xflux = opts.get('Vx', 0.0) * density
    yflux = opts.get('Vy', 0.0) * density
    zflux = opts.get('Vz', 0.0) * density
    return {
        'density': density,
        'x flux': xflux,
        'y flux': yflux,
        'z flux': zflux,
    }


def write_arrays(
    filepath: pathlib.Path,
    vlasov: typing.Dict[str, numpy.ndarray],
    opts: dict,
    verbose: bool=False,
) -> None:
    """Write the computed arrays to disk."""
    path = filepath.with_suffix('.h5')
    with h5py.File(path, 'w') as f:
        arrays = f.create_group('arrays')
        for name, array in vlasov.items():
            if verbose:
                print(f"Writing {name} to {path}")
            dset = arrays.create_dataset(name, data=array)
            for k, v in opts.items():
                dset.attrs[k] = v
            print(dset)
    if dset:
        raise IOError("Dataset was not properly closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        __file__,
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-o',
        '--output',
        dest='filepath',
        help="output path (this program will force an HDF5 suffix)",
    )
    parser.add_argument(
        '-nx',
        help="number of grid points along the x axis (default: 7)",
        type=int,
    )
    parser.add_argument(
        '-ny',
        help="number of grid points along the y axis (default: 7)",
        type=int,
    )
    parser.add_argument(
        '-nz',
        help="number of grid points along the z axis (default: 7)",
        type=int,
    )
    parser.add_argument(
        '-Sx',
        help="gaussian width along the x axis (default: infinite)",
        type=float,
    )
    parser.add_argument(
        '-Sy',
        help="gaussian width along the y axis (default: infinite)",
        type=float,
    )
    parser.add_argument(
        '-Sz',
        help="gaussian width along the z axis (default: infinite)",
        type=float,
    )
    parser.add_argument(
        '-x0',
        help="gaussian center along the x axis (default: 0.5)",
        type=float,
    )
    parser.add_argument(
        '-y0',
        help="gaussian center along the y axis (default: 0.5)",
        type=float,
    )
    parser.add_argument(
        '-z0',
        help="gaussian center along the z axis (default: 0.5)",
        type=float,
    )
    parser.add_argument(
        '-Mx',
        help="sinusoidal wave number along the x axis (default: 0)",
        type=int,
    )
    parser.add_argument(
        '-My',
        help="sinusoidal wave number along the y axis (default: 0)",
        type=int,
    )
    parser.add_argument(
        '-Mz',
        help="sinusoidal wave number along the z axis (default: 0)",
        type=int,
    )
    parser.add_argument(
        '-n0',
        help="baseline density amplitude (default: 1.0)",
        type=float,
    )
    parser.add_argument(
        '-dn',
        help="perturbed density amplitude (default: 0.0)",
        type=float,
    )
    parser.add_argument(
        '-Vx',
        help="bulk velocity [m/s] along the x axis (default: 0.0)",
        type=float,
    )
    parser.add_argument(
        '-Vy',
        help="bulk velocity [m/s] along the y axis (default: 0.0)",
        type=float,
    )
    parser.add_argument(
        '-Vz',
        help="bulk velocity [m/s] along the z axis (default: 0.0)",
        type=float,
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help="print runtime messages",
        action='store_true',
    )
    args = parser.parse_args()
    main(**vars(args))

