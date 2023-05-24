import argparse
import pathlib

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

def main(filepath, verbose: bool=False, **user):
    """Create 3-D density and flux arrays from an analytic form.
    
    All axis lengths are 1.0.
    """

    opts = DEFAULTS.copy()
    opts.update({k: v for k, v in user.items() if v})

    x = numpy.linspace(0, 1, opts['nx'])
    y = numpy.linspace(0, 1, opts['ny'])
    z = numpy.linspace(0, 1, opts['nz'])

    xx, yy, zz = numpy.meshgrid(x, y, z)

    sinusoids = (
          numpy.cos(opts['Mx']*numpy.pi*xx)
        * numpy.cos(opts['My']*numpy.pi*yy)
        * numpy.cos(opts['Mz']*numpy.pi*zz)
    )
    gx = (xx - opts['x0']) / opts['Sx'] if opts['Sx'] else 0.0
    gy = (yy - opts['y0']) / opts['Sy'] if opts['Sy'] else 0.0
    gz = (zz - opts['z0']) / opts['Sz'] if opts['Sz'] else 0.0
    gaussian = numpy.exp(-0.5 * (gx**2 + gy**2 + gz**2))

    density = opts['n0'] + opts['dn']*sinusoids*gaussian
    xflux = opts.get('Vx', 0.0) * density
    yflux = opts.get('Vy', 0.0) * density
    zflux = opts.get('Vz', 0.0) * density

    path = pathlib.Path(filepath).resolve().expanduser().with_suffix('.h5')
    with h5py.File(path, 'w') as f:
        if verbose:
            print(f"Writing density to {path}")
        dset = f.create_dataset('density', data=density)
        f.create_dataset('x flux', data=xflux)
        f.create_dataset('y flux', data=yflux)
        f.create_dataset('z flux', data=zflux)
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
        'filepath',
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

