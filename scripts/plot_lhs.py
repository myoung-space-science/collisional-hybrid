import argparse
import pathlib
import sys
import typing

import numpy
import matplotlib as mpl
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.lines as lns

PETSC_PATH = pathlib.Path('~/petsc/lib/petsc/bin').expanduser()
sys.path.append(str(PETSC_PATH))

import PetscBinaryIO
import tools


class Mat:
    """A custom representation of a PETSc Mat object."""

    def __init__(self, filepath: pathlib.Path) -> None:
        io = PetscBinaryIO.PetscBinaryIO()
        fh = open(filepath)
        io.readObjectType(fh)
        self._data = io.readMatDense(fh)
        self._array = None
        self._normalized = None
        self._nr = None
        self._nc = None

    def mask(self, value: float, normalized: bool=False):
        """Mask the data array at `value`."""
        array = self.normalized if normalized else self.array
        return numpy.ma.masked_values(array, value)

    @property
    def nr(self):
        """The number of rows in the matrix."""
        if self._nr is None:
            self._nr = self.array.shape[0]
        return self._nr

    @property
    def nc(self):
        """The number of columns in the matrix."""
        if self._nc is None:
            self._nc = self.array.shape[1]
        return self._nc

    @property
    def array(self):
        """The corresponding numpy array."""
        if self._array is None:
            self._array = numpy.array(self._data)
        return self._array

    @property
    def normalized(self):
        """Alias for `self.normalize(vcenter=0.0)`."""
        if self._normalized is None:
            self._normalized = self.normalize(vcenter=0.0)
        return self._normalized

    @typing.overload
    def normalize(
        self,
        vmin: float=None,
        vcenter: float=None,
        vmax: float=None,
    ) -> numpy.ndarray: ...

    @typing.overload
    def normalize(
        self,
        vmin: float=None,
        vmax: float=None,
        clip: bool=False,
    ) -> numpy.ndarray: ...

    def normalize(self, **kwargs) -> numpy.ndarray:
        """Normalized the data array."""
        normalize = (
            clr.TwoSlopeNorm(**kwargs) if 'vcenter' in kwargs
            else clr.Normalize(*kwargs)
        )
        return normalize(self._array)


def create_legend(keys, values, cmap: clr.Colormap, exclude: float=None):
    """Create a legend object from `keys` and `values`."""
    legend = {
        x: y for x, y in zip(keys, values)
        if x != exclude
    } if exclude is not None else dict(zip(keys, values))
    colors = []
    labels = []
    for x, y in legend.items():
        label = numpy.format_float_positional(
            x, sign=True,
            trim='0',
            precision=3,
            unique=True,
        )
        labels.append(fr'${label}$')
        colors.append(
            lns.Line2D([0], [0], color=cmap(y), linestyle='', marker='.')
        )
    return plt.legend(
        colors,
        labels,
        loc='upper left',
        bbox_to_anchor=(1.0, 1.0),
        edgecolor='k',
    )


def set_axes(ax, nx, ny, nz):
    """Perform various manipulations to plot axes."""
    xmaj = ymaj = ny*nz
    xmin = ymin = nx
    ax.xaxis.set_major_locator(tkr.MultipleLocator(xmaj))
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(xmin))
    ax.yaxis.set_major_locator(tkr.MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(tkr.MultipleLocator(ymin))
    ax.set_aspect('equal')
    ax.invert_yaxis()


def add_text(ax, options: tools.Options):
    """Add annotations for runtime options."""
    kappa = {f'K{q}': f'\kappa_{q}' for q in ('x', 'y', 'z')}
    textstr = '\n'.join(
        fr"${s}={options.get(k, 0.0)}$"
        for k, s in kappa.items()
    )
    ax.text(
        0.95, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        horizontalalignment='right',
        verticalalignment='top',
        bbox={'boxstyle': 'round', 'facecolor': 'w'},
    )


def main(
    inpath: str=None,
    outdir: str=None,
    figname: str=None,
    colormap: str=None,
    spacing: str='even',
    verbose: bool=False,
) -> None:
    """Plot the dense structure of the LHS operator matrix."""

    filepath = pathlib.Path(inpath or 'LHS.dat').resolve().expanduser()
    options = tools.Options(filepath.parent / 'options.txt')
    outpath = pathlib.Path(outdir or filepath.parent).expanduser().resolve()

    mat = Mat(filepath)

    nx = ny = nz = int(numpy.cbrt(mat.nr))

    array = mat.array
    cmap = mpl.colormaps[colormap or 'jet']

    index = int(nx*ny*(nz//2 - 1) + nx*(ny//2 - 1) + nx//2)
    ni = len(INDICES)
    points = numpy.zeros(ni, dtype=int)
    # (i  , j-1, k-1)
    points[ni//2 - 9] = index - nx*ny - nx
    # (i-1, j  , k-1)
    points[ni//2 - 8] = index - nx*ny - 1
    # (i  , j  , k-1)
    points[ni//2 - 7] = index - nx*ny
    # (i+1, j  , k-1)
    points[ni//2 - 6] = index - nx*ny + 1
    # (i  , j+1, k-1)
    points[ni//2 - 5] = index - nx*ny + nx
    # (i-1, j-1, k  )
    points[ni//2 - 4] = index - nx - 1
    # (i  , j-1, k  )
    points[ni//2 - 3] = index - nx
    # (i+1, j-1, k  )
    points[ni//2 - 2] = index - nx + 1
    # (i-1, j  , k  )
    points[ni//2 - 1] = index - 1
    # (i  , j  , k  )
    points[ni//2] = index
    # (i+1, j  , k  )
    points[ni//2 + 1] = index + 1
    # (i-1, j+1, k  )
    points[ni//2 + 2] = index + nx - 1
    # (i  , j+1, k  )
    points[ni//2 + 3] = index + nx
    # (i+1, j+1, k  )
    points[ni//2 + 4] = index + nx + 1
    # (i  , j-1, k+1)
    points[ni//2 + 5] = index + nx*ny - nx
    # (i-1, j  , k+1)
    points[ni//2 + 6] = index + nx*ny - 1
    # (i  , j  , k+1)
    points[ni//2 + 7] = index + nx*ny
    # (i+1, j  , k+1)
    points[ni//2 + 8] = index + nx*ny + 1
    # (i  , j+1, k+1)
    points[ni//2 + 9] = index + nx*ny + nx
    values = array[index, points].round(decimals=6)
    values[numpy.abs(values) < sys.float_info.min] = 0.0
    stencil = dict(zip(INDICES, values))
    if verbose:
        print(f"Matrix stencil:")
        for p, v in stencil.items():
            print(f"{p}: {v:+}")

    colors = []
    for v in values:
        if v < 0:
            colors.append('blue')
        elif v > 0:
            colors.append('red')
        else:
            colors.append('white')

    plt.figure(dpi=500.0, figsize=(12, 3))
    sizes = 1e4 * (numpy.abs(values) / numpy.max(numpy.abs(values)))
    xvals = points if spacing == 'column' else range(ni)
    yvals = numpy.full_like(xvals, index)
    plt.scatter(
        xvals,
        yvals,
        c='black',
        marker='.',
    )
    plt.scatter(
        xvals,
        yvals,
        s=sizes,
        c=colors,
        marker='|',
    )
    idxlabels = [fr'${i}$' for i in INDICES]
    plt.xticks(xvals, idxlabels, rotation='vertical', fontsize='xx-small')
    plt.yticks([])
    figpath = (outpath / 'stencil').with_suffix('.png')
    if verbose:
        print(f"Saving {figpath}")
    plt.subplots_adjust(top=0.85, bottom=0.25)
    plt.savefig(figpath)
    plt.close()

    plt.figure(dpi=500.0)
    plt.pcolormesh(mat.mask(0.5, normalized=True), cmap=cmap)
    ax = plt.gca()

    set_axes(ax, nx, ny, nz)

    plt.title(fr"Structure of ${nx} \times {ny} \times {nz}$ LHS Matrix")
    plt.xlabel("Row")
    plt.ylabel("Column")
    plt.grid(which='minor', axis='both', color='gray')
    plt.grid(which='major', axis='both', color='black')

    figpath = (outpath / (figname or 'lhs')).with_suffix('.png')
    if verbose:
        print(f"Saving {figpath}")
    plt.savefig(figpath)
    plt.close()


INDICES = [
    '(i  , j-1, k-1)',
    '(i-1, j  , k-1)',
    '(i  , j  , k-1)',
    '(i+1, j  , k-1)',
    '(i  , j+1, k-1)',
    '(i-1, j-1, k  )',
    '(i  , j-1, k  )',
    '(i+1, j-1, k  )',
    '(i-1, j  , k  )',
    '(i  , j  , k  )',
    '(i+1, j  , k  )',
    '(i-1, j+1, k  )',
    '(i  , j+1, k  )',
    '(i+1, j+1, k  )',
    '(i  , j-1, k+1)',
    '(i-1, j  , k+1)',
    '(i  , j  , k+1)',
    '(i+1, j  , k+1)',
    '(i  , j+1, k+1)',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        __file__,
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-i',
        '--input',
        dest='inpath',
        help="The path to the binary matrix file.",
    )
    parser.add_argument(
        '-o',
        '--outdir',
        help="The directory to which to save the figure.",
    )
    parser.add_argument(
        '-f',
        '--figname',
        help="The name of the figure to save.",
    )
    parser.add_argument(
        '-c',
        '--colormap',
        help="The matplotlib colormap to use.",
    )
    parser.add_argument(
        '--spacing',
        help="space stencil points by column or evenly (default)",
        choices=['even', 'column'],
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help="Print runtime messages.",
        action='store_true',
    )
    args = parser.parse_args()
    main(**vars(args))

