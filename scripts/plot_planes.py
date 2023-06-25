import argparse
import pathlib
import typing

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy

import tools


def add_plot(ax, data, xstr: str, ystr: str, show_min_max: bool=False):
    """Add a 2-D figure from `data` to `ax`.
    
    Notes
    -----
    This function inverts the logical y axis in order to set the origin in the
    bottom left corner.
    """
    ax.pcolormesh(numpy.transpose(data))
    ax.set_xlabel(xstr)
    ax.set_ylabel(ystr)
    if show_min_max:
        dmax = numpy.max(data)
        dmin = numpy.min(data)
        ax.set_title(
            f"max = {dmax:g} "
            f"min = {dmin:g}\n"
            f"(max-min) = {dmax-dmin:g} ",
            {'fontsize': 6},
        )
    ax.label_outer()
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(tck.MaxNLocator(5))
    ax.yaxis.set_major_locator(tck.MaxNLocator(5))


def create(array: tools.HDFArray, options: tools.Options=None, **kwargs):
    """Plot the planes in the given dataset."""
    planes = {
        (0, 0): {'data': array.xy, 'xstr': 'x', 'ystr': 'y'},
        (1, 0): {'data': array.xz, 'xstr': 'x', 'ystr': 'z'},
        (1, 1): {'data': array.yz, 'xstr': 'y', 'ystr': 'z'},
    }
    axes = {k: v.copy() for k, v in planes.items() if v['data'] is not None}
    nrows = 1 + (len(axes) == 3)
    ncols = 2
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex='col',
        sharey='row',
        squeeze=False,
        figsize=(6, 6),
    )
    axs[0, 1].remove()
    if options:
        plt.figtext(
            0.55, 0.95,
            str(options),
            horizontalalignment='left',
            verticalalignment='top',
        )
    for i, this in axes.items():
        ax = axs[i[0], i[1]]
        add_plot(ax, **this, **kwargs)
    fig.tight_layout()


def main(
    filename: str,
    vectors: typing.Iterable[str],
    source: str=None,
    show_min_max: bool=False,
    origin: typing.Sequence[typing.SupportsInt]=None,
    outdir: str=None,
    optsfile: str=None,
    verbose: bool=False,
) -> None:
    """Plot 2-D planes of the named 3-D array(s)."""
    srcdir = pathlib.Path(source or '.').expanduser().resolve()
    results = tools.Results(srcdir / filename, origin=origin)
    try:
        options = tools.Options(optsfile)
    except (TypeError, FileNotFoundError):
        options = None
    figdir = pathlib.Path(outdir).expanduser().resolve() if outdir else srcdir
    if verbose:
        print(f"Figure directory: {figdir}")
    names = vectors or iter(results)
    for name in names:
        if verbose:
            print(f"Plotting {name}")
        create(results[name], options=options, show_min_max=show_min_max)
        figname = f"{name.replace(' ', '_')}.png"
        if verbose:
            print(f"Saving {figname}")
        plt.savefig(figdir / figname)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        __file__,
        description=main.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'filename',
        help="the name of the file containing output vectors",
    )
    parser.add_argument(
        'vectors',
        help=(
            "the name(s) of the vector(s) to plot"
            "\n(default: plot all vectors in the dataset)"
        ),
        nargs='*',
    )
    parser.add_argument(
        '-i',
        '--indir',
        dest="source",
        help=(
            "the directory containing the data to plot"
            "\n(default: current working directory)"
        ),
    )
    parser.add_argument(
        '--min-max',
        dest='show_min_max',
        help="print information about min and max values",
        action='store_true',
    )
    parser.add_argument(
        '--origin',
        help=(
            "the coordinates at which 2D planes intersect"
            "\n(default: midpoint)"
        ),
        type=int,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
    )
    parser.add_argument(
        '-o',
        '--outdir',
        help=(
            "the directory to which to save the figure"
            "\n(default: SOURCE)"
        ),
    )
    parser.add_argument(
        '--options',
        dest='optsfile',
        help="path to a file from which to create a legend of options values",
        metavar="PATH",
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help="print runtime messages",
        action='store_true',
    )
    args = parser.parse_args()
    main(**vars(args))

