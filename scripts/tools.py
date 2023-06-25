import pathlib
import subprocess
import typing

import h5py
import numpy
import numpy.typing
import matplotlib.figure
import matplotlib.pyplot as plt


class HDFArray:
    """Interface to an output array in HDF5 format.
    
    Instances of this class represent a given dataset as a logically 3-D array
    in row-major order.
    """

    def __init__(
        self,
        dataset: h5py.Dataset,
        origin: typing.Sequence[typing.SupportsIndex]=None,
        order: str='C',
    ) -> None:
        self._dataset = dataset
        self._order = self._check_order(order)
        self._origin = origin
        self._array = None
        self._nx = None
        self._ny = None
        self._nz = None
        self._xy = None
        self._xz = None
        self._yz = None
        self._x0 = None
        self._y0 = None
        self._z0 = None

    def _check_order(self, order: str):
        """Make sure `order` is valid, and return standard form."""
        if order.lower() in {'c', 'row'}:
            return 'C'
        if order.lower() in {'f', 'col'}:
            return 'F'
        raise ValueError(f"Unknown array order: {order!r}") from None


    @property
    def xy(self):
        """The array in the x-y plane, at `z0`."""
        if self._xy is None:
            if self.nz:
                self._xy = self.array[:, :, self.z0].squeeze()
            else:
                self._xy = self.array.copy()
        return self._xy

    @property
    def xz(self):
        """The array in the x-z plane, at `y0`."""
        if self._xz is None:
            if self.nz > 1:
                self._xz = self.array[:, self.y0, :].squeeze()
        return self._xz

    @property
    def yz(self):
        """The array in the y-z plane, at `x0`."""
        if self._yz is None:
            if self.nz > 1:
                self._yz = self.array[self.x0, :, :].squeeze()
        return self._yz

    @property
    def x0(self):
        """The x coordinate at which 2D planes intersect."""
        if self._x0 is None:
            try:
                self._x0 = self._origin[0]
            except TypeError:
                self._x0 = self.nx // 2
        return self._x0

    @property
    def y0(self):
        """The y coordinate at which 2D planes intersect."""
        if self._y0 is None:
            try:
                self._y0 = self._origin[1]
            except TypeError:
                self._y0 = self.ny // 2
        return self._y0

    @property
    def z0(self):
        """The z coordinate at which 2D planes intersect."""
        if self._z0 is None:
            try:
                self._z0 = self._origin[2]
            except TypeError:
                self._z0 = self.nz // 2
        return self._z0

    @property
    def nx(self):
        """The number of grid points along the x axis."""
        if self._nx is None:
            self._nx = self.array.shape[0]
        return self._nx

    @property
    def ny(self):
        """The number of grid points along the y ayis."""
        if self._ny is None:
            self._ny = self.array.shape[1]
        return self._ny

    @property
    def nz(self):
        """The number of grid points along the z azis."""
        if self._nz is None:
            self._nz = self.array.shape[2]
        return self._nz

    @property
    def array(self):
        """The logically 3-D array."""
        if self._array is None:
            data = (
                numpy.transpose(self._dataset) if self._order == 'C'
                else self._dataset
            )
            self._array = numpy.array(data, ndmin=3)
        return self._array

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        shape = [self.nx, self.ny]
        if self.nz:
            shape.append(self.nz)
        return ', '.join(str(i) for i in shape)


class Results(typing.Mapping):
    """A container for output arrays."""

    def __init__(self, user: pathlib.Path, **kwargs) -> None:
        path = pathlib.Path(user).expanduser().resolve()
        self._type = None
        suffix = path.suffix
        if not suffix:
            raise ValueError(
                f"Could not determine file format from {path!r}"
            ) from None
        elif suffix.lower() in {'.hdf', '.h5'}:
            self._data = h5py.File(path, 'r')
            self._type = HDFArray
        else:
            raise NotImplementedError(
                f"{self.__class__.__qualname__} does not support "
                f"files of type {suffix}"
            ) from None
        self._path = path
        self._kwargs = kwargs

    def __len__(self) -> int:
        """Compute the number of available arrays. Called for len(self)."""
        return len(self._data)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over available arrays. Called for iter(self)."""
        return iter(self._data)

    def __getitem__(self, __key: str):
        """Access arrays by name."""
        try:
            return self._type(self.data[__key], **self._kwargs)
        except KeyError as err:
            raise KeyError(f"No data available for {__key!r}") from err

    @property
    def data(self):
        """The output dataset."""
        return self._data

    @property
    def path(self):
        """The fully resolved path to computed results."""
        return self._path


class Options(typing.Mapping):
    """A container for runtime parameter values."""

    _defaults = {
        'BC': 'periodic',
        'Kx': 0.0,
        'Ky': 0.0,
        'Kz': 0.0,
        'Sx': 0.0,
        'Sy': 0.0,
        'Sz': 0.0,
        'Mx': 0.0,
        'My': 0.0,
        'Mz': 0.0,
        'n0': 1.0,
        'dn': 0.0,
    }

    def __init__(self, filepath: typing.Union[str, pathlib.Path]) -> None:
        self._userpath = filepath
        self._path = None
        options = {}
        with self.path.open('r') as fp:
            lines = fp.readlines()
            for line in (line for line in lines if line.startswith('-')):
                parts = line.split()
                key = parts[0].lstrip('-')
                try:
                    value = parts[1]
                except IndexError:
                    value = True
                options[key] = value
        self._values = {**self._defaults, **options}

    def __len__(self) -> int:
        """The number of parameters. Called for len(self)."""
        return len(self._values)

    def __iter__(self) -> int:
        """Iterate over parameters. Called for iter(self)."""
        return iter(self._values)

    def __getitem__(self, __key: str):
        """Return the value for this parameter"""
        return self._values[__key]

    @property
    def path(self):
        """The full path to the options file."""
        if self._path is None:
            self._path = pathlib.Path(self._userpath)
        return self._path

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self) -> str:
        """A simplified representation of this object."""
        parts = [
            f"BC: {self.get('bc', 'periodic')}",
            rf"$\kappa_x = {self.get('Kx', 0.0)}$",
            rf"$\kappa_y = {self.get('Ky', 0.0)}$",
            rf"$\kappa_z = {self.get('Kz', 0.0)}$",
            rf"$\sigma_x = {self.get('Sx', 0.0)}$",
            rf"$\sigma_y = {self.get('Sy', 0.0)}$",
            rf"$\sigma_z = {self.get('Sz', 0.0)}$",
            rf"$M_x = {self.get('Kx', 0.0)}$",
            rf"$M_y = {self.get('Ky', 0.0)}$",
            rf"$M_z = {self.get('Kz', 0.0)}$",
            rf"$n_0 = {self.get('n0', 1.0)}$",
            rf"$\delta_n = {self.get('dn', 0.0)}$",
        ]
        return '\n'.join(parts)


class Video:
    """A class to handle creating videos."""

    def __init__(self, wdir: str):
        self._frame_count = 0
        self._wdir = pathlib.Path(wdir) or pathlib.Path.cwd()
        self._tmp_path = self._wdir / ".ffmpeg"
        self._tmp_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self,):
        return f"{self.__class__.__qualname__}()"

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def add_frame(self, **savefig_kw):
        """Add a frame to the video stream."""
        self._frame_count += 1
        _tmp_name = f"frame-{self._frame_count}.png"
        plt.savefig(self._tmp_path / _tmp_name, **savefig_kw)
        plt.close()

    def save(
        self,
        path: typing.Union[str, pathlib.Path]=None,
        verbose: bool=False,
    ) -> None:
        """Save the video."""
        out = pathlib.Path(
            path or self._wdir / 'new_video.mp4'
        ).expanduser().resolve()
        if verbose:
            print(f"Saving {out} ...", end=' ', flush=True)
        _format = f"0{order_of_magnitude(self.frame_count)}d"
        _regex = self._tmp_path / f"frame-%{_format}.png"
        _png_glob = self._tmp_path / f"{out.stem}-*.png"
        commands = [
            f"""cd {self._tmp_path} && rename 's/\\d+/sprintf("%{_format}",$&)/e' frame-*.png""",
            f"ffmpeg -loglevel panic -i {_regex} {self._tmp_path / out.name}",
            f"mv {self._tmp_path / out.name} {out}",
            f"rm -rf {self._tmp_path}"
        ]
        for command in commands:
            _ = subprocess.call(command, shell=True)
        if verbose:
            print("Done")


def order_of_magnitude(n):
    """Compute the order of magnitude of a number."""
    return int(numpy.ceil(numpy.log10(n)))

