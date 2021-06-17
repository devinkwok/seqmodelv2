import os
import abc
import typing
from pathlib import Path


class OsInterface(abc.ABC):

    sep = os.sep

    def split(self, path: os.PathLike) -> list:
        return os.path.split(path)

    def join(self, *paths) -> os.PathLike:
        return os.path.join(*paths)

    @abc.abstractmethod
    def mkdir(self, path: os.PathLike):
        pass

    @abc.abstractmethod
    def find(self,
        path: os.PathLike,
        suffix: str = None,
    ) -> typing.List[typing.Tuple[os.PathLike, str]]:
        return None

    @abc.abstractmethod
    def list(self, path: os.PathLike, suffix: str = None) -> typing.List[os.PathLike]:
        return None

    @abc.abstractmethod
    def type_of(self, path: os.PathLike) -> str:
        return None

    @abc.abstractmethod
    def read(self, file_path: os.PathLike):
        pass

    @abc.abstractmethod
    def write(self, contents: str, file_path: os.PathLike):
        pass

    @abc.abstractmethod
    def command(self, linux_command: str) -> str:
        return None


class LocalInterface(OsInterface):

    def mkdir(self, path: os.PathLike):
        Path(path).mkdir(parents=True, exist_ok=True)

    def find(self,
        path: os.PathLike,
        suffix: str = None,
    ) -> typing.List[typing.Tuple[os.PathLike, str]]:
        paths = []
        for root, dirs, files in os.walk(path):
            for d in dirs:
                if suffix is None or (d + '/').endswith(suffix):
                    paths.append((root, d + '/'))
            for f in files:
                if suffix is None or f.endswith(suffix):
                    paths.append((root, f))
        return paths

    def list(self, path: os.PathLike, suffix: str = None) -> typing.List[os.PathLike]:
        paths = []
        if not os.path.isdir(path):
            return paths
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                f = f + '/'
            if suffix is None or f.endswith(suffix):
                paths.append(f)
        return paths

    def type_of(self, path: os.PathLike) -> str:
        if os.path.isfile(path):
            return 'file'
        if os.path.isdir(path):
            return 'dir'
        if os.path.exists(path):
            return 'other'
        return 'none'

    def read(self, file_path: os.PathLike):
        with open(file_path, 'r') as file:
            for line in file:
                yield line

    def write(self, contents: str, file_path: os.PathLike):
        with open(file_path, 'w') as f:
            f.write(contents)

    def command(self, linux_command: str) -> str:
        return os.system(linux_command)


class SshInterface(OsInterface):

    pass #TODO
