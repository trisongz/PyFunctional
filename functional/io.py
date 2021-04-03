import os
import gzip
import lzma
import bz2
import io
import builtins
import simdjson as json
import pickle
import csv
from functional.base import lazy_import

gf = lazy_import('tensorflow.io.gfile')
gfile = gf.GFile
glob = gf.glob
gcopy = gf.copy
isdir = gf.isdir
listdir = gf.listdir
mkdirs = gf.makedirs
mv = gf.rename
exists = gf.exists
rmdir = gf.rmtree
rm = gf.remove
jparser = json.Parser()

tf = lazy_import('tensorflow')
TextLineDataset = tf.data.TextLineDataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
enable_eager_execution = tf.compat.v1.enable_eager_execution
disable_v2_behavior = tf.compat.v1.disable_v2_behavior


WRITE_MODE = "wt"


class ReusableFile(object):
    """
    Class which emulates the builtin file except that calling iter() on it will return separate
    iterators on different file handlers (which are automatically closed when iteration stops). This
    is useful for allowing a file object to be iterated over multiple times while keep evaluation
    lazy.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path,
        delimiter=None,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
    ):
        """
        Constructor arguments are passed directly to builtins.open
        :param path: passed to open
        :param delimiter: passed to open
        :param mode: passed to open
        :param buffering: passed to open
        :param encoding: passed to open
        :param errors: passed to open
        :param newline: passed to open
        :return: ReusableFile from the arguments
        """
        self.path = path
        self.delimiter = delimiter
        self.mode = mode
        self.buffering = buffering
        self.encoding = encoding
        self.errors = errors
        self.newline = newline

    def __iter__(self):
        """
        Returns a new iterator over the file using the arguments from the constructor. Each call
        to __iter__ returns a new iterator independent of all others
        :return: iterator over file
        """
        # pylint: disable=no-member
        with builtins.open(
            self.path,
            mode=self.mode,
            buffering=self.buffering,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            for line in file_content:
                yield line

    def read(self):
        # pylint: disable=no-member
        with builtins.open(
            self.path,
            mode=self.mode,
            buffering=self.buffering,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            return file_content.read()


class CloudFile(ReusableFile):
    def __init__(self, path, mode="r"):
        super(CloudFile, self).__init__(path, delimiter=None, mode=mode, buffering=None, encoding=None, errors=None, newline=None)

    def __iter__(self):
        with gfile(self.path, mode=self.mode) as file_content:
            for line in file_content:
                yield line
    
    def read(self):
        with gfile(self.path, mode=self.mode) as file_content:
            return file_content.read()


class CompressedFile(ReusableFile):
    magic_bytes = None

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path,
        delimiter=None,
        mode="rt",
        buffering=-1,
        compresslevel=9,
        encoding=None,
        errors=None,
        newline=None,
    ):
        super(CompressedFile, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
        self.compresslevel = compresslevel

    @classmethod
    def is_compressed(cls, data):
        return data.startswith(cls.magic_bytes)


class GZFile(CompressedFile):
    magic_bytes = b"\x1f\x8b\x08"

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path,
        delimiter=None,
        mode="rt",
        buffering=-1,
        compresslevel=9,
        encoding=None,
        errors=None,
        newline=None,
    ):
        super(GZFile, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    def __iter__(self):
        if "t" in self.mode:
            with gzip.GzipFile(self.path, compresslevel=self.compresslevel) as gz_file:
                gz_file.read1 = gz_file.read
                with io.TextIOWrapper(
                    gz_file,
                    encoding=self.encoding,
                    errors=self.errors,
                    newline=self.newline,
                ) as file_content:
                    for line in file_content:
                        yield line
        else:
            with gzip.open(
                self.path, mode=self.mode, compresslevel=self.compresslevel
            ) as file_content:
                for line in file_content:
                    yield line

    def read(self):
        with gzip.GzipFile(self.path, compresslevel=self.compresslevel) as gz_file:
            gz_file.read1 = gz_file.read
            with io.TextIOWrapper(
                gz_file,
                encoding=self.encoding,
                errors=self.errors,
                newline=self.newline,
            ) as file_content:
                return file_content.read()


class BZ2File(CompressedFile):
    magic_bytes = b"\x42\x5a\x68"

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path,
        delimiter=None,
        mode="rt",
        buffering=-1,
        compresslevel=9,
        encoding=None,
        errors=None,
        newline=None,
    ):
        super(BZ2File, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    def __iter__(self):
        with bz2.open(
            self.path,
            mode=self.mode,
            compresslevel=self.compresslevel,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            for line in file_content:
                yield line

    def read(self):
        with bz2.open(
            self.path,
            mode=self.mode,
            compresslevel=self.compresslevel,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            return file_content.read()


class XZFile(CompressedFile):
    magic_bytes = b"\xfd\x37\x7a\x58\x5a\x00"

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path,
        delimiter=None,
        mode="rt",
        buffering=-1,
        compresslevel=9,
        encoding=None,
        errors=None,
        newline=None,
        check=-1,
        preset=None,
        filters=None,
        format=None,
    ):
        super(XZFile, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
        self.check = check
        self.preset = preset
        self.format = format
        self.filters = filters

    def __iter__(self):
        with lzma.open(
            self.path,
            mode=self.mode,
            format=self.format,
            check=self.check,
            preset=self.preset,
            filters=self.filters,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            for line in file_content:
                yield line

    def read(self):
        with lzma.open(
            self.path,
            mode=self.mode,
            format=self.format,
            check=self.check,
            preset=self.preset,
            filters=self.filters,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            return file_content.read()


COMPRESSION_CLASSES = [GZFile, BZ2File, XZFile]
N_COMPRESSION_CHECK_BYTES = max(len(cls.magic_bytes) for cls in COMPRESSION_CLASSES)


def get_read_function(filename, disable_compression):
    if filename.startswith('gs://'):
        return CloudFile
    if disable_compression:
        return ReusableFile
    else:
        with open(filename, "rb") as f:
            start_bytes = f.read(N_COMPRESSION_CHECK_BYTES)
            for cls in COMPRESSION_CLASSES:
                if cls.is_compressed(start_bytes):
                    return cls
            return ReusableFile


def universal_write_open(
    path,
    mode,
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    compresslevel=9,
    format=None,
    check=-1,
    preset=None,
    filters=None,
    compression=None,
):
    if path.startswith('gs://'):
        return gfile(path, mode)
    if compression is None:
        return builtins.open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    elif compression in ("gz", "gzip"):
        return gzip.open(
            path,
            mode=mode,
            compresslevel=compresslevel,
            errors=errors,
            newline=newline,
            encoding=encoding,
        )
    elif compression in ("lzma", "xz"):
        return lzma.open(
            path,
            mode=mode,
            format=format,
            check=check,
            preset=preset,
            filters=filters,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    elif compression == "bz2":
        return bz2.open(
            path,
            mode=mode,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    else:
        raise ValueError(
            "compression must be None, gz, gzip, lzma, or xz and was {0}".format(
                compression
            )
        )

class File(object):
    @classmethod
    def join(cls, path, *paths):
        return os.path.join(path, *paths)

    @classmethod
    def listfiles(cls, filepath):
        return listdir(filepath)

    @classmethod
    def isfile(cls, filepath):
        return isfile(filepath)

    @classmethod
    def listdir(cls, filepath):
        return listdir(filepath)

    @classmethod
    def mkdir(cls, filepath):
        return mkdirs(filepath)

    @classmethod
    def makedirs(cls, filepath):
        return mkdirs(filepath)

    @classmethod
    def mkdirs(cls, filepath):
        return mkdirs(filepath)

    @classmethod
    def glob(cls, filepath):
        return glob(filepath)

    @classmethod
    def mv(cls, src, dest, overwrite=False):
        return mv(src, dest, overwrite)

    @classmethod
    def rm(cls, filename):
        return rm(filename)
    
    @classmethod
    def rmdir(cls, filepath):
        return rmdir(filepath)

    @classmethod
    def copy(cls, src, dest, overwrite=True):
        return gcopy(src, dest, overwrite)

    @classmethod
    def exists(cls, filepath):
        return exists(filepath)
    
    @classmethod
    def base(cls, filepath):
        return os.path.basename(filepath)

    @classmethod
    def ext(cls, filepath):
        f = os.path.basename(filepath)
        _, e = os.path.splitext(f)
        return e

    @classmethod
    def writemode(cls, filepath, overwrite=False):
        if exists(filepath):
            return 'a'
        return 'w'
    
    @classmethod
    def touch(cls, filepath, overwrite=False):
        if not exists or overwrite:
            with gfile(filepath, 'w') as f:
                f.write('\n')
                f.flush()
            f.close()

    @classmethod
    def open(cls, filename, mode='r', auto=False):
        if 'r' in mode and auto:
            if filename.endswith('.pkl'):
                return File.pload(filename)
            
            if filename.endswith('.jsonl') or filename.endswith('.jsonlines'):
                return File.jg(filename)
            
            if filename.endswith('.json'):
                return File.jsonload(filename)
        return gfile(filename, mode)
    
    @classmethod
    def write(cls, filename, mode='w'):
        return gfile(filename, mode)
    
    @classmethod
    def append(cls, filename, mode='a'):
        return gfile(filename, mode)
    
    @classmethod
    def read(cls, filename, mode='r'):
        return gfile(filename, mode)

    @classmethod
    def rb(cls, filename):
        return gfile(filename, 'rb')
    
    @classmethod
    def wb(cls, filename):
        return gfile(filename, 'wb')

    @classmethod
    def readlines(cls, filename):
        with gfile(filename, 'r') as f:
            return f.readlines()

    @classmethod
    def pklsave(cls, obj, filename):
        return pickle.dump(obj, gfile(filename, 'wb'))

    @classmethod
    def pload(cls, filename):
        return pickle.load(gfile(filename, 'rb'))
    
    @classmethod
    def pklload(cls, filename):
        return pickle.load(gfile(filename, 'rb'))

    @classmethod
    def csvload(cls, filename):
        return list(csv.reader(gfile(filename, 'r')))

    @classmethod
    def tsvload(cls, filename):
        return list(csv.reader(gfile(filename, 'r'), delimiter='\t'))
    
    @classmethod
    def csvdictload(cls, filename):
        return dict(csv.DictReader(gfile(filename, 'r')))

    @classmethod
    def tsvdictload(cls, filename):
        return dict(csv.DictReader(gfile(filename, 'r'), delimiter='\t'))

    @classmethod
    def jsonload(cls, filename):
        return json.load(gfile(filename, 'r'))
    
    @classmethod
    def jsonloads(cls, string):
        return json.loads(string)
    
    @classmethod
    def jsondump(cls, obj, filename, indent=2, ensure_ascii=False):
        return json.dump(obj, gfile(filename, 'w'), indent=indent, ensure_ascii=ensure_ascii)
    
    @classmethod
    def jsondumps(cls, pdict, ensure_ascii=False):
        return json.dumps(pdict, ensure_ascii=ensure_ascii)
    
    @classmethod
    def jp(cls, line):
        return jparser.parse(line).as_dict()

    @classmethod
    def jl(cls, line):
        return json.loads(line)
    
    @classmethod
    def jlp(cls, line):
        try:
            return File.jp(line)
        except:
            return File.jl(line)

    @classmethod
    def jldump(cls, data, f):
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
    
    @classmethod
    def twrite(cls, data, f):
        f.write(data + '\n')

    @classmethod
    def flush(cls, f):
        f.flush()

    @classmethod
    def fclose(cls, f):
        f.flush()
        f.close()

    @classmethod
    def jg(cls, filename, handle_errors=True):
        with gfile(filename, 'r') as f:
            for l in f:
                try:
                    yield json.loads(l)
                except Exception as e:
                    if not handle_errors:
                        logger.log(f'Error parsing File: {str(e)}')
                        raise e
    
    @classmethod
    def autowrite(cls, filename, overwrite=False):
        if overwrite or not exists(filename):
            return 'w'
        return 'a'

    @classmethod
    def jlwrite(cls, data, filename, mode='auto'):
        mode = mode if mode != 'auto' else File.autowrite(filename)
        with gfile(filename, mode=mode) as f:
            File.jldump(data, f)
        File.fclose(f)

    @classmethod
    def jlwrites(cls, data_items, filename, mode='auto'):
        mode = mode if mode != 'auto' else File.autowrite(filename)
        with gfile(filename, mode=mode) as f:
            for data in data_items:
                File.jldump(data, f)
        File.fclose(f)

    @classmethod
    def jlload(cls, filename, as_iter=False, index=False, handle_errors=True):
        if as_iter:
            if not index:
                return File.jg(filename, handle_errors=handle_errors)
            yield from enumerate(File.jg(filename, handle_errors=handle_errors))

        else:
            if index:
                return {x: item for x, item in enumerate(File.jg(filename, handle_errors=handle_errors))}
            return [item for item in File.jg(filename, handle_errors=handle_errors)]

    @classmethod
    def jlw(cls, data, filename, mode='auto', verbose=True):
        mode = mode if mode != 'auto' else File.autowrite(filename)
        if isinstance(data, dict):
            return File.jlwrite(data, filename, mode=mode)
        _good, _bad, failed = 0, 0, []
        with gfile(filename, mode) as f:
            for d in data:
                try:
                    File.jldump(d, f)
                    _good += 1
                except Exception as e:
                    if verbose:
                        print(f'Error: {str(e)} Writing Line {d}')
                    failed.append({'data': d, 'error': str(e)})
                    _bad += 1
            File.fclose(f)
        _total = _good + _bad
        print(f'Wrote {_good}/{_total} Lines [Mode: {mode}] - Failed: {_bad}')
        return failed

    @classmethod
    def fsorter(cls, filenames):
        fnames = []
        if isinstance(filenames, str) or not isinstance(filenames, list):
            filenames = [filenames]
        for fn in filenames:
            if not isinstance(fn, str):
                fn = str(fn)
            if fn.endswith('*'):
                _newfns = glob(fn)
                _newfns = [f for f in _newfns if isfile(f) and exists(f)]
                fnames.extend(_newfns)
            elif not isdir(fn) and exists(fn):
                fnames.append(fn)
        return fnames

    @classmethod
    def jgs(cls, filenames, handle_errors=True):
        filenames = File.fsorter(filenames)
        for fname in filenames:
            yield File.jg(fname, handle_errors)

    def __call__(self, filename, mode='r'):
        return self.open(filename, mode)

    @classmethod
    def gfile(cls, filename, mode):
        return gfile(filename, mode)
    
    @classmethod
    def gfiles(cls, filenames, mode='r'):
        fnames = File.fsorter(filenames)
        for fn in fnames:
            yield gfile(fn, mode)
    
    @classmethod
    def tfeager(cls, enable=True):
        if enable:
            enable_eager_execution()
        else:
            disable_v2_behavior()
    
    @classmethod
    def tflines(cls, filenames):
        File.tfeager()
        fnames = File.fsorter(filenames)
        return TextLineDataset(fnames, num_parallel_reads=AUTOTUNE)
    
    @classmethod
    def csvreader(cls, f):
        return csv.DictReader(f)
    
    @classmethod
    def tsvreader(cls, f):
        return csv.DictReader(f, delimiter='\t')

    @classmethod
    def tfjl(cls, filenames, handle_errors=True, verbose=False):
        pipeline = File.tflines(filenames)
        for idx, x in enumerate(pipeline.as_numpy_iterator()):
            if not handle_errors:
                yield File.jlp(x)
            else:
                try:
                    yield File.jlp(x)
                except Exception as e:
                    if verbose:
                        print(f'Error on {idx}: {str(e)} - {x}')

    @classmethod
    def tfcsv(cls, filenames):
        for f in File.gfiles(filenames):
            reader = File.csvreader(f)
            yield from reader

    @classmethod
    def tftl(cls, filenames, handle_errors=True, verbose=True):
        pipeline = File.tflines(filenames)
        for idx, x in enumerate(pipeline.as_numpy_iterator()):
            if not handle_errors:
                yield x.strip()
            else:
                try:
                    yield x.strip()
                except Exception as e:
                    if verbose:
                        print(f'Error on {idx}: {str(e)} - {x}')

    @property
    def root(self):
        return os.path.abspath(os.path.dirname(__file__))
    
    @classmethod
    def get_root(cls, path=None):
        if not path:
            return File.root
        return os.path.abspath(os.path.dirname(path))