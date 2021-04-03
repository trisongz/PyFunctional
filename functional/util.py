import time
import collections
import math
import dill as serializer
import dateparser
from functools import reduce
from itertools import chain, count, islice, takewhile
from multiprocessing import Pool, cpu_count
from datetime import datetime, timezone, timedelta

from functional.base import lazy_import

PROTOCOL = serializer.HIGHEST_PROTOCOL
CPU_COUNT = cpu_count()


def is_primitive(val):
    """
    Checks if the passed value is a primitive type.

    >>> is_primitive(1)
    True

    >>> is_primitive("abc")
    True

    >>> is_primitive(True)
    True

    >>> is_primitive({})
    False

    >>> is_primitive([])
    False

    >>> is_primitive(set([]))

    :param val: value to check
    :return: True if value is a primitive, else False
    """
    return isinstance(val, (str, bool, float, complex, bytes, int))


def is_namedtuple(val):
    """
    Use Duck Typing to check if val is a named tuple. Checks that val is of type tuple and contains
    the attribute _fields which is defined for named tuples.
    :param val: value to check type of
    :return: True if val is a namedtuple
    """
    val_type = type(val)
    bases = val_type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(val_type, "_fields", None)
    return all(isinstance(n, str) for n in fields)


def identity(arg):
    """
    Function which returns the argument. Used as a default lambda function.

    >>> obj = object()
    >>> obj is identity(obj)
    True

    :param arg: object to take identity of
    :return: return arg
    """
    return arg


def is_iterable(val):
    """
    Check if val is not a list, but is a collections.Iterable type. This is used to determine
    when list() should be called on val

    >>> l = [1, 2]
    >>> is_iterable(l)
    False
    >>> is_iterable(iter(l))
    True

    :param val: value to check
    :return: True if it is not a list, but is a collections.Iterable
    """
    if isinstance(val, list):
        return False
    return isinstance(val, collections.abc.Iterable)


def is_tabulatable(val):
    if is_primitive(val):
        return False
    if is_iterable(val) or is_namedtuple(val) or isinstance(val, list):
        return True
    return False


def split_every(parts, iterable):
    """
    Split an iterable into parts of length parts

    >>> l = iter([1, 2, 3, 4])
    >>> split_every(2, l)
    [[1, 2], [3, 4]]

    :param iterable: iterable to split
    :param parts: number of chunks
    :return: return the iterable split in parts
    """
    return takewhile(bool, (list(islice(iterable, parts)) for _ in count()))


def unpack(packed):
    """
    Unpack the function and args then apply the function to the arguments and return result
    :param packed: input packed tuple of (func, args)
    :return: result of applying packed function on packed args
    """
    func, args = serializer.loads(packed)
    result = func(*args)
    if isinstance(result, collections.abc.Iterable):
        return list(result)
    return None


def pack(func, args):
    """
    Pack a function and the args it should be applied to
    :param func: Function to apply
    :param args: Args to evaluate with
    :return: Packed (func, args) tuple
    """
    return serializer.dumps((func, args), PROTOCOL)


def parallelize(func, result, processes=None, partition_size=None):
    """
    Creates an iterable which is lazily computed in parallel from applying func on result
    :param func: Function to apply
    :param result: Data to apply to
    :param processes: Number of processes to use in parallel
    :param partition_size: Size of partitions for each parallel process
    :return: Iterable of applying func on result
    """
    parallel_iter = lazy_parallelize(
        func, result, processes=processes, partition_size=partition_size
    )
    return chain.from_iterable(parallel_iter)


def lazy_parallelize(func, result, processes=None, partition_size=None):
    """
    Lazily computes an iterable in parallel, and returns them in pool chunks
    :param func: Function to apply
    :param result: Data to apply to
    :param processes: Number of processes to use in parallel
    :param partition_size: Size of partitions for each parallel process
    :return: Iterable of chunks where each chunk as func applied to it
    """
    if processes is None or processes < 1:
        processes = CPU_COUNT
    else:
        processes = min(processes, CPU_COUNT)
    partition_size = partition_size or compute_partition_size(result, processes)
    pool = Pool(processes=processes)
    partitions = split_every(partition_size, iter(result))
    packed_partitions = (pack(func, (partition,)) for partition in partitions)
    for pool_result in pool.imap(unpack, packed_partitions):
        yield pool_result
    pool.terminate()


def compute_partition_size(result, processes):
    """
    Attempts to compute the partition size to evenly distribute work across processes. Defaults to
    1 if the length of result cannot be determined.

    :param result: Result to compute on
    :param processes: Number of processes to use
    :return: Best partition size
    """
    try:
        return max(math.ceil(len(result) / processes), 1)
    except TypeError:
        return 1


def compose(*functions):
    """
    Compose all the function arguments together
    :param functions: Functions to compose
    :return: Single composed function
    """
    # pylint: disable=undefined-variable
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)



class Time(object):
    @classmethod
    def timer(cls, start_time=None):
        if not start_time:
            return time.time()
        return time.time() - start_time

    @classmethod
    def now(cls, as_iso=False, as_string=True):
        if as_string:
            if as_iso:
                return Time.iso_timestamp()
            return Time.timestamp()
        return datetime.now(timezone.utc)
    
    @classmethod
    def iso_timestamp(cls):
        return datetime.now(timezone.utc).isoformat('T')
    
    def timestamp(cls):
        return datetime.now(timezone.utc).strftime('%m-%d-%Y H:%M:%S')
    
    @classmethod
    def from_iso(cls, dstring):
        try:
            return datetime.fromisoformat(dstring)
        except:
            datetime.strptime(dstring, '%Y-%m-%dT%H:%M:%S.%f%z')

    @classmethod
    def pdate(cls, dstring, from_past=True, as_string=True, to_iso=False, tformat='%m-%d-%Y H:%M:%S'):
        _prefer = 'past' if from_past else 'future'
        dt = dateparser.parse(dstring, settings={'PREFER_DATES_FROM': _prefer, 'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})
        if as_string:
            if to_iso:
                return dt.isoformat('T')
            return dt.strftime(tformat)
        return dt

    @property
    def _tm(self):
        return {
            's': {
                'vars': ['s', 'sec', 'secs', 'second', 'seconds'],
                'v': 1
            },
            'm': {
                'vars': ['m', 'min', 'mins', 'minute', 'minutes'],
                'v': 60
            },
            'h': {
                'vars': ['h', 'hr', 'hrs', 'hour', 'hours'],
                'v': 3600
            },
            'd': {
                'vars': ['d', 'dy', 'dys', 'day', 'days'],
                'v': 86400
            },
        }

    @classmethod
    def tm(cls, secs, tval='secs', as_string=False):
        for tv, vals in File._tm:
            if tval in vals['vars']:
                nval = secs / vals['v']
                if as_string:
                    return f'{nval:.2f} {tval}'
                return nval

    @classmethod
    def from_now(cls, dstring, as_string=True, as_dtime=False, as_time=False, to_iso=False, tval='secs', tformat='%m-%d-%Y H:%M:%S'):
        _now = Time.now(as_string=False)
        _past = Time.pdate(dstring, as_string=False, to_iso=True)
        dt = _now - _past
        if as_dtime:
            return dt
        if as_string:
            if to_iso:
                return dt.isoformat('T')
            return dt.strftime(tformat)
        if as_time:
            return Time.tm(dt.total_seconds(), tval=tval, as_string=as_string)

    @classmethod
    def secs_from_now(cls, dstring, as_string=False):
        return Time.from_now(dstring, as_string=as_string, as_time=True, tval='secs')
    
    @classmethod
    def mins_from_now(cls, dstring, as_string=False):
        return Time.from_now(dstring, as_string=as_string, as_time=True, tval='mins')
    
    @classmethod
    def hrs_from_now(cls, dstring, as_string=False):
        return Time.from_now(dstring, as_string=as_string, as_time=True, tval='hrs')
    

timer = Time.timer
now = Time.now
timestamp = Time.timestamp
iso_timestamp = Time.iso_timestamp
parse_date = Time.pdate