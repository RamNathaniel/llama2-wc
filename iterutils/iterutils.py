import itertools


class IterUtil(object):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return self.iterable.__iter__()

    def __next__(self):
        return self.iterable.__next__()

    def __getitem__(self, index):
        return self.iterable.__getitem__(index)

    def __len__(self):
        return self.iterable.__len__()

    def __contains__(self, item):
        return self.iterable.__contains__(item)

    def __reversed__(self):
        return self.iterable.__reversed__()

    def __str__(self):
        return self.iterable.__str__()

    def __repr__(self):
        return self.iterable.__repr__()

    def __eq__(self, other):
        return self.iterable.__eq__(other)

    def __ne__(self, other):
        return self.iterable.__ne__(other)

    def __lt__(self, other):
        return self.iterable.__lt__(other)

    def __le__(self, other):
        return self.iterable.__le__(other)

    def __gt__(self, other):
        return self.iterable.__gt__(other)

    def __ge__(self, other):
        return self.iterable.__ge__(other)

    def __add__(self, other):
        return self.iterable.__add__(other)

    def __mul__(self, other):
        return self.iterable.__mul__(other)

    def __rmul__(self, other):
        return self.iterable.__rmul__(other)

    def __iadd__(self, other):
        return self.iterable.__iadd__(other)

    def __imul__(self, other):
        return self.iterable.__imul__(other)

    def __contains__(self, item):
        return self.iterable.__contains__(item)

    def __getslice__(self, i, j):
        return self.iterable.__getslice__(i, j)

    def __setslice__(self, i, j, sequence):
        return self.iterable.__setslice__(i, j, sequence)

    def __delslice__(self, i, j):
        return self.iterable.__delslice__(i, j)

    def __hash__(self):
        return self.iterable.__hash__()

    def __sizeof__(self):
        return self.iterable.__sizeof__()
    
    def max(self, score_func=None):
        if len(self.iterable) == 0:
            return None
        
        if score_func is None:
            return max(self.iterable)
        
        return max(self.iterable, key=score_func)

    def max_item(self, score_func):
        if len(self.iterable) == 0:
            return None
        
        return max(self.iterable, key=score_func)

    def min(self):
        if len(self.iterable) == 0:
            return None
        
        return min(self.iterable)

    def min_item(self, score_func):
        if len(self.iterable) == 0:
            return None
        
        return min(self.iterable, key=score_func)

    def sum(self, score_func=None):
        if len(self.iterable) == 0:
            return 0

        if score_func is None:
            return sum(self.iterable)

        return sum(score_func(item) for item in self.iterable)
    
    def where(self, predicate):
        return IterUtil((item for item in self.iterable if predicate(item)))
    
    def select(self, selector):
        return IterUtil((selector(item) for item in self.iterable))

    def first(self, predicate=None):
        if predicate is None:
            return next(iter(self.iterable))
        
        return next(item for item in self.iterable if predicate(item))
    
    def last(self, predicate=None):
        if predicate is None:
            return next(reversed(self.iterable))
        
        return next(reversed(self.iterable), None)
    
    def take(self, count):
        return IterUtil(itertools.islice(self.iterable, count))
    
    def skip(self, count):
        return IterUtil(itertools.islice(self.iterable, count, None))
    
    def do(self, action):
        for item in self.iterable:
            action(item)
