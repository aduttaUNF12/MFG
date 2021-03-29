import math
import sys


class Fixed_Queue:

    def __init__(self, queue_len):
        self.queue = []
        self.queue_len = queue_len

    def qsize(self):
        return len(self.queue)

    def enq(self, item):
        if self.qsize() == self.queue_len:
            self.deq()
        self.queue.append(item)
    
    def deq(self):
        assert (self.qsize() > 0), "Can't dequeue from an empty queue."
        del self.queue[0]

    def first(self):
        assert (self.qsize() > 0), "No items in queue."
        return self.queue[0]
    
    def last(self):
        assert (self.qsize() > 0), "No items in queue."
        return self.queue[-1]
        

def squared_euclidean(P1, P2):
    out = 0.0
    for x in range(len(P1)):
        out = out + (P1[x] - P2[x])**2
    return out

def euclidean(P1, P2):
    out = squared_euclidean(P1, P2)
    return math.sqrt(out)

def manhattan(P1, P2):
    out = 0.0
    for x in range(len(P1)):
        out = out + abs(P1[x]-P2[x])
    return out

def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

#only works for 1D iterables of ints/floats
def minMax(itr):
    min = sys.float_info.max
    max = -1*sys.float_info.max
    
    for x in itr:
        min = x if x < min else min
        max = x if x > max else max
    
    return min, max

#only works for 1D iterables of ints/floats
def normalize(itr):
    min, max = minMax(itr)

    out = []
    for x in itr:
        out.append((x-min)/(max-min))

    return out
