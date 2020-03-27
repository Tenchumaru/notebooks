import cv2
import itertools as it
import numpy as np
import os
import random
from datetime import datetime

random.seed(9)
np.random.seed(9)

def get_file_names(input_directory_path):
    _, _, file_names = next(os.walk(input_directory_path), (None, None, []))
    return file_names

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)

def show_and_wait(frame, title='tesst', wait_time=0):
    cv2.imshow(title, frame)
    key = cv2.waitKey(wait_time)
    return None if key < 0 else chr(key)

def read_and_wait(video, title='tesst', wait_time=0):
    result, frame = video.read()
    if result:
        return show_and_wait(frame, title, wait_time)

def play_video(video, title='tesst', starting_index=0, wait_time=25):
    if video.set(cv2.CAP_PROP_POS_FRAMES, starting_index):
        while read_and_wait(video, title, wait_time) != 'q':
            continue

def for_plt(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # image.take([2, 1, 0], axis=2)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def yield_frames(video):
    while True:
        result, frame = video.read()
        if not result:
            break
        yield frame

def print_date_formatters():
    g = (chr(i) for i in range(ord('A'), ord('Z') + 1))
    for ch in g:
        try:
            print(ch, datetime.now().strftime('%' + ch))
        except:
            pass
        try:
            print(ch.lower(), datetime.now().strftime('%' + ch.lower()))
        except:
            pass

def demonstrate_opencv_bgr():
    # Demonstrate BGR nature of OpenCV.
    for i in range(3):
        image = np.zeros((88, 88, 3), 'uint8')
        image[:, :, i] = 255
        print(image[0, 0])
        show_and_wait(image)
    cv2.destroyAllWindows()

# This differs from random.sample.  It produces a generator and the selected elements are in original order.
def sample(l, k):
    if k < 0:
        raise ValueError('desired sample size is negative')
    n = len(l)
    if n < k:
        raise ValueError('desired sample size is greater than population size')
    def fn(l, n, k):
        for x in l:
            if random.randrange(0, n) < k:
                yield x
                k -= 1
                if k < 1:
                    break
            n -= 1
    return fn(l, n, k)

def demonstrate_sample():
    print([list(sample(list(range(22)), 9)) for _ in range(9)])

def partition(l, k):
    n = len(l)
    samples = sorted(random.sample(range(n), k))
    g = iter(samples)
    j = next(g, None)
    left, right = [], []
    for i, x in enumerate(l):
        if i == j:
            left.append(x)
            j = next(g, None)
        else:
            right.append(x)
    return left, right

def demonstrate_partition():
    print([partition(list(range(22)), 9) for _ in range(9)])

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
