import cv2
import itertools as it
import math
import numpy as np
import os
import pickle
import random
from datetime import datetime

random.seed(9)
np.random.seed(9)

class Timer:
    def __init__(self, message: str):
        self.__message = message

    def __enter__(self):
        self.__start_at = datetime.now()

    def __exit__(self, *args):
        print(f'{self.__message} time:', (datetime.now() - self.__start_at).total_seconds(), 'seconds')

def Grayscale(image):
    if getattr(image, 'shape', None) is None:
        image = np.array(list(image.getdata())).reshape((image.height, image.width, 3))
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    image = image[:, :, np.newaxis]
    return image

def create_square_image(images, aspect_ratio=1):
    image = np.zeros_like(images[0])
    nrows, ncolumns, _ = images[0].shape
    width = math.ceil(math.sqrt(len(images) * aspect_ratio * nrows / ncolumns))
    return np.vstack([np.hstack((images[i:i+width] + [image] * width)[:width]) for i in range(0, len(images), width)])

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
        while True:
            result, frame = video.read()
            if result:
                return show_and_wait(frame, title, wait_time)
            else:
                break
        cv2.destroyAllWindows()

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

def pickle_iter(fin, is_ignoring_all=False):
    if type(fin) is str:
        fin = open(fin, 'rb')
    with fin:
        try:
            while True:
                yield pickle.load(fin)
        except Exception as ex:
            if type(ex) is EOFError or is_ignoring_all:
                return
            raise

def prime_factors(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            yield i
    if n > 1:
        yield n

def put_text(image: np.ndarray, text: str, x: int, y: int, *, scale: int=1, color: tuple[int, int, int]=(255,), thickness: int=2) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
