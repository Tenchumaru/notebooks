import cv2
import numpy as np
import pickle
from utilities import *

def fn(file_title):
    print(file_title)
    def fn(image):
        # Take the color channel with the lowest value.  This changes the shape
        # of the image from (60, 160, 3) to (60, 160).
        g = (image[:, :, i] for i in range(image.shape[2]))
        image = min(g, key=lambda image: image.sum())
        return image
    images = list(map(fn, pickle_iter(fr"D:\Dota 2\Heroes\Pickles\{file_title}.pickle")))

    # Combine the list of two-dimensional tensors into a single three-
    # dimensional tensor.
    images = np.stack(images, axis=0)

    # Determine the desired whiteness by computing the mean of all whiteness.
    desired_whiteness = np.mean(images, axis=0)
    print(desired_whiteness.shape, desired_whiteness.min(), desired_whiteness.max())
    yield desired_whiteness

    # Determine the desired factor by scaling the range of standard deviations
    # of all whiteness from [min, max] to [1, 0].
    std = np.std(images, axis=0)
    desired_factor = (std - np.max(std)) / (np.min(std) - np.max(std))

    # Apply a threshold to remove the background.
    image = (desired_factor * 255).astype(np.uint8)
    threshold = 76
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)
    desired_factor = image.astype(np.float32) / 255
    yield desired_factor

parameters = (
    # file_title, top, bottom, offset, *(letter, left, right)
    ('ancient_apparition', 5, 14, 49, (('A', 0, 7), ('N', 7, 13), ('C', 13, 19), ('I', 19, 23), ('E', 23, 28), ('N', 28, 35), ('T', 35, 40), ('A', 44, 50), ('P', 50, 56), ('P', 56, 61), ('A', 61, 67), ('R', 67, 73), ('I', 73, 77), ('T', 77, 83), ('I', 83, 87), ('O', 87, 93), ('N', 93, 99))),
    ('anti-mage', 4, 13, 68, (('A', 0, 8), ('N', 8, 16), ('T', 16, 23), ('I', 23, 27), ('-', 27, 32), ('M', 32, 41), ('A', 41, 48), ('G', 48, 56), ('E', 56, 62))),
    ('broodmother', 4, 13, 57, (('B', 0, 7), ('R', 7, 13), ('O', 13, 22), ('O', 22, 30), ('D', 30, 38), ('M', 38, 47), ('O', 47, 56), ('T', 56, 63), ('H', 63, 71), ('E', 71, 77), ('R', 77, 84))),
    ('centaur_warrunner', 5, 13, 48, (('C', 0, 6), ('E', 6, 12), ('N', 12, 18), ('T', 18, 23), ('A', 23, 29), ('U', 29, 35), ('R', 35, 41), ('W', 45, 52), ('A', 52, 59), ('R', 59, 65), ('R', 65, 71), ('U', 71, 77), ('N', 77, 84), ('N', 84, 90), ('E', 90, 96), ('R', 96, 102))),
    ('clinkz', 4, 13, 79, (('C', 0, 7), ('L', 7, 13), ('I', 13, 18), ('N', 18, 26), ('K', 26, 33), ('Z', 33, 40))),
    ('io', 4, 13, 93, (('I', 0, 4), ('O', 4, 13))),
    ('juggernaut', 4, 13, 63, (('J', 0, 5), ('U', 5, 13), ('G', 13, 21), ('G', 21, 29), ('E', 29, 35), ('R', 35, 42), ('N', 42, 50), ('A', 50, 58), ('U', 58, 66), ('T', 66, 73))),
    ('keeper_of_the_light', 5, 13, 49, (('K', 0, 6), ('E', 6, 12), ('E', 12, 17), ('P', 17, 23), ('E', 23, 28), ('R', 28, 34), ('O', 37, 44), ('F', 44, 49), ('T', 52, 58), ('H', 58, 64), ('E', 64, 70), ('L', 73, 78), ('I', 78, 81), ('G', 81, 88), ('H', 88, 95), ('T', 95, 100))),
    ("nature's_prophet", 4, 13, 47, (('N', 0, 8), ('A', 8, 15), ('T', 15, 22), ('U', 22, 29), ('R', 29, 36), ('E', 36, 42), ("'", 42, 45), ('S', 45, 52), ('P', 55, 62), ('R', 62, 68), ('O', 68, 77), ('P', 77, 83), ('H', 83, 91), ('E', 91, 97), ('T', 97, 104))),
    ('nyx_assassin', 4, 13, 60, (('N', 0, 8), ('Y', 8, 16), ('X', 16, 23), ('A', 26, 34), ('S', 34, 40), ('S', 40, 46), ('A', 46, 53), ('S', 53, 59), ('S', 59, 65), ('I', 65, 70), ('N', 70, 78))),
    ('outworld_devourer', 5, 14, 49, (('O', 0, 7), ('U', 7, 14), ('T', 14, 19), ('W', 19, 27), ('O', 27, 34), ('R', 34, 40), ('L', 40, 45), ('D', 45, 51), ('D', 53, 60), ('E', 60, 65), ('V', 65, 71), ('O', 71, 77), ('U', 77, 84), ('R', 84, 89), ('E', 89, 94), ('R', 94, 100))),
    ('queen_of_pain', 4, 14, 57, (('Q', 0, 8), ('U', 8, 16), ('E', 16, 23), ('E', 23, 29), ('N', 29, 37), ('O', 41, 49), ('F', 49, 55), ('P', 59, 65), ('A', 65, 72), ('I', 72, 77), ('N', 77, 85))),
)

file_path = r'D:\Dota 2\Heroes\letters.pickle'
if os.path.isfile(file_path):
    os.remove(file_path)
for file_title, top, bottom, offset, pairs in parameters:
    desired_whiteness, desired_factor = fn(file_title)
    with open(file_path, 'ab') as fout:
        for letter, left, right in pairs:
            letter_whiteness = desired_whiteness[top:bottom, left + offset:right + offset]
            letter_factor = desired_factor[top:bottom, left + offset:right + offset]
            pickle.dump(letter, fout)
            pickle.dump(letter_whiteness, fout)
            pickle.dump(letter_factor, fout)
with open(file_path, 'ab') as fout:
    pickle.dump(None, fout)
