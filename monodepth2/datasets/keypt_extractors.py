import cv2
import numpy as np


def get_keypts(img, extractor='orb'):

    if extractor == 'orb':
        extractor = cv2.ORB_create()
    elif extractor == 'sift':
        extractor = cv2.SIFT_create()
    else:
        raise NotImplementedError('Extractor not found.')

    kp = extractor.detect(img, None)
    pts = np.array([x.pt for x in kp])

    return pts


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.cvtColor(cv2.imread('../assets/test_image.jpg'), cv2.COLOR_BGR2RGB)
    keypts = get_keypts(img, 'sift').astype(np.int)

    print(len(keypts))

    img[keypts[:, 1], keypts[:, 0]] = [0, 255, 0]

    plt.imshow(img)
    plt.show()
