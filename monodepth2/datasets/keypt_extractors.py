import cv2
import numpy as np


orb_extractor = cv2.ORB_create()
sift_extractor = cv2.SIFT_create()


def get_keypts(img, extractor='orb'):

    if extractor == 'orb':
        extractor = orb_extractor
    elif extractor == 'sift':
        extractor = sift_extractor
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
