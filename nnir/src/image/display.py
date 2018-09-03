import cv2


def display_image(lb, path):
    im = cv2.imread(path)
    cv2.imshow(lb, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
