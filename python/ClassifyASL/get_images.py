import cv2
import numpy as np
from imageio import imwrite

from test import model
from test import getMaxIndex
from test import get_guess_text

cap = cv2.VideoCapture(0)


class FileInfo:
    location = "captured/"
    symbol = "I"
    index = 1
    extra = "isaiah"


def get_file_name():
    FileInfo.index = FileInfo.index + 1
    return FileInfo.location + FileInfo.symbol + "." + str(FileInfo.index) + "." + FileInfo.extra + ".jpg"


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img0 = cv2.resize(frame, (200, 200))
    img = cv2.resize(img0, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    np_array = np.array(gray)
    inputArray = np_array.reshape(-1, 64, 64, 1)

    guess = model.predict(inputArray)
    print(getMaxIndex(guess))
    print(get_guess_text(getMaxIndex(guess)))

    # Display the resulting frame
    cv2.imshow('frame', img0)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    if cv2.waitKey(100) & 0xFF == ord('c'):
        print("Pressing c")
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        imwrite(get_file_name(), img0)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
