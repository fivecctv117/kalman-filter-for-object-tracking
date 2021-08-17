# Import python libraries
import cv2
import copy
from detectors import Detectors
from detectors2 import Detectors2


from tracker import Tracker

def readImgSeq(src, num):
    strnum = '0'
    if (num<10):
        strnum = strnum + '00' + str(num)
    elif num >= 10 and num < 100:
        strnum = strnum + '0' + str(num)
    elif num >=100:
        strnum = strnum + str(num)
    img = cv2.imread(src +'/' + strnum + '.jpg', cv2.IMREAD_COLOR)
    return img



def main():
    """Main function for multi object tracking
    Usage:
        $ python2.7 objectTracking.py
    Pre-requisite:
        - Python2.7
        - Numpy
        - SciPy
        - Opencv 3.0 for Python
    Args:
        None
    Return:
        None
    """

    # Create opencv video capture object
    cap = cv2.VideoCapture('F:/testfile/record1.mp4')
    #cap = cv2.VideoCapture('F:/opencv-4.4.0/opencv-4.4.0/samples/data/vtest.avi')

    print(cap.get(cv2.CAP_PROP_FPS))
    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker
    tracker = Tracker(10, 30, 10, 100)

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False
    centers = [[[192],[44]]]  #record1
    #centers = [[[365], [27]]]  #record2
    #centers = [[[640],[240]]]  #vtest
    #centers = [[[112],[100]],[[186],[83]]]  #vtest

    #centers = [[[200],[142]]]


    detector.select_centers = centers
    detector.select_rects = [[[192], [44], [23], [70]]]
    #detector.select_rects = [[[365], [27], [30], [65]]]
    #detector.select_rects = [[[640],[240],[20],[70]],[[186],[83],[30],[110]]]

    # Infinite loop to process video frames
    src = 'F:/testfile/Jogging/Jogging/img'
    num = 307
    cnt = 1
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        #frame = readImgSeq(src, cnt)

        if (cnt == 1):
            detector.init_HistRects(frame)


        #cnt = cnt + 1
        # Make copy of original frame
        orig_frame = copy.copy(frame)

        # Skip initial frames that display logo
        if (skip_frame_count < 15):
            skip_frame_count += 1
            #cnt += 1
            continue
        detector.cnt = cnt

        #if (cnt == 29 or cnt == 80 or cnt == 177 or cnt == 381 ):
        if (cnt == 29  or cnt == 177 or cnt == 400 or cnt == 600):

            print(cnt)
        # Detect and return centeroids of the objects in the frame
        centers = detector.Detect(frame)



        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9

            # Display the resulting tracking frame
            cv2.imshow('Tracking', frame)

        # Display the original frame
        cv2.imshow('Original', orig_frame)
        cnt = cnt + 1
        # Slower the FPS
        cv2.waitKey(50)

        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()
