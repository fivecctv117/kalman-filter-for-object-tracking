# Import python libraries
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as pl


# set to 1 for pipeline images
debug = 0


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.select_centers = [[190, 88]]
        self.select_rects = [[[190], [190], [18], [60]]]
        self.roi_rect = []
        self.histGrayRoi = []
        self.cnt = 0
        self.isUseCenter = False

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        blurred = cv2.GaussianBlur(frame,(3,3),0)

        # Convert BGR to GRAY
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


        if (debug == 1):
            cv2.imshow('gray', gray)


        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)

        if (debug == 0):
            cv2.imshow('bgsub', fgmask)

        # Detect edges
        edges = cv2.Canny(fgmask, 50, 150, 3)

        if (debug == 1):
            cv2.imshow('Edges', edges)

        # Retain only edges within the threshold
        ret, thresh = cv2.threshold(edges, 50, 255, 0)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

        if (debug == 0):
            cv2.imshow('thresh', thresh)

        centers = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tunned based on expected object size for
        # improved performance
        blob_radius_thresh = 10
        radius_array = []
        rect_array = []

        #test = self.select_rects[0]
        for i in range(len(self.select_rects)):
            test = self.select_rects[i]
            if(self.cnt == 1):
                cv2.rectangle(frame, (test[0][0], test[1][0]), (test[0][0] + test[2][0], test[1][0] + test[3][0]), (0, 255, 0), 2)
        cv2.imshow('rectangle', frame)        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                x, y, w,h = cv2.boundingRect(cnt)
                if (w > blob_radius_thresh and h > blob_radius_thresh):
                    b = np.array([[x+w/2], [y+h/2]])
                    centers.append(np.round(b))
                    rect_array.append([[x],[y],[w], [h]])
            except ZeroDivisionError:
                pass
        # show contours of tracking objects
        # cv2.imshow('Track Bugs', frame)
        if(len(centers) == 0):
            return centers
        select_cts,c_ind = self.SelectCenters(centers)
        select_rcts,r_ind = self.SelectRects(gray,centers,rect_array)

        for i in range(len(select_cts)):
            if(select_rcts[i][0][0] != select_cts[i][0][0] or select_rcts[i][1][0] != select_cts[i][1][0] ):
                select_rcts[i] = select_cts[i]
                r_ind[i] = c_ind[i]
            elif self.isUseCenter is False:
                print(self.cnt, 'use color hist')

            if (rect_array[r_ind[i]][2][0] > blob_radius_thresh and rect_array[r_ind[i]][3][0] > blob_radius_thresh):
                rect = rect_array[r_ind[i]]
                if(self.cnt == 29 or self.cnt == 80 or self.cnt == 177 or self.cnt == 381):
                    dect_rect = frame[rect[1][0]:rect[1][0] + rect[3][0],
                            rect[0][0]:rect[0][0] + rect[2][0]]
                    self.img_hist(frame, dect_rect)
                cv2.rectangle(frame, (rect[0][0], rect[1][0]), (rect[0][0] + rect[2][0], rect[1][0] + rect[3][0]), (0, 255, 0), 2)

        self.isUseCenter = False
        return select_rcts
        #return centers

    def SelectCenters(self,detect_centers):
        N = len(self.select_centers)
        M = len(detect_centers)
        cost = np.zeros(shape=(N, M))
        costInd = np.array([0,0])
        min_distance = 0
        if N == 1 and M == 1:
            return detect_centers, [0]

        for i in range(len(self.select_centers)):
            for j in range(len(detect_centers)):
                try:
                    diff = self.select_centers[i] - detect_centers[j]
                    distance = np.sqrt(diff[0][0] * diff[0][0] +
                                   diff[1][0] * diff[1][0])
                    cost[i][j] = distance
                except:
                    pass


        assignment = []
        dist_thresh = 30
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        self.select_centers = []
        detect_ind = []
        for i in range(len(assignment)):
            if(assignment[i] != -1 and cost[i][assignment[i]] < dist_thresh):

                self.select_centers.append(detect_centers[assignment[i]])
                detect_ind.append(assignment[i])

        return self.select_centers, detect_ind

    def SelectRects(self,frame,centers,detect_rects):
        N = len(self.select_rects)
        M = len(detect_rects)
        correl = np.zeros(shape=(N, M))
        costInd = np.array([0, 0])
        min_distance = 0
        if N == 1 and M == 1:
            self.isUseCenter = True
            return centers, [0]

        for i in range(len(self.select_rects)):
            # calculate select rectangle Hist and normalize
            for j in range(len(detect_rects)):
                dect_rect = frame[detect_rects[j][1][0]:detect_rects[j][1][0] + detect_rects[j][3][0],detect_rects[j][0][0]:detect_rects[j][0][0] + detect_rects[j][2][0]]
                histGrayDect = cv2.calcHist([dect_rect], [0], None, [256], [0, 255])
                cv2.normalize(histGrayDect, histGrayDect, 0, 255 * 0.9, cv2.NORM_MINMAX)

                correlation = cv2.compareHist(self.histGrayRoi[i], histGrayDect, cv2.HISTCMP_CORREL)

                correl[i][j] = correlation


        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(correl,True)
        for i in range(len(row_ind)):
                assignment[row_ind[i]] = col_ind[i]

        print(self.cnt, np.max(correl[0]))
        self.select_centers = []

        detect_ind = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                self.select_centers.append(centers[assignment[i]])
                detect_ind.append(assignment[i])

        return self.select_centers, detect_ind

    def init_HistRects(self, frame):
        for i in range(len(self.select_rects)):
            # calculate select rectangle Hist and normalize
            roi_rect = frame[self.select_rects[i][1][0]:self.select_rects[i][1][0]+self.select_rects[i][3][0],self.select_rects[i][0][0]:self.select_rects[i][0][0] + self.select_rects[i][2][0]]
            histGrayRoi = cv2.calcHist([roi_rect], [0], None, [256], [0, 255])
            cv2.normalize(histGrayRoi, histGrayRoi, 0, 255 * 0.9, cv2.NORM_MINMAX)
            self.roi_rect.append(roi_rect)
            self.histGrayRoi.append(histGrayRoi)
        test = self.select_rects[0]


    def img_hist(self,frame,detect):

        gray = cv2.cvtColor(self.roi_rect[0], cv2.COLOR_BGR2GRAY)
        gray_detect = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
        #cv2.normalize(gray, gray, 0, 255 * 0.9, cv2.NORM_MINMAX)
        #cv2.normalize(gray_detect, gray_detect, 0, 255 * 0.9, cv2.NORM_MINMAX)
        hist_roi = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_detect = cv2.calcHist([gray_detect], [0], None, [256], [0, 256])

        pl.plot(hist_roi,'y*-')
        pl.plot(hist_detect,'m--')

        pl.xlim([0, 256])
        pl.title('frame '+ str(self.cnt))
        pl.show()
