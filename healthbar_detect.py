import cv2
import numpy as np
from os import listdir
from time import time


# Two functions have been implemented: the standard detect_healthbar,
# as well as a detect_healthbar_fast, which reduces computation time
# at the cost of lower accuracy. See the test metrics for more details.


# Testing procedure and metrics:
# A total of 117 screen captures were taken from the sample short_game.mp4 
# video at an interval of 10 seconds. All of these frames were run through
# both detection algorithms, and any discrepancies were highlighted.
# The results are as follows:

# Standard detection [ detect_healthbar ]
# Total time taken: 650 seconds / 5.56 seconds per frame
# Sensitivity: 145/167 = 86.8%
# Total False Positives: 4
# Perfectly Identified Frames: 95/117 = 81.2%

# Fast detection [ detect_healthbar_fast ]
# Total time taken: 109 seconds / 0.93 seconds per frame
# Sensitivity: 139/167 = 83.2%
# Total False Positives: 4
# Perfectly Identified Frames: 91/117 = 77.8%


threshold = 0.2
min_dist = 33
templates_grayscale_source = 'templates/480p/arena/health_bar_grayscale'
templates_source = 'templates/480p/arena/health_bar'

templates_grayscale = [cv2.imread('{}/{}'.format(templates_grayscale_source, drct), cv2.IMREAD_GRAYSCALE) for drct in listdir(templates_grayscale_source)]
templates = [cv2.imread('{}/{}'.format(templates_source, drct)) for drct in listdir(templates_source)]


def downscale_480(image):
	height, width, _ = image.shape
	if height == 480: return image
	return cv2.resize(image, None, fx=852/width, fy=480/height, interpolation=cv2.INTER_AREA)

def pos(num):
	return num if num >= 0 else 0

def pad(rois, padding):
	return [((pos(roi[0][0]-padding), pos(roi[0][1]-padding)), (roi[1][0]+padding, roi[1][1]+padding)) for roi in rois]

def grayscale_pass(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	all_matches = []
	for template in templates_grayscale:
		height, width = template.shape[0], template.shape[1]
		search = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
		matches = np.where(search <= threshold)
		for pt in zip(*matches[::-1]):
			if not all_matches:
				all_matches.append((pt, (pt[0]+width, pt[1]+height)))
			else:
				if min([((m[0][0]-pt[0])**2 + abs(m[0][1]-pt[1])*100)**0.5 for m in all_matches]) > min_dist:
					all_matches.append((pt, (pt[0]+width, pt[1]+height)))
	return pad(all_matches, 10)


def detect_healthbar(image):
	image = downscale_480(image)
	all_matches = []
	for template in templates:
		height, width, _ = template.shape
		search = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
		matches = np.where(search <= threshold)
		for pt in zip(*matches[::-1]):
			if not all_matches:
				all_matches.append(pt)
				cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)
			else:
				if min([((m[0] - pt[0])**2 + abs(m[1] - pt[1])*100)**0.5 for m in all_matches]) > min_dist:
					all_matches.append(pt)
					cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)
				else:
					cv2.circle(image, pt, 3, (0, 255, 0), -1)
	return image, len(all_matches)

def detect_healthbar_fast(image):
	image = downscale_480(image)
	rois = grayscale_pass(image)
	total = 0
	for roi in rois:
		target = image[int(roi[0][1]):int(roi[1][1]),int(roi[0][0]):int(roi[1][0])]
		all_matches = []
		for template in templates:
			height, width, _ = template.shape
			search = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
			matches = np.where(search <= threshold)
			for pt in zip(*matches[::-1]):
				if not all_matches:
					all_matches.append(pt)
				else:
					if min([((m[0]-pt[0])**2 + abs(m[1]-pt[1])*100)**0.5 for m in all_matches]) > min_dist:
						all_matches.append(pt)
					else:
						cv2.circle(image, (pt[0]+roi[0][0], pt[1]+roi[0][1]), 3, (0, 255, 0), -1)
		for pt in all_matches:
			cv2.rectangle(image, (pt[0]+roi[0][0], pt[1]+roi[0][1]), (pt[0]+roi[0][0]+width, pt[1]+roi[0][1]+height), (0, 0, 255), 2)
			total += 1
	return image, total