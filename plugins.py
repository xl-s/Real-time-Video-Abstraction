import cv2
import numpy as np
import pytesseract
from datetime import datetime
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def time_since_start(time_start):
	time_diff=datetime.now()-time_start
	return time_diff

class Record:

	def __init__(self, output_name='output.mp4'):
		self.output_name = output_name

	def start(self, parent):
		self.recording = False
		self.width, self.height = 0, 0
		self.frame_rate = parent.frame_rate
		self.time_start=parent.time_start


	def run(self, frame):
		if not self.width or not self.height:
			self.height, self.width, _ = frame.shape
		if self.recording: self.output.write(frame)
		return frame

	def start_record(self):
		self.output = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*'X264'), self.frame_rate, (self.width, self.height), True)
		self.recording = True
		print('Recording Started.')

	def stop_record(self):
		self.recording = False
		self.output.release()
		print('Recording Stopped.')

	def key_func(self, key):
		if key == 32:    # space
			if self.recording:
				self.stop_record()
			else:
				self.start_record()

	def stop(self):
		if self.recording: self.stop_record()


class ColourTrack:

	def start(self, parent):
		self.threshold = 20
		self.colour = np.array([])
		self.frame = None
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('CT Threshold', 'Settings', self.threshold, 255, self.settings_handler)
		cv2.createTrackbar('CT Mask', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def settings_handler(self, pos):
		self.threshold = cv2.getTrackbarPos('CT Threshold', 'Settings')
		self.mask = cv2.getTrackbarPos('CT Mask', 'Settings')

	def mouse_event(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			try:
				self.colour = self.frame[y, x]
			except:
				pass

	def find_pixels(self, hsv, colour, threshold):
		colour = colour[0, 0, :]
		lower = np.array([colour[0] - threshold, colour[1] - threshold, colour[2] - threshold])
		upper = np.array([colour[0] + threshold, colour[1] + threshold, colour[2] + threshold])
		return cv2.inRange(hsv, lower, upper)

	def get_center(self, mask):
		i_s = 0
		j_s = 0
		count = 0
		for j, row in enumerate(mask):
			for i, val in enumerate(row):
				if val:
					i_s += i
					j_s += j
					count += 1
		if count:
			pos = (int(i_s/count), int(j_s/count))
		else:
			pos = 0
		return pos

	def run(self, frame):
		self.frame = frame

		if self.colour.any() and len(frame.shape) == 3:
			hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
			colour = cv2.cvtColor(np.uint8([[self.colour]]), cv2.COLOR_BGR2HSV)

			mask = self.find_pixels(hsv, colour, self.threshold)
			pos = self.get_center(mask)

			if pos: cv2.circle(frame, pos, 5, tuple([int(255-c) for c in self.colour]), -1)
			if self.mask: frame = mask

		return frame.astype(np.uint8)


class EdgeDetect:

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('ED Lower Threshold', 'Settings', 100, 255, self.settings_handler)
		cv2.createTrackbar('ED Upper Threshold', 'Settings', 200, 255, self.settings_handler)
		cv2.createTrackbar('ED Blue', 'Settings', 0, 1, self.settings_handler)
		cv2.createTrackbar('ED Green', 'Settings', 0, 1, self.settings_handler)
		cv2.createTrackbar('ED Red', 'Settings', 0, 1, self.settings_handler)
		cv2.createTrackbar('ED Gray', 'Settings', 0, 1, self.settings_handler)
		self.thresh_lower = 100
		self.thresh_upper = 200
		self.mask_blue = 0
		self.mask_green = 0
		self.mask_red = 0
		self.mask_gray = 0

	def settings_handler(self, pos):
		self.thresh_lower = cv2.getTrackbarPos('ED Lower Threshold', 'Settings')
		self.thresh_upper = cv2.getTrackbarPos('ED Upper Threshold', 'Settings')
		self.mask_blue = cv2.getTrackbarPos('ED Blue', 'Settings')
		self.mask_green = cv2.getTrackbarPos('ED Green', 'Settings')
		self.mask_red = cv2.getTrackbarPos('ED Red', 'Settings')
		self.mask_gray = cv2.getTrackbarPos('ED Gray', 'Settings')

	def get_canny(self, image):
		edges = cv2.Canny(image, self.thresh_lower, self.thresh_upper)
		return edges

	def run(self, frame):
		mask = np.zeros(frame.shape[:2])
		dims = len(frame.shape)
		if self.mask_blue and dims == 3:
			b = frame[:,:,0]
			blue_edges = self.get_canny(b)
			mask += blue_edges
		if self.mask_red and dims == 3:
			r = frame[:,:,2]
			red_edges = self.get_canny(r)
			mask += red_edges
		if self.mask_green and dims == 3:
			g = frame[:,:,1]
			green_edges = self.get_canny(g)
			mask += green_edges
		if self.mask_gray:
			if dims == 3:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			else:
				gray = frame
			gray_edges = self.get_canny(gray)
			mask += gray_edges

		if mask.any(): frame = mask
		return frame.astype(np.uint8)


class ColourFilter:

	def __init__(self, lower_hsv, upper_hsv, smooth=0, substitute=False, name='', invert=False):
		self.lower = np.array(lower_hsv, np.uint8)
		self.upper = np.array(upper_hsv, np.uint8)
		self.smooth = smooth
		self.sub = substitute
		self.name = name
		self.invert = invert

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('{} CF Mask'.format(self.name), 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('{} CF Mask'.format(self.name), 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def run(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		if self.mask:
			mask = cv2.inRange(hsv, self.lower, self.upper)
			if self.smooth:
				kernel = np.ones((self.smooth, self.smooth), np.uint8)
				mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
			if self.invert: mask = cv2.bitwise_not(mask)
			if self.sub:
				mask = cv2.bitwise_and(frame, frame, mask=mask)
			frame = mask
		return frame


class ROIFilter:

	def __init__(self, roi):
		roi = np.array(roi)
		if len(roi.shape) == 2: roi = np.array([roi])
		self.rois = roi
		if max(self.rois.reshape(-1)) > 1:
			self.width, self.height = 1, 1
		else:
			self.width, self.height = 0, 0

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('ROI Filter', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('ROI Filter', 'Settings', 0, len(self.rois), self.settings_handler)
		self.mask = 0

	def run(self, frame):
		if not self.width or not self.height:
			self.height, self.width, _ = frame.shape

		if self.mask:
			roi = self.rois[self.mask-1]
			frame = frame[int(roi[0,1]*self.height):int(roi[1,1]*self.height+1),
						  int(roi[0,0]*self.width):int(roi[1,0]*self.width+1)]
		else:
			for roi in self.rois:
				ul = tuple((roi[0]*[self.width, self.height]).astype(int))
				br = tuple((roi[1] *[self.width, self.height]).astype(int))
				cv2.rectangle(frame, ul, br, (0, 0, 255), 2)
		return frame


class ObjectFinder:

	def __init__(self, object_data=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'):
		self.cascade = cv2.CascadeClassifier(object_data)

	def run(self, frame):
		objects = self.cascade.detectMultiScale(frame, 1.3, 5)
		for x, y, w, h in objects:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		return frame.astype(np.uint8)


class BackgroundSubtractMOG:

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('BSM Mask', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('BSM Mask', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0
		self.subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

	def run(self, frame):
		mask = self.subtractor.apply(frame)
		if self.mask:
			frame = mask
		return frame


class BackgroundSubtractDirect:

	def __init__(self, threshold=10, minimum_samples=10, recency=30, sample_period=10, smooth=True):
		self.images = []
		self.minimum_samples = minimum_samples
		self.threshold = threshold
		self.recency = recency
		self.count = 0
		self.sample_period = sample_period
		self.smooth = smooth

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('BSD Mask', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('BSD Mask', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def apply(self, image):
		if self.recency:
			if len(self.images) > self.recency:
				self.images = self.images[1:]
		if self.count >= self.sample_period:
			self.images.append(image)
			self.count = 0
		else:
			self.count += 1

		if len(self.images) < self.minimum_samples: return image

		error = cv2.cvtColor((np.average([cv2.absdiff(ref, image) for ref in self.images], axis=0)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
		mask = np.where(error<=self.threshold, 0, 255).astype(np.uint8)
		kernel = np.ones((3, 3), np.uint8)

		if self.smooth: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		return mask

	def run(self, frame):
		mask = self.apply(frame)
		if self.mask:
			frame = mask
		return frame


class Invert:

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('Invert', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('Invert', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def run(self, frame):
		dims = len(frame.shape)
		if self.mask:
			if dims == 3:
				frame = (np.ones(frame.shape)*255 - frame).astype(np.uint8)
			elif dims == 2:
				frame = np.where(frame>0, 0, 255).astype(np.uint8)
		return frame


class ClickInfo:

	def mouse_event(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			bgr = self.frame[y,x]
			hsv = cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2HSV)
			prop = (x/self.width, y/self.height)
			print('\nPixel Location: ({}, {})'.format(x, y))
			print('RGB Colour: {}'.format(bgr[::-1]))
			print('HSV Colour: {}'.format(hsv[0][0]))
			print('Relative Location: ({}, {})'.format(round(prop[0], 5), round(prop[1], 5)))

	def start(self, parent):
		self.height, self.width = 0, 0

	def run(self, frame):
		if not self.height or not self.width:
			self.height, self.width, _ = frame.shape
		self.frame = frame
		return frame


class Otsu:

	def __init__(self, substitute=False):
		self.sub = substitute

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('Otsu', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('Otsu', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def run(self, frame):
		if self.mask:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			if self.sub:
				otsu = cv2.bitwise_and(frame, frame, mask=otsu)
			frame = otsu
		return frame


class TemplateMatch:

	def __init__(self, template, name=''):
		self.template = template
		self.height, self.width = self.template.shape[0], self.template.shape[1]
		self.name = name

	def settings_handler(self, pos):
		self.threshold = cv2.getTrackbarPos('{} TM Threshold'.format(self.name), 'Settings')/100

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('{} TM Threshold'.format(self.name), 'Settings', 80, 100, self.settings_handler)
		self.threshold = 0.8

	def run(self, frame):
		search = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
		if len(frame.shape) == 3:
			col = (0, 0, 255)
		else:
			col = (255, 255, 255)
		if self.threshold == 1:
			_, max_val, _, max_loc = cv2.minMaxLoc(search)
			cv2.rectangle(frame, max_loc, (max_loc[0] + self.width, max_loc[1] + self.height), col, 2)
		else:
			matches = np.where(search >= self.threshold)
			for pt in zip(*matches[::-1]):
				cv2.rectangle(frame, pt, (pt[0] + self.width, pt[1] + self.height), col, 2)
		return frame


class Screenshot:

	def key_func(self, key):
		if key == 32:
			self.take = True

	def start(self, parent):
		self.take = False

	def run(self, frame):
		if self.take:
			cv2.imwrite('screenshot.png', frame)
			self.take = False
			print('Screenshot taken.')
		return frame


class Gray:

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('Gray', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('Gray', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def run(self, frame):
		if self.mask:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return frame


class Contour:

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('Contour Mask', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('Contour Mask', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def run(self, frame):
		if self.mask:
			_, thresh = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			frame = np.zeros(frame.shape, np.uint8)
			frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
		return frame


class HoughLineHorizontal:

	def settings_handler(self, pos):
		self.minlength = cv2.getTrackbarPos('Min. Length', 'Settings')
		self.maxlength = cv2.getTrackbarPos('Max. Length', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('Min. Length', 'Settings', 100, 200, self.settings_handler)
		cv2.createTrackbar('Max. Length', 'Settings', 150, 200, self.settings_handler)
		self.minlength = 100
		self.maxlength = 150

	def run(self, frame):
		edges = cv2.Canny(frame, 50, 150, apertureSize=3)
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=self.minlength, maxLineGap=5)
		for line in lines:
			x1, y1, x2, y2 = line[0]
			if (not(y2 - y1)) and (abs(x2-x1) < self.maxlength):
				cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		return frame


class HoughLineVertical:

	def settings_handler(self, pos):
		self.minlength = cv2.getTrackbarPos('Min. Length', 'Settings')
		self.maxlength = cv2.getTrackbarPos('Max. Length', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('Min. Length', 'Settings', 100, 200, self.settings_handler)
		cv2.createTrackbar('Max. Length', 'Settings', 150, 200, self.settings_handler)
		self.minlength = 100
		self.maxlength = 150

	def run(self, frame):
		edges = cv2.Canny(frame, 50, 150, apertureSize=3)
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=self.minlength, maxLineGap=5)
		for line in lines:
			x1, y1, x2, y2 = line[0]
			if (not(x2 - x1)) and (abs(y2-y1) < self.maxlength):
				cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		return frame


class OCR:

	def __init__(self, config=''):
		self.config = config

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('OCR', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('OCR', 'Settings', 0, 1, self.settings_handler)
		self.mask = 0

	def run(self, frame):
		if self.mask:
			print(pytesseract.image_to_string(frame, config=self.config))
		return frame


class Pad:

	def settings_handler(self, pos):
		self.mask = cv2.getTrackbarPos('Pad', 'Settings')

	def start(self, parent):
		cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
		cv2.createTrackbar('Pad', 'Settings', 0, 20, self.settings_handler)
		self.mask = 0

	def run(self, frame):
		if self.mask:
			frame = cv2.copyMakeBorder(frame, self.mask, self.mask, self.mask, self.mask, cv2.BORDER_REPLICATE)
		return frame
