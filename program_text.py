import cv2
import numpy as np
import UdpComms as U
import time
import json
import serial
import pytesseract

# non_max_suppression taken from https://github.com/PyImageSearch/imutils/blob/master/imutils/object_detection.py#L4
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

print("serial", serial.__version__)
ser = serial.Serial('COM5', 9800, timeout=1)

# Lets wait a little for the Arduino to setup
time.sleep(2)

# Create UDP socket to use for sending (and receiving)
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

cv2.namedWindow("output")
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

height, width, channels = frame.shape
print(width, height)

pressed_switch = 1
x_input = 512
y_input = 512
line = ''


# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
    ]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('./frozen_east_text_detection.pb')

def get_truncated (string):
    perm = []

    # forward pass
    for i in range(len(string)):
        sub_string = string[i:]
        perm.append(sub_string)
    # backwards pass
    for i in range(len(string)):
        sub_string = string[:i]
        perm.append(sub_string)

    # both pass
    for i in range(len(string)):
        sub_string = string[i: len(string) - 1]
        perm.append(sub_string)

    return perm

# Define known texts
known_text_base = ['birch', 'broad', 'rock', 'cliff', 'bush', 'branch', 'grass', 'flower', 'fir']
known_text = []
known_text_mapping = {}

# Lets also add all the subsets of the known text
for item in known_text_base:
     power_set = get_truncated(item)
     for subset_item in power_set:
          if len(subset_item) > 2:
            known_text.append(subset_item)
            known_text_mapping[subset_item] = item

print('known words', known_text_base)

while rval:
    while ser.in_waiting:
        line = ser.readline()   # read a byte

    if line:
        string = line.decode()  # convert the byte string to a unicode string

        # Only update user info if we get 3 values from the Ardino serial output
        values = string.strip().split()
        if len(values) == 3:
            # Update user inputs
            pressed_switch_raw, x_input_raw, y_input_raw = values

            # Parse user inputs, if their invalid set them to their default values
            pressed_switch = int(pressed_switch_raw) if pressed_switch_raw.isdecimal() else 1
            x_input = int(x_input_raw) if x_input_raw.isdecimal() else 512
            y_input = int(y_input_raw) if y_input_raw.isdecimal() else 512

    frame = cv2.flip(frame, -1)

    img = frame.copy()
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets

    blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(0, numCols):
            # ignore probability values below 0.75
            if scoresData[x] < 0.75:
                continue
            
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # an object contains a centre position, points, and type
    detected_objects = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        padding = 12
        startX = int(startX) - padding
        startY = int(startY) - padding
        endX = int(endX) + padding
        endY = int(endY) + padding

        y = startY
        x = startX
        h = endY - startY
        w = endX - startX

        centerX = x
        centerY = y

        # make sure we're not getting a rect that is outside the image bounds
        if y + h < height and x + w < width and x >= 0 and y >= 0:
            rect = frame[y:y+h, x:x+w]

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(rect, lang='eng', config='--oem 3 --psm 8').strip().lower()
            
            if len(text) > 0 and text in known_text:
                # We've detected known text
                root_text = known_text_mapping[text]
                cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 2)

                detected_objects.append({
                    "x" : ((centerX / width) - 0.5) * 2.0,
                    "y" : ((centerY / height) - 0.5) * 2.0,
                    "type" : root_text,
                })
            else:
                # draw the bounding box on the image
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("output", img)

    #cv2.imshow("output", img)
    rval, frame = vc.read()

    # Send detected objects to unity
    sock.SendData(json.dumps({"data" : detected_objects, "userInputX" : x_input, "userInputY" : y_input, "userInputSwitch" : pressed_switch}))

    # Check if we should shutdown the server
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("output")

# Close serial
ser.close()

# Thanks for reading :)