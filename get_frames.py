from video_stream import VideoStream
import numpy as np 
import cv2
import time
import os
import uuid
import pandas as pd

PATH = './data_siw'
PATH_FACE_DETECTOR = './face_detector'
SKIP = 10
result = []

#face detection net
proto_path = os.path.join(PATH_FACE_DETECTOR, 'deploy.prototxt.txt')
model_path = os.path.join(PATH_FACE_DETECTOR, 'res10_300x300_ssd_iter_140000.caffemodel')
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

def get_frames(file_path, type):
	#start video capture
	videoStream = VideoStream(file_path).start()
	time.sleep(1.0)

	start_time = time.time()
	count = 0
	while videoStream.more():
		frame = videoStream.read()
		count += 1
		if count % SKIP != 0:
			continue

		h, w = frame.shape[:2]
		#face detection
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()

		if len(detections) > 0:
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY-100:endY+100, startX-100:endX+100]
				face = cv2.resize(face, (224, 224))
				name = str(uuid.uuid4())
				file_name = './output/' + name + '.png'
				cv2.imwrite(file_name, face)
				result.append((name, type))

for root, dirs, files in os.walk(PATH):
	if files:
		for file in files:
			print("Processing: {}".format(file))
			type = os.path.relpath(root, PATH)
			get_frames(os.path.join(root, file), type)

print("Get {} images".format(len(result)))
data = pd.DataFrame(result, columns=['file_name', 'class'])
data.to_csv('data.csv', index=False)