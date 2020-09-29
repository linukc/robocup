from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout
#import segmentation_models as sm
import cv2
import numpy as np
import pyrealsense2 as rs

class FixedDropout(Dropout):
	def _get_noise_shape(self, inputs):
		if self.noise_shape is None:
			return self.noise_shape
		return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])
customObjects = {
	'swish': nn.swish,
	'FixedDropout': FixedDropout
}

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

try:
	pipeline.start(config)
except:
	pipeline.stop()
	pipeline.start(config)

#model = sm.Unet('efficientnetb3',
#		classes=6,
#		activation='softmax',
#		input_shape=(480, 640, 3))

#model.load_weights('best_model.h5')
model = load_model('model_v3_from_checkpoint.h5', custom_objects=customObjects)

while True:
	frames = pipeline.wait_for_frames()
	color_frame = frames.get_color_frame()
	if not color_frame:
		continue
	color_image = np.asanyarray(color_frame.get_data())
	batch_with_one_image = np.expand_dims(color_image, axis=0)
	mask = model.predict(batch_with_one_image).squeeze()
	beam = cv2.resize(mask[..., 0], (192, 256))
	shaft = cv2.resize(mask[..., 1], (192, 256))
	sleeve = cv2.resize(mask[..., 2], (192, 256))
	square = cv2.resize(mask[..., 3], (192, 256))
	kremlin = cv2.resize(mask[..., 4], (192, 256))
	images = np.hstack((beam, shaft, sleeve, square, kremlin))
	cv2.imshow('RealSense', images)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#pipeline.stop()
cv2.destroyAllWindows()



