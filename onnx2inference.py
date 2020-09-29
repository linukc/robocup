import cv2
import onnxruntime as rt
import numpy as np
import segmentation_models as sm

prep = sm.get_preprocessing('efficientnetb3')
sess = rt.InferenceSession('/home/sergey/librealsense/wrappers/python/examples/model_v4_onnx.onnx')

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)
x = np.expand_dims(prep(cv2.cvtColor(cv2.imread('30_Color.png'),cv2.COLOR_BGR2RGB)).astype(np.float32), axis=0)
print(x.dtype, x.shape)
res = sess.run([output_name], {input_name: x})
mask = res[0].squeeze()[..., 0]
while True:

	cv2.imshow('RealSense', mask)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

