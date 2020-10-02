import cv2
import tensorrt as trt
from segmentation_models import get_preprocessing
import pyrealsense2 as rs
from utils import make_binary, get_info, get_camera, predict, pars, make_msg, MySocket, load_engine

#camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
for _ in range(20):
	pipeline.wait_for_frames()
#image
preproc = get_preprocessing('efficientnetb3')

#engine
ENGINE = 'modelv4.engine'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(trt_runtime, ENGINE)

#socket
socket = MySocket()
try:
	socket.bind('', 19090)
except:
	print('Cannot connect to host-port')

#eventloop
while True:
	print('ready')
	input = socket.myreceive().decode()
	sync, obj_index = pars(input)
	print(sync, obj_index)

	color, depth, camera_params = get_camera(pipeline, align)
	raw_mask = predict(engine, preproc(color), obj_index)
	binary_mask = make_binary(raw_mask, border=185)
	info = get_info(binary_mask, depth, camera_params, min_area=1500)
	if not info:
		msg = 'c:{}:None'.format(sync)
	else:
		x, y, z, angle = info
		msg = make_msg(sync, x, y, z, angle)
	print(msg)
	socket.mysend(msg.encode(), '192.168.0.208', 19090)

#final
pipeline.stop()
