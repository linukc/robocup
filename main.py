import cv2
import tensorrt as trt
from segmentation_models import get_preprocessing
import pyrealsense2 as rs
from utils import make_binary, get_info_c, get_camera, predict, pars, make_msg_c, MySocket, load_engine
from utils import make_hole, get_info_p, make_msg_p
from utils import save_image

def main():
	#eventloop
	while True:
		print('ready')
		input = socket.myreceive().decode()
		command, sync, obj_index = pars(input)
		
		print(sync, obj_index)
		color, depth, camera_params = get_camera(pipeline, align)
		
		if command == 'c':
			probability_mask = predict(engine, preproc(color), obj_index)
			binary_mask = make_binary(probability_mask, border=185)
			info = get_info_c(obj_index, binary_mask, depth, camera_params, min_area=1500)
			if not info:
				msg = 'c:{}:None'.format(sync)
			else:
				x, y, z, angle = info
				msg = make_msg_c(sync, x, y, z, angle)
			print('---Message---')
			print(msg)
			print('----------------------')
			print()
			#save for debug
			save_image(color, depth, probability_mask, binary_mask)
			
		elif command == 'p':	
			bin_mask = make_hole(color)
			info = get_info_p(bin_mask, obj_index, camera_params)
			if not info:
				msg = 'p:{}:None'.format(sync)
			else:
				x, y, z = info
				msg = make_msg_p(sync, x, y, z)
			print('---Message---')
			print(msg)
			print('----------------------')
			print()
				
		socket.mysend(msg.encode(), '172.34.0.254', 19090)			

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

try:
	main()
except KeyboardInterrupt:
	pipeline.stop()
	print('finish')
except Exception as e:
	print(e)
