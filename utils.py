import cv2
import math
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pyrealsense2 as rs
from pyrealsense2 import rs2_deproject_pixel_to_point as convert
import socket

class MySocket:
	def __init__(self, sock=None, recv_size=16):
		self.recv_size = recv_size
		if sock is None:
			self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		else:
			self.sock = sock

	def bind(self, host, port):
		self.sock.bind((host, port))

	def mysend(self, msg, host, port):
		self.sock.sendto(msg, (host, port))

	def myreceive(self):
		chunk, _ = self.sock.recvfrom(self.recv_size)
		return chunk

def load_engine(trt_runtime, plan_path):
	with open(plan_path, 'rb') as f:
		engine_data = f.read()
	engine = trt_runtime.deserialize_cuda_engine(engine_data)
	return engine

def allocate_buffers(engine, batch_size, data_type):
    """
    This is the function to allocate buffers for input and output in the device
    Args:
       engine : The path to the TensorRT engine.
       batch_size : The batch size for execution time.
       data_type: The type of the data for input and output, for example trt.float32.

    Output:
       h_input_1: Input in the host.
       d_input_1: Input in the device.
       h_output_1: Output in the host.
       d_output_1: Output in the device.
       stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)


def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
    """
    This is the function to run the inference
    Args:
       engine : Path to the TensorRT engine
       pics_1 : Input images to the model.
       h_input_1: Input in the host
       d_input_1: Input in the device
       h_output_1: Output in the host
       d_output_1: Output in the device
       stream: CUDA stream
       batch_size : Batch size for execution time
       height: Height of the output image
       width: Width of the output image

    Output:
       The list of output images

    """

    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
	#context.Profiler()
        #context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output.reshape((batch_size, height, width, -1))
        return out


def make_binary(mask, border):
	mask = np.where(mask < border, mask, 255)
	mask = np.where(mask >= border, mask, 0)
	return mask

def preprocessing_for_shaft(usedEdge, c_x, c_y, angle, mask):
	print('do preprocessing with shaft')
	axis = None
	angle1 = math.atan2(-usedEdge[1], usedEdge[0]) % 1.57
	angle2 = 1.57 + angle1
	num_pixels = 100
	trans = np.array([[math.cos(angle1), -math.sin(angle1)],
				[math.sin(angle1), math.cos(angle1)]])
	for i in range(1, num_pixels+2, 2):
		x_1_ = i
		y_1_ = 0
		x_2_ = 0
		y_2_ = i
		x_1, y_1 = map(int, np.dot(trans, np.array([x_1_, y_1_])))
		x_2, y_2 = map(int, np.dot(trans, np.array([x_2_, y_2_])))
		x1 = x_1 + c_x
		x2 = x_2 + c_x
		y1 = c_y - y_1
		y2 = c_y - y_2
		if mask[y1][x1] == 0 and mask[y2][x2] == 255:
			axis = 2
			x_1_ += 10
			y_1_ = 0
			x_1, y_1 = map(int, np.dot(trans, np.array([x_1_, y_1_])))
			x1 = x_1 + c_x
			y1 = c_y - y_1
			angle = 1.57 - angle2
			break
		elif mask[y1][x1] == 255 and mask[y2][x2] == 0:
			axis = 1
			x_2_ = 0
			y_2_ += 10
			x_2, y_2 = map(int, np.dot(trans, np.array([x_2_, y_2_])))
			x2 = x_2 + c_x
			y2 = c_y - y_2
			angle = 1.57 - angle1
			break

	if axis == 1:
		for j in range(1, num_pixels+2, 2):
			x_r = j
			x_l = -j
			y_ = y_2_
			xr, yr = map(int, np.dot(trans, np.array([x_r, y_])))
			xl, yl = map(int, np.dot(trans, np.array([x_l, y_])))
			x1 = xr + c_x
			y1 = c_y - yr
			x2 = xl + c_x
			y2 = c_y - yl
			if mask[y1][x1] == 0 and mask[y2][x2] == 255:
    				x, y = map(int, np.dot(trans, np.array([20, 0])))
    				c_x_new = x + c_x
    				c_y_new = c_y - y
    				return angle, c_x_new, c_y_new
			elif mask[y1][x1] == 255 and mask[y2][x2] == 0:
    				x, y = map(int, np.dot(trans, np.array([-20, 0])))
    				c_x_new = x + c_x
    				c_y_new = c_y - y
    				return angle, c_x_new, c_y_new

	elif axis == 2:
		for j in range(1, num_pixels+2, 2):
			y_u = j
			y_d = -j
			x_ = x_1_
			xu, yu = map(int, np.dot(trans, np.array([x_, y_u])))
			xd, yd = map(int, np.dot(trans, np.array([x_, y_d])))
			x1 = xu + c_x
			y1 = c_y - yu
			x2 = xd + c_x
			y2 = c_y - yd
			if mask[y1][x1] == 0 and mask[y2][x2] == 255:
    				x, y = map(int, np.dot(trans, np.array([0, 20])))
    				c_x_new = x + c_x
    				c_y_new = c_y - y
    				return angle, c_x_new, c_y_new
			elif mask[y1][x1] == 255 and mask[y2][x2] == 0:
				x, y = map(int, np.dot(trans, np.array([0, -20])))
				c_x_new = x + c_x
				c_y_new = c_y - y
				return angle, c_x_new, c_y_new
	else:
		print('cant do preproccesing for shaft, return lod values')
		return angle, c_x, c_y


def get_info(obj_index, mask, depth, params, min_area):
	contours, _ = cv2.findContours(np.array(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = np.int64(cv2.boxPoints(rect))
		(center_x, center_y), (length_x, length_y), _ = rect
		center_x, center_y = int(center_x), int(center_y)
		edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
		edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
		usedEdge = edge1
		if cv2.norm(edge2) > cv2.norm(edge1):
			usedEdge = edge2

		#vertical
		if usedEdge[1] == 0:
			angle = 90*math.pi/180
		else:
			angle = 90*math.pi/180 - math.atan2(-usedEdge[1], usedEdge[0])
		#shaft
		if obj_index == 2:
			try:
				angle, center_x, center_y = preprocessing_for_shaft(usedEdge, center_x, center_y, angle, mask)
			except Exception as e:
				return
		#depth
		try:
			depth_mat = depth[center_y-1:center_y+2, center_x-1:center_x+2]
			sum_ = 0
			count_ = 0
			for row in depth_mat:
				for elem in row:
					if elem != 0:
						sum_ += elem
						count_ += 1
			center_z = sum_ / count_
			print('z with neighbors')
		except ZeroDivisionError:
			print('zero_division_error')
			center_z = 0

		area = int(length_x * length_y)
		if area > min_area:
			box_x = list(set([para[0] for para in box]))
			box_y = list(set([para[1] for para in box]))
			for i in box_x:
				if i - 70 < 0 or i + 70 > 640:
					return
			for j in box_y:
				if j - 5 < 0 or j + 5 > 480:
					return
			if center_z == 0:
				print('zero depth')
				return
			else:
				x, y, z = convert(params, [center_x, center_y], center_z)
				return round(x,1), round(y,1), round(z, 1), round(angle, 3)
		else:
			print('Rejected', round(center_x,1), round(center_y,1), round(center_z,1), round(angle,3), round(area))
	print('Nothing to capture!')
	return

def make_hole(color):
	black_MIN = np.array([0, 0, 0],np.uint8)
	black_MAX = np.array([255, 255, 70],np.uint8)

	hsv_img = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

	frame = cv2.inRange(hsv_img, black_MIN, black_MAX)
	for i in range(285, frame.shape[0]):
		for j in range(frame.shape[1]):
			frame[i][j] = 0
			
	frame = cv2.medianBlur(frame, 7)
	
	return frame
			
def get_info_p(bin_mask, obj_index, params):
	center_z = 275
	
	contours, _ = cv2.findContours(np.array(bin_mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = np.int64(cv2.boxPoints(rect))
		(center_x, center_y), (l_x, l_y), _ = rect
		center_x, center_y = int(center_x), int(center_y)
		l_x, l_y = int(l_x), int(l_y)
		area = l_x * l_y
		
		if area > 4500:
			if  (160 <= l_y <= 200 and obj_index == 0 and area >= 8000) or (4500 <= area <= 6000 and obj_index == 1) or (0 <= abs(l_x-l_y) <= 20 and area >= 8000 and obj_index == 2):
				x, y, z = convert(params, [center_x, center_y], center_z)
				#
				im = np.dstack((bin_mask, bin_mask, bin_mask))
				cv2.drawContours(im, [box], -1, (255, 0, 0), 2)
				cv2.line(im, (box[0][0], box[0][1]), (center_x, center_y), (255, 0, 0))
				cv2.imwrite('prod/pr_mask.png', im)	
				print('find hole', l_x, l_y, area)
		
				return round(x,1), round(y,1), round(z, 2)		
	print('Holes didnt find!')
	return 
	
def make_msg_p(sync, x, y, z):
	return 'p:{}:{}:{}:{}'.format(sync, x, y, z)
	
def get_camera(pipeline, align):
	while True:
		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)
		aligned_depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()
		if aligned_depth_frame and color_frame:
			break

	d_in = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
	depth_image = np.asanyarray(aligned_depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())
	return color_image, depth_image, d_in


def predict(engine, color, obj_index):

	h_input, d_input, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)
	predict = do_inference(engine, color, h_input, d_input, h_output, d_output, stream, 1, 480, 640)

	predict = predict.squeeze()
	return predict[..., obj_index]*255

def pars(input):
	command, sync, obj_index = input.split(':')
	return command, int(sync), int(obj_index)

def make_msg(sync, x, y, z, angle):
	return 'c:{}:{}:{}:{}:{}'.format(sync, x, y, z, angle)

def save_image(color,depth,  predict, binary_mask):
	cv2.imwrite('prod/engine_predict.png', predict)
	cv2.imwrite('prod/binary.png', binary_mask)
	cv2.imwrite('prod/color.png', color)
	cv2.imwrite('prod/depth.png', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha = 0.3), cv2.COLORMAP_JET))
