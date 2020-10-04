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

def preprocessing_for_shaft(mask, box):
	black = []
	white = []
	for point in box:
		if mask[point[0], point[1]] == '0':
			black.append(point)
		else:
			white.append(point)


def get_info(obj_index, mask, depth, params, min_area):
	contours, _ = cv2.findContours(np.array(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = np.int64(cv2.boxPoints(rect))
		(center_x, center_y), (length_x, length_y), _ = rect
		center_x, center_y = int(center_x), int(center_y)
		edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
		edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
		if obj_index == 2:
			usedEdge, box = preprocessing_for_shaft(mask, box)
			#po novu box polychit coordinati x and y pereraschitanie
		else:
			usedEdge = edge1
			if cv2.norm(edge2) > cv2.norm(edge1):
				usedEdge = edge2
		#vertical
		if usedEdge[1] == 0:
			angle = 90*math.pi/180
		else:
			angle = 90*math.pi/180 - math.atan2(-usedEdge[1], usedEdge[0])

		center_z = depth[center_y][center_x]
		area = int(length_x * length_y)
		if area > min_area:
			x, y, z = convert(params, [center_x, center_y], center_z)
			return round(x,1), round(y,1), round(z, 1), round(angle, 3)
		else:
			print('Rejected', round(center_x,1), round(center_y,1), round(center_z,1), round(angle,3), round(area))
	print('Nothing to capture!')
	return


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
	_, sync, obj_index = input.split(':')
	return int(sync), int(obj_index)

def make_msg(sync, x, y, z, angle):
	return 'c:{}:{}:{}:{}:{}'.format(sync, x, y, z, angle)

def save_image(color, predict, binary_mask):
	cv2.imwrite('prod/engine_predict.png', predict)
	cv2.imwrite('prod/binaty.png', binary_mask)
	cv2.imwrite('prod/color.png', color)
