import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import segmentation_models as sm #вытянуть функцию обработки чтобы не надо делать импорт
import pyrealsense2 as rs
from pyrealsense2 import rs2_deproject_pixel_to_point as convert

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

def probability2explicit_bin_mask(border, *args):
	res = []
	for mask in args:
		_, mask = cv2.threshold(mask, border, 255, cv2.THRESH_BINARY)
		res.append(mask)
	return res

def printing_info(min_area, depth, params, *args):
	objects = ['beam', 'shaft', 'sleeve', 'square', 'kremlin']
	for img, obj in zip(args, objects):
		contours, _ = cv2.findContours(np.array(img, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
			(center_x, center_y), (length_x, length_y), angle = cv2.minAreaRect(cnt)
			center_x, center_y = int(center_x), int(center_y)
			center_z = depth[center_x][center_y]
			area = int(length_x * length_y)
			if area > min_area:
				x, y, z = convert(params, [center_x, center_y], center_z)
				print('-----------', obj, '-----------')
				print('CenterX -', x)
				print('CenterY -', y)
				print('CenterZ -', z)
				print('Angle -', angle)
				print('-------------------------------')

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
serialized = "modelv4.engine"
preproc = sm.get_preprocessing('efficientnetb3')
print('---STARTING LOAD ENGINE---')
engine = load_engine(trt_runtime, serialized)
print('---STARTING PIPELINE---')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

output = np.ones((480, 640))
print('---SKIPPING FEW FRAMES FOR CALIBRATION---')
for i in range(20):
	pipeline.wait_for_frames()

print('---STARTING LOOP---')
while True:
	frames = pipeline.wait_for_frames()
	aligned_frames = align.process(frames)
	aligned_depth_frame = aligned_frames.get_depth_frame()
	color_frame = aligned_frames.get_color_frame()

	if not aligned_depth_frame or not color_frame:
		continue
	key = cv2.waitKey(1)
	if key & 0xFF == ord('p'):
		print('p button pressed')
		d_in = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
		depth_image = np.asanyarray(aligned_depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		h_input, d_input, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)
		predict = do_inference(engine, preproc(color_image), h_input, d_input, h_output, d_output, stream, 1, 480, 640)

		predict = predict.squeeze()
		beam = predict[..., 0]*255
		shaft = predict[..., 1]*255
		sleeve = predict[..., 2]*255
		square = predict[..., 3]*255
		kremlin = predict[..., 4]*255

		beam, shaft, sleeve, square, kremlin = probability2explicit_bin_mask(225,
					      beam,
					      shaft,
					      sleeve,
					      square,
					      kremlin)
		printing_info(0, depth_image, d_in, beam, shaft, sleeve, square, kremlin)

		output = np.hstack((np.resize(color_image, (192, 256)),
					np.resize(beam, (192, 256)),
					np.resize(shaft, (192, 256)),
					np.resize(sleeve, (192, 256)),
					np.resize(square, (192, 256)),
					np.resize(kremlin, (192, 256))))

	elif key & 0xFF == ord('q'):
		print('q button pressed')
		cv2.destroyAllWindows()
		pipeline.stop()
		break

	cv2.imshow('Predict', output)
