
from engine import *

shape = [1, 480, 640, 3]

engine = build_engine('model_v3_onnx.onnx')
save_engine(engine, 'unet_engine.plan')
