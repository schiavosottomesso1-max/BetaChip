from ultralytics import YOLO
import time, os
import importlib

mmc_const = importlib.import_module('mmcensor.const')

t_1 = time.perf_counter()
for model_basename in mmc_const.get_model_profile_base_names():
    model_path = f'../neuralnet_models/{model_basename}.pt'
    if not os.path.isfile( model_path ):
        continue
    model = YOLO( model_path, task='detect' )
    model.export( format='onnx', dynamic=True )

# someday, it would be nice to get half=True models for onnx for directml
# unfortunately, it seems these can only be generated with CUDA, which
# of course defeats the point, since we should be using CUDA, not directml
# it's possible that in the future I'll distribute onnx model files 
# generated on my machine.  In testing, the speed benefit is about 10%
# over just a single dynamic onnx file.
#supported_sizes = importlib.import_module( 'mmcensor.const' ).supported_sizes
#for size in supported_sizes:
    #model.export( format='onnx', imgsz=size, dynamic=False, half=True, simplify=True )
    #os.rename( '../neuralnet_models/%s.onnx'%model_basename, '../neuralnet_models/%s-%d.onnx'%(model_basename,size) )
t_2 = time.perf_counter()
print( t_2-t_1 )
