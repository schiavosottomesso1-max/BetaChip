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
    model.export( format='openvino', half=False, dynamic=True )

t_2 = time.perf_counter()
print( t_2-t_1 )
