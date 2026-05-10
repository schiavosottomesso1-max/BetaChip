from ultralytics import YOLO
import time, os
import importlib

mmc_const = importlib.import_module('mmcensor.const')
supported_sizes = mmc_const.supported_sizes

t_1 = time.perf_counter()
for model_basename in mmc_const.get_model_profile_base_names():
    model_path = f'../neuralnet_models/{model_basename}.pt'
    if not os.path.isfile( model_path ):
        continue

    model = YOLO( model_path, task='detect' )
    for size in supported_sizes:
        model.export( format='engine',imgsz=size,half=True,dynamic=True)
        os.rename( f'../neuralnet_models/{model_basename}.engine', f'../neuralnet_models/{model_basename}-{size}.engine' )
t_2 = time.perf_counter()
print( t_2-t_1 )
