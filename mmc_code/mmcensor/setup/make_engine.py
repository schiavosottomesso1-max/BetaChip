from ultralytics import YOLO
import time, os
import importlib

mmc_const = importlib.import_module('mmcensor.const')
supported_sizes = mmc_const.supported_sizes

t_1 = time.perf_counter()
for model_basename in mmc_const.get_model_profile_base_names():
    model_path = '../neuralnet_models/%s.pt'%model_basename
    if not os.path.isfile( model_path ):
        continue

    model = YOLO( model_path, task='detect' )
    for size in supported_sizes:
        model.export( format='engine',imgsz=size,half=True,dynamic=True)
        os.rename( '../neuralnet_models/%s.engine'%model_basename, '../neuralnet_models/%s-%d.engine'%(model_basename,size) )
t_2 = time.perf_counter()
print( t_2-t_1 )
