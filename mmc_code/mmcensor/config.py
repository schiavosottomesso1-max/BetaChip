# here you can define how large
# the images run through the net
# will be.
# [ 1280, 640 ] runs the images at
# two different scales, and works better
# for groups or images where the features
# are very small.
# Use [ 640 ] if you want to improve
# performance, at the cost of accuracy
# adding 2560 is a significant performance
# cost.  It helps identify small features
# in large censor areas (like thumbnails,
# or smaller images in a full-screen capture).
def get_net_sizes():
    net_sizes = [ 1280, 640, 2560 ]
    #net_sizes = [ 1280 ]
    #net_sizes = [ 1280, 640 ]
    #net_sizes = [ 640 ]
    return( net_sizes )

# time-safety is how long detected features are censored.
# for example, if MMCensor detects a face at 01:25.39 in a 
# video, and your time-safety is 0.15, MMCensor will put
# a censor feature in that location from 01:25.24 to
# 01:25.54.  Longer time-safety helps with temporary missed
# features, but may over-censor.  Longer time-safety also
# means more delay in realtime censoring.
# 
# I find 0.15s a reasonable setting.

def get_time_settings():
    time_settings = {
            'time-safety':  0.15
            }
    return time_settings

def get_perf_settings():
    perf_settings = {
            # Enable half precision where supported by the backend/device.
            'use-fp16': True,
            # Inference workers can be pinned to these CPU cores (empty disables pinning).
            'inference-affinity-cores': [],
            # Capture / GUI process can be pinned separately (empty disables pinning).
            'capture-gui-affinity-cores': [],
            # Overlay HUD controls.
            'hud-enabled': True,
            # Processing delay warning threshold in milliseconds.
            'sync-warning-ms': 150,
            # Timeout for one VRAM polling call via nvidia-smi.
            'vram-query-timeout-s': 0.8,
            }
    return perf_settings
