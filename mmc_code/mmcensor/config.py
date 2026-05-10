import mmcensor.const as mmc_const

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
    model_settings = get_model_settings()
    default_profile = model_settings['default-profile']
    net_sizes = list( model_settings['profile-net-sizes'][ default_profile ] )
    #net_sizes = [ 1280 ]
    #net_sizes = [ 1280, 640 ]
    #net_sizes = [ 640 ]
    return( net_sizes )

def get_model_settings():
    profile_net_sizes = {
            profile: list( mmc_const.model_profiles[ profile ][ 'default-net-sizes' ] )
            for profile in mmc_const.model_profiles
            }
    profile_net_sizes[ mmc_const.model_profile_auto ] = list(
        mmc_const.model_profiles[ mmc_const.model_profile_medium ][ 'default-net-sizes' ]
        )
    model_settings = {
            'default-profile': mmc_const.model_profile_auto,
            'profile-net-sizes': profile_net_sizes,
            'auto-profile-vram-thresholds-gb': {
                mmc_const.model_profile_medium: 8,
                mmc_const.model_profile_large: 12,
            },
            }
    return model_settings

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

            # --- CUDA Execution Provider (onnxruntime) optimisation ---
            # Used when mmcNNenv=cuda-onnx.  Irrelevant for PyTorch / TensorRT / OpenVINO paths.

            # cuDNN convolution algorithm selection strategy.
            # 'HEURISTIC' (fast startup, consistent) | 'EXHAUSTIVE' (slow, finds optimal) | 'DEFAULT'
            'cudnn-conv-algo-search': 'HEURISTIC',

            # Allow cuDNN to allocate the largest possible workspace so the fastest
            # (most memory-intensive) kernels are available from the first frame.
            'cudnn-conv-max-workspace': True,

            # Memory arena growth strategy. 'kNextPowerOfTwo' avoids many small
            # Windows VirtualAlloc calls that cause 100 ms+ jitter.
            'arena-extend-strategy': 'kNextPowerOfTwo',

            # Maximum GPU memory (bytes) the CUDA EP may allocate.
            # 8 GB covers an RTX 3080; lower this for cards with less VRAM.
            'gpu-mem-limit-bytes': 8 * 1024 * 1024 * 1024,

            # Number of dummy inference passes to run per resolution after model
            # load.  20 passes pre-allocates all CUDA kernels / memory arenas
            # and gives the GPU enough time to ramp from its idle P-state to
            # boost clocks, so the first real frame does not cause a spike and
            # performance is consistent across reboots.
            'warmup-iterations': 20,
            }
    return perf_settings
