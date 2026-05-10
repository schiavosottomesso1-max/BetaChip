import numpy as np
import dxcam
import ctypes
import win32api, win32con, win32ui, win32gui
import time
from collections import deque
import mmcensor.const as mmc_const
import mmcensor.geo as geo
import random
from multiprocessing import shared_memory, Manager, Process
import cv2
import threading
import os
import sys
import copy
import importlib
import subprocess
import tempfile
import wave
from functools import partial
import mmcensor.config as mmc_config
import mmcensor.nn as nn
import statistics
from datetime import datetime

NS_PER_SECOND = 1_000_000_000
VRAM_QUERY_CACHE_NS = NS_PER_SECOND
MIN_FPS_INTERVAL_NS = 1_000_000
FPS_EMA_ALPHA = 0.15
_N_ROLLING = 120         # rolling buffer size for post-calibration re-calibration
_ROLLING_RECAL_EVERY = 20  # recalibrate delay every N new rolling samples
_AUTO_RESET_EMA_ALPHA = 0.12   # EMA smoothing for auto-reset SYNC check (absorbs single-frame GPU spikes)
_HUD_SYNC_EMA_ALPHA   = 0.20   # EMA smoothing for SYNC value shown in the HUD (cosmetic only)
_DEFAULT_AUTO_PROFILE_VRAM_GB = {
    mmc_const.model_profile_medium: 8,
    mmc_const.model_profile_large: 12,
}

def _disable_windows_quick_edit():
    # Only targets the console STD_INPUT_HANDLE (-10) to disable Quick-Edit
    # mode that freezes the process when the user clicks in the terminal.
    # This call does NOT affect Win32 window-message queues or GUI event
    # handling in any way.
    if os.name != 'nt':
        return
    try:
        kernel32 = ctypes.windll.kernel32
        h_stdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        if h_stdin == 0 or h_stdin == -1:
            return
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)) == 0:
            return
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        new_mode = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
        kernel32.SetConsoleMode(h_stdin, new_mode)
    except Exception:
        pass

def _set_process_affinity(cores):
    if not cores:
        return False
    try:
        normalized = sorted({int(c) for c in cores if int(c) >= 0})
    except Exception:
        return False
    if not normalized:
        return False
    try:
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, set(normalized))
            return True
    except Exception:
        pass
    if os.name == 'nt':
        try:
            mask = 0
            for core in normalized:
                mask |= (1 << core)
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            return bool(kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)))
        except Exception:
            return False
    return False

def _begin_high_precision_timer():
    """Set Windows multimedia timer resolution to 1 ms.

    The default Windows timer slice is ~15.6 ms.  Any call to cv2.waitKey(1)
    or time.sleep(<small value>) can silently stall for that long, causing
    irregular frame delivery and the latency spikes described in the issue.
    timeBeginPeriod(1) drops the system-wide timer resolution to 1 ms for
    this process.  Must be paired with _end_high_precision_timer() on exit.

    Returns True when the call succeeded, False otherwise (non-Windows or
    winmm unavailable).
    """
    if os.name != 'nt':
        return False
    try:
        result = ctypes.windll.winmm.timeBeginPeriod(1)
        return result == 0  # TIMERR_NOERROR == 0
    except Exception:
        return False

def _end_high_precision_timer():
    """Restore default timer resolution.  Pair with _begin_high_precision_timer."""
    if os.name != 'nt':
        return
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass

def _set_high_priority_class():
    """Raise the current process to HIGH_PRIORITY_CLASS on Windows.

    Prevents the Windows scheduler from de-prioritising the AI inference
    worker when the application window loses focus.  Returns True on
    success, False otherwise (non-Windows or permission denied).
    """
    if os.name != 'nt':
        return False
    try:
        HIGH_PRIORITY_CLASS = 0x00000080
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        return bool(kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS))
    except Exception:
        return False

def _check_hags_enabled():
    """Return True/False/None for HAGS state on Windows.

    Hardware-Accelerated GPU Scheduling (HAGS) significantly reduces
    GPU latency on RTX 30-series cards.  Reads the registry key
    HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\HwSchMode:
      2 → enabled, anything else → disabled.
    Returns None when the check cannot be performed (non-Windows, no access).
    """
    if os.name != 'nt':
        return None
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r'SYSTEM\CurrentControlSet\Control\GraphicsDrivers',
            0, winreg.KEY_READ,
        )
        value, _ = winreg.QueryValueEx(key, 'HwSchMode')
        winreg.CloseKey(key)
        return value == 2
    except OSError:
        return None
    except Exception:
        return None

def _create_optimized_cuda_ort_session(onnx_path, perf_settings):
    """Create an onnxruntime InferenceSession using the CUDA EP with tuned options.

    Provider options applied:
      - cudnn_conv_algo_search   → HEURISTIC (consistent fast startup)
      - cudnn_conv_use_max_workspace → 1 (use fastest cuDNN kernels)
      - arena_extend_strategy   → kNextPowerOfTwo (fewer VirtualAlloc calls)
      - gpu_mem_limit            → configurable (default 8 GB)

    Returns the InferenceSession on success, None on failure (onnxruntime
    unavailable, CUDA EP not present, file not found, etc.).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print('cuda-onnx: onnxruntime not installed; falling back to PyTorch backend')
        return None
    algo_search = str(perf_settings.get('cudnn-conv-algo-search', 'HEURISTIC'))
    max_workspace = '1' if perf_settings.get('cudnn-conv-max-workspace', True) else '0'
    arena_strategy = str(perf_settings.get('arena-extend-strategy', 'kNextPowerOfTwo'))
    gpu_mem_limit = int(perf_settings.get('gpu-mem-limit-bytes', 8 * 1024 * 1024 * 1024))
    cuda_ep_options = {
        'cudnn_conv_algo_search':      algo_search,
        'cudnn_conv_use_max_workspace': max_workspace,
        'arena_extend_strategy':       arena_strategy,
        'gpu_mem_limit':               str(gpu_mem_limit),
        'do_copy_in_default_stream':   '1',
    }
    providers = [('CUDAExecutionProvider', cuda_ep_options), 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        active = [ep for ep in session.get_providers() if 'CUDA' in ep]
        if not active:
            print('cuda-onnx: CUDA EP not active (driver/hardware issue?); session uses CPU fallback')
        else:
            print('cuda-onnx: CUDA EP active with %s cuDNN search, %s arena, %d MB GPU limit'
                  % (algo_search, arena_strategy, gpu_mem_limit // (1024 * 1024)))
        return session
    except Exception as exc:
        print('cuda-onnx: could not create optimized CUDA ORT session (%s); falling back' % exc)
        return None

def _replace_yolo_ort_session(model, new_session):
    """Swap the onnxruntime session inside an ultralytics YOLO ONNX predictor.

    The predictor is only initialised after the first predict() call.
    Returns True when the replacement succeeded.
    """
    try:
        if (model is not None
                and getattr(model, 'predictor', None) is not None
                and getattr(model.predictor, 'model', None) is not None
                and hasattr(model.predictor.model, 'session')):
            model.predictor.model.session = new_session
            return True
    except Exception:
        pass
    return False


def _check_windows_power_plan():
    """Return (guid, name) of the active Windows power plan, or None on failure.

    High Performance  = 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
    Ultimate Perf.    = e9a42b02-d5df-448d-aa00-03f14749eb61
    Balanced          = 381b4222-f694-41f0-9685-ff5bb260df2e
    Power Saver       = a1841308-3541-4fab-bc81-f71556f20b4a

    Prints a startup warning when the plan is not High or Ultimate Performance.
    Returns the (guid, name) tuple so callers can store it (e.g. for the HUD).
    """
    if os.name != 'nt':
        return None
    _HIGH_PERF_GUIDS = {
        '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c': 'High Performance',
        'e9a42b02-d5df-448d-aa00-03f14749eb61': 'Ultimate Performance',
    }
    try:
        result = subprocess.run(
            ['powercfg', '/getactivescheme'],
            capture_output=True, timeout=5,
        )
        line = result.stdout.decode(errors='replace').strip()
        # Output format: "Power Scheme GUID: <guid>  (<name>)"
        import re
        m = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\s+\(([^)]+)\)', line, re.IGNORECASE)
        if not m:
            return None
        guid = m.group(1).lower()
        name = m.group(2).strip()
        if guid not in _HIGH_PERF_GUIDS:
            print(
                'WARNING: Windows power plan is "%s". '
                'For consistent performance, switch to High Performance or Ultimate Performance: '
                'Settings → System → Power & Sleep → Additional power settings.' % name
            )
        else:
            print('Power plan: %s (optimal).' % name)
        return (guid, name)
    except Exception:
        return None

def _query_total_vram_mb( timeout_s ):
    try:
        result = subprocess.run(
            [ 'nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits' ],
            capture_output=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            return None
        lines = [ x.strip() for x in result.stdout.decode( errors='replace' ).splitlines() if x.strip() ]
        if not lines:
            return None
        return int( lines[0] )
    except Exception:
        return None

def _resolve_auto_model_profile( env, model_settings, perf_settings ):
    if env in ( 'openvino', 'pytorch-cpu', 'directml' ):
        return mmc_const.model_profile_small

    thresholds = dict( _DEFAULT_AUTO_PROFILE_VRAM_GB )
    thresholds.update( model_settings.get( 'auto-profile-vram-thresholds-gb', {} ) )
    timeout_s = float( perf_settings.get( 'vram-query-timeout-s', 0.8 ) )
    total_vram_mb = _query_total_vram_mb( timeout_s )
    if total_vram_mb is not None:
        if total_vram_mb >= int( thresholds[ mmc_const.model_profile_large ] * 1024 ):
            return mmc_const.model_profile_large
        if total_vram_mb >= int( thresholds[ mmc_const.model_profile_medium ] * 1024 ):
            return mmc_const.model_profile_medium
        return mmc_const.model_profile_small

    return mmc_const.model_profile_medium

def _iter_model_profiles( requested_profile ):
    queue = [ mmc_const.normalize_model_profile( requested_profile ) ]
    seen = set()
    while queue:
        profile = queue.pop( 0 )
        if profile in seen or profile == mmc_const.model_profile_auto:
            continue
        seen.add( profile )
        yield profile
        queue.extend( mmc_const.model_profiles.get( profile, {} ).get( 'fallback_profiles', [] ) )

def _get_model_asset_path( env, model_basename, size ):
    if env == 'openvino':
        return "../neuralnet_models/%s_openvino_model"%model_basename
    if env in ( 'directml', 'cuda-onnx' ):
        return "../neuralnet_models/%s.onnx"%model_basename
    if env == 'tensorrt':
        return "../neuralnet_models/%s-%d.engine"%(model_basename,size)
    return "../neuralnet_models/%s.pt"%model_basename

def _model_asset_exists( env, model_path ):
    if env == 'openvino':
        return os.path.isdir( model_path )
    return os.path.isfile( model_path )


def _check_gpu_boost_clocks(n_extra_warmup, models, warmup_img, use_fp16):
    """Check whether the GPU has reached its advertised boost clock after warmup.

    Queries ``nvidia-smi`` for the current graphics clock and the maximum
    boost clock.  If the GPU is still below 90 % of its boost clock (common
    when the system has just booted and the GPU P-state hasn't ramped up yet),
    up to ``n_extra_warmup`` additional inference passes are run per model to
    encourage the driver to move to P0, and then the clock is re-checked.

    This is the main cause of run-to-run performance variability: on a cold
    boot the GPU idles at a low P-state.  The standard warmup passes are
    enough to allocate CUDA memory arenas but not always enough to raise the
    clock all the way to boost.

    Returns True when the GPU reached ≥ 90 % of boost (or when nvidia-smi is
    unavailable), False when it could not ramp up within the extra passes.
    """
    def _read_clocks():
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=clocks.current.graphics,clocks.max.graphics',
                 '--format=csv,noheader,nounits'],
                capture_output=True, timeout=3,
            )
            if r.returncode != 0:
                return None, None
            parts = r.stdout.decode(errors='replace').strip().split(',')
            if len(parts) < 2:
                return None, None
            return int(parts[0].strip()), int(parts[1].strip())
        except Exception:
            return None, None

    current, max_clock = _read_clocks()
    if current is None or max_clock is None or max_clock == 0:
        # nvidia-smi not available — nothing to do
        return True

    ratio = current / max_clock
    if ratio >= 0.90:
        print('GPU clocks: %d / %d MHz (%.0f%% of boost) — OK.' % (current, max_clock, ratio * 100))
        return True

    print(
        'GPU clocks: %d / %d MHz (%.0f%% of boost) — running %d extra warmup passes to ramp up...'
        % (current, max_clock, ratio * 100, n_extra_warmup)
    )
    for size, model in models.items():
        for _ in range(n_extra_warmup):
            model.predict(warmup_img, imgsz=size, verbose=False, half=use_fp16)

    current, max_clock = _read_clocks()
    if current is None or max_clock is None or max_clock == 0:
        return True
    ratio = current / max_clock
    if ratio >= 0.90:
        print('GPU clocks after extra warmup: %d / %d MHz (%.0f%%) — OK.' % (current, max_clock, ratio * 100))
        return True
    print(
        'WARNING: GPU clocks still at %d / %d MHz (%.0f%%) after extra warmup. '
        'Performance may be lower than normal until the GPU fully warms up.'
        % (current, max_clock, ratio * 100)
    )
    return False


# this is in a function because the weird backdoor
# I do doesn't work inside a class
# all of this is horrible and dxcam should just expose
# the data I need.
# I apologize for all of this
def get_dxcams():
    cams = []
    outputs = dxcam.__factory.outputs
    for i in range(len(outputs)):
        for j in range(len( outputs[i] )):
            cam = dxcam.create( device_idx=i, output_idx=j, output_color='BGR' )
            cam_coords = cam._output.desc.DesktopCoordinates
            cams.append( { 'cam':cam, 'cam_coords': [ cam_coords.left, cam_coords.top, cam_coords.right, cam_coords.bottom ], 'desc':str(outputs[i][j]) } )
    return( cams )

class mmc_screencap:

    def initialize( self ):
        self._closed = False
        # determine screen geometry
        self.populate_dxcams()
        self.visible_bounds = self.get_visible_bounds()
        self.img_shape = ( self.visible_bounds[3] - self.visible_bounds[1], self.visible_bounds[2] - self.visible_bounds[0], 3 )

        # set up shared memory
        self.img_shm_name    = 'img_shm_name_%d'%random.randint(0,10000000)     # the actual image data
        self.img_coords_name = 'img_coords_name_%d'%random.randint(0,10000000)  # the coordinates of each window and its hwnd
        self.img_ref_name    = 'img_time_name_%d'%random.randint(0,100000000)   # the time of the snap, and the number of windows snapped

        self.img_shm = shared_memory.SharedMemory( name=self.img_shm_name, create=True, size = self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
        self.img_coords_shm = shared_memory.SharedMemory( name = self.img_coords_name, create=True, size = 10000 )
        self.img_ref_shm = shared_memory.SharedMemory( name = self.img_ref_name, create=True, size = 1000 )

        self.img_shared = np.ndarray( self.img_shape, dtype = np.uint8, buffer = self.img_shm.buf        )
        self.img_coords = np.ndarray( (100, 5 ),      dtype = np.int64, buffer = self.img_coords_shm.buf )
        self.img_ref    = np.ndarray( ( 2, ),         dtype = np.int64, buffer = self.img_ref_shm.buf    )

        # warm up cams
        for cam in self.cams:
            _dummy = cam['cam'].grab()

    def shutdown( self ):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        for shm in (getattr(self, 'img_shm', None), getattr(self, 'img_coords_shm', None), getattr(self, 'img_ref_shm', None)):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

    def populate_dxcams( self ):
        self.cams = get_dxcams()
        
    def get_visible_bounds( self ):
        l = t = r = b = None
        for cam in self.cams:
            l = cam['cam_coords'][0] if l is None else min( l, cam['cam_coords'][0] )
            t = cam['cam_coords'][1] if t is None else min( l, cam['cam_coords'][1] )
            r = cam['cam_coords'][2] if r is None else max( r, cam['cam_coords'][2] )
            b = cam['cam_coords'][3] if b is None else max( b, cam['cam_coords'][3] )

        return( [ l, t, r, b ] )

    def get_hwnds( self ):
        l = []

        for i in range(len(self.cams)):
            l.append( [ -1 * i, self.cams[i]['desc'] ] )

        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText( hwnd )
                if len(title):
                    l.append( [ hwnd, win32gui.GetWindowText(hwnd) ] )
        win32gui.EnumWindows(winEnumHandler, l)

        return( l )

    def get_hwnd_coords_unintersected( self, hwnd ):
        if hwnd <= 0:
            return self.cams[ -1 * hwnd ]['cam_coords']
        else:
            rect = ctypes.wintypes.RECT()
            DWMWA_EXTENDED_FRAME_BOUNDS = 9 # magic windows number 
            ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(hwnd),
              ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
              ctypes.byref(rect),
              ctypes.sizeof(rect)
              )

            window_coords = [ rect.left, rect.top, rect.right, rect.bottom ]
            return( window_coords )

    def get_hwnd_coords( self, hwnd ):
            window_coords = self.get_hwnd_coords_unintersected( hwnd )
            visible_coords = geo.intersection_box( window_coords, self.visible_bounds )

            return visible_coords

    def snap_hwnds( self, hwnds ):
        tasks = []

        hwnd_coords = {}
        cam_tasks = [ None for cam in self.cams ]

        for hwnd in hwnds:
            this_hwnd_coords = self.get_hwnd_coords( hwnd )
            if this_hwnd_coords is not None:
                for i in range(len(self.cams)):
                    cam = self.cams[i]
                    int_xyxy = geo.intersection_box( cam['cam_coords'], this_hwnd_coords )
                    if int_xyxy is not None:
                        hwnd_coords[hwnd] = this_hwnd_coords # we found at least one part in a visible area
                        grab_coords = [
                                int_xyxy[0]-cam['cam_coords'][0],
                                int_xyxy[1]-cam['cam_coords'][1],
                                int_xyxy[2]-cam['cam_coords'][0],
                                int_xyxy[3]-cam['cam_coords'][1],
                                ]
                        if cam_tasks[i] is not None:
                            cam_tasks[i] = geo.union_box( cam_tasks[i], grab_coords )
                        else:
                            cam_tasks[i] = grab_coords

        #self.profiler.mark( 'get_coords' )

        snap_time = time.perf_counter_ns()

        for i in range(len( cam_tasks )):
            if cam_tasks[i] is not None:
                cam = self.cams[i]
                subimg = cam['cam'].grab( region=tuple(cam_tasks[i] ) )
                if subimg is not None:
                    subimg_xyxy = [
                            cam_tasks[i][0]+cam['cam_coords'][0]-self.visible_bounds[0],
                            cam_tasks[i][1]+cam['cam_coords'][1]-self.visible_bounds[1],
                            cam_tasks[i][2]+cam['cam_coords'][0]-self.visible_bounds[0],
                            cam_tasks[i][3]+cam['cam_coords'][1]-self.visible_bounds[1],
                            ]
                    self.img_shared[subimg_xyxy[1]:subimg_xyxy[3],subimg_xyxy[0]:subimg_xyxy[2]]=subimg

        i = 0
        for hwnd in hwnd_coords:
            self.img_coords[i] = ( 
                    hwnd_coords[hwnd][0]-self.visible_bounds[0], 
                    hwnd_coords[hwnd][1]-self.visible_bounds[1], 
                    hwnd_coords[hwnd][2]-self.visible_bounds[0], 
                    hwnd_coords[hwnd][3]-self.visible_bounds[1], 
                    hwnd
                    )
            i = i + 1
            #self.profiler.mark('shared_coords')

        self.img_ref[0] = snap_time
        self.img_ref[1] = len(hwnd_coords)

        #self.profiler.mark('set_img_ref')

# this class is created on the main process
# see mmc_detect_remote_func and 
# mmc_detect_loop_remote for the class
# that is created on the remote process
class mmc_detect_loop_async:

    def initialize( self, img_shm_name, img_coords_name, img_ref_name, img_shape, sizes, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name ):
        self._manager = Manager()
        self.sizes = self._manager.list()
        self.state = self._manager.list()
        self.state.append( 0 )  # ready
        self.state.append( 0 )  # stop
        self.sizes.extend( sizes )
        self.P1 = Process( target = mmc_detect_loop_remote, args = ( self.sizes, self.state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name ) )

    def start( self ):
        self.P1.start()

        while( self.state[0] == 0 and self.P1.is_alive() ):
            print( "waiting for neural net to be ready..." )
            time.sleep( 2 )

    def shutdown( self ):
        if self.P1.is_alive():
            try:
                self.state[1] = 1
            except Exception:
                pass
            self.P1.join(timeout=15.0)
            if self.P1.is_alive():
                self.P1.terminate()
                self.P1.join()
        # Shut down the Manager process that backs self.sizes / self.state.
        # Without this, each reset leaks one Manager OS process, causing
        # cumulative slowdown after several net switches.
        try:
            self._manager.shutdown()
        except Exception:
            pass

def mmc_detect_loop_remote( sizes, state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name):
    detector = mmc_detect_loop_class()
    detector.initialize( sizes, state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name )
    detector.go_detect()

class mmc_detect_loop_class:

    def initialize( self, sizes, state, img_shm_name, img_coords_name, img_ref_name, img_shape, boxes_shm_name, box_hwnds_shm_name, box_info_shm_name ):
        from ultralytics import YOLO
        self.sizes = sizes
        self.state = state
        self.env = os.getenv( 'mmcNNenv' )
        self.model_settings = mmc_config.get_model_settings()
        self.perf_settings = mmc_config.get_perf_settings()
        requested_profile = os.getenv( 'mmcModelProfile', self.model_settings.get( 'default-profile', mmc_const.model_profile_medium ) )
        self.requested_model_profile = mmc_const.normalize_model_profile( requested_profile, mmc_const.model_profile_medium )
        self.active_model_profile = self.requested_model_profile
        if self.active_model_profile == mmc_const.model_profile_auto:
            self.active_model_profile = _resolve_auto_model_profile( self.env, self.model_settings, self.perf_settings )
        self.known_classes = mmc_const.get_detection_classes()
        self.use_fp16 = bool(self.perf_settings.get('use-fp16', True))
        _set_process_affinity(self.perf_settings.get('inference-affinity-cores', []))
        if not _set_high_priority_class():
            print( 'inference worker: could not raise process priority (non-Windows or permission denied)' )
        self.last_t = 0
        self.fps_limit = 300
        self.last_detect_finish = 0

        # Track ONNX paths for 'cuda-onnx' env so we can swap the session after warmup.
        self._cuda_onnx_paths = {}
        self.model_class_indices = {}

        self.models = {}
        print( 'model profile requested: %s | active: %s'%( self.requested_model_profile, self.active_model_profile ) )
        for size in mmc_const.supported_sizes:
            model, loaded_profile, model_path = self._load_model_for_size( YOLO, size )

            if model is not None:
                self.models[size] = model
                self.model_class_indices[size] = mmc_const.get_detection_class_index_map(
                    getattr( model, 'names', {} ),
                    self.known_classes
                )
                if self.env == 'cuda-onnx':
                    self._cuda_onnx_paths[size] = model_path
                if loaded_profile != self.active_model_profile:
                    print( 'size %d: using %s fallback asset %s'%( size, loaded_profile, model_path ) )
                else:
                    print( 'size %d: using %s asset %s'%( size, loaded_profile, model_path ) )

        # For the PyTorch backend, disable cuDNN auto-tuning benchmarking.
        # With benchmark=True (PyTorch default) cuDNN re-runs algorithm search
        # on the first inference at each input shape, which adds a variable
        # cold-start delay that differs between reboots depending on whether
        # the cuDNN benchmark cache is warm.  With benchmark=False cuDNN uses
        # its heuristic to pick an algorithm consistently every run.
        if self.env not in ('tensorrt', 'openvino', 'directml', 'cuda-onnx'):
            try:
                import torch
                torch.backends.cudnn.benchmark = False
            except ImportError:
                pass

        if not len( self.models ):
            print( 'WARNING: no model assets were found for profile %s.'%self.active_model_profile )

        # --- GPU engine prime -----------------------------------------------
        # Run multiple dummy inferences per resolution so that CUDA allocates
        # all necessary kernels and memory arenas before the real-time loop
        # starts.  The first real frame would otherwise cause a large spike.
        n_warmup = max(1, int(self.perf_settings.get('warmup-iterations', 20)))
        warmup_img = np.full( ( 2560, 2560, 3 ), 127, dtype=np.uint8 )
        for size in self.models:
            model = self.get_model_for_size( size )
            if model is not None:
                print( 'priming GPU engine: resolution %d (%d iterations)...' % (size, n_warmup) )
                for _ in range( n_warmup ):
                    model.predict(warmup_img, imgsz=size, verbose=False)

        # --- cuda-onnx: swap session with optimized CUDA EP session ----------
        # At this point the ultralytics predictor has been fully initialised
        # (the warmup calls above trigger predictor creation).  Replace the
        # internal onnxruntime session with one configured for deterministic
        # cuDNN algorithm selection and large memory arenas.
        if self.env == 'cuda-onnx':
            for size, model in self.models.items():
                onnx_path = self._cuda_onnx_paths.get( size )
                if onnx_path is None:
                    continue
                new_session = _create_optimized_cuda_ort_session( onnx_path, self.perf_settings )
                if new_session is not None:
                    replaced = _replace_yolo_ort_session( model, new_session )
                    if replaced:
                        # Re-prime with the new session so its memory arenas are warmed up.
                        print( 'cuda-onnx: re-priming with optimized session at resolution %d...' % size )
                        for _ in range( n_warmup ):
                            model.predict(warmup_img, imgsz=size, verbose=False)
                    else:
                        print( 'cuda-onnx: session replacement unsupported in this ultralytics version; '
                               'optimized CUDA EP options may not be active' )
        # --------------------------------------------------------------------

        # --- GPU boost-clock check ------------------------------------------
        # After all warmup passes, verify the GPU has reached its advertised
        # boost clock.  On a cold boot the GPU idles at a low P-state; the
        # standard warmup allocates memory arenas but may not be long enough to
        # fully ramp up the clock.  If the GPU is still below 90% of boost,
        # run up to n_warmup extra passes to encourage P0 and re-check.
        _check_gpu_boost_clocks( n_warmup, self.models, warmup_img, self.use_fp16 )
        # --------------------------------------------------------------------

        self.img_shape = img_shape

        self.img_shm = shared_memory.SharedMemory( name=img_shm_name )
        self.img_coords_shm = shared_memory.SharedMemory( name = img_coords_name )
        self.img_ref_shm = shared_memory.SharedMemory( name = img_ref_name )

        self.img_shared = np.ndarray( self.img_shape, dtype = np.uint8, buffer = self.img_shm.buf        )
        self.img_coords = np.ndarray( (100, 5 ),      dtype = np.int64, buffer = self.img_coords_shm.buf )
        self.img_ref    = np.ndarray( ( 2, ),         dtype = np.int64, buffer = self.img_ref_shm.buf    )

        self.boxes_shm = shared_memory.SharedMemory( name=boxes_shm_name )
        self.box_hwnds_shm = shared_memory.SharedMemory( name=box_hwnds_shm_name )
        self.box_info_shm = shared_memory.SharedMemory( name = box_info_shm_name )

        self.boxes_np = np.ndarray( (20,500,8), dtype = np.int64, buffer = self.boxes_shm.buf        )
        self.box_hwnds_np = np.ndarray( (50,4), dtype = np.int64, buffer = self.box_hwnds_shm.buf        )
        self.box_info_np = np.ndarray( (8,),      dtype = np.int64, buffer = self.box_info_shm.buf )

        self.state[0] = 1

    def get_model_for_size( self, size ):
        return self.models.get( size )

    def _load_model_for_size( self, YOLO, size ):
        for profile in _iter_model_profiles( self.active_model_profile ):
            basename = mmc_const.model_profiles[ profile ][ 'basename' ]
            model_path = _get_model_asset_path( self.env, basename, size )
            if not _model_asset_exists( self.env, model_path ):
                continue
            try:
                return YOLO( model_path, task='detect' ), profile, model_path
            except Exception as exc:
                print( 'could not load %s for size %d: %s'%( model_path, size, exc ) )
        return None, None, None

    def _write_box_info(self, sstime, num_hwnds, sizes_key, write_time_ns, inference_latency_ns):
        self.box_info_np[0] = sstime
        self.box_info_np[1] = num_hwnds
        self.box_info_np[2] = sizes_key
        self.box_info_np[3] = write_time_ns
        self.box_info_np[4] = inference_latency_ns
        self.box_info_np[5] = write_time_ns - sstime
        self.box_info_np[6] = sstime
        self.box_info_np[7] = 0  # reserved for future telemetry extension

    def go_detect( self ):
        n = 0
        t_start = time.perf_counter()
        self.profiler = profiler()
        self.profiler.initialize( 5, 0.0001 )
        while( True ):
            if len(self.state) > 1 and self.state[1]:
                break
            self.profiler.loop()
            self.profiler.mark( "start" )
            sstime = self.img_ref[0]
            self.profiler.mark( "got_time" )

            if sstime > self.last_t:
                detect_started_ns = time.perf_counter_ns()
                _sizes = list(self.sizes)  # snapshot Manager proxy once per cycle to avoid repeated IPC calls
                num_hwnds = self.img_ref[1]
                self.profiler.mark( "got_hwnds" )

                if num_hwnds:
                    batch = []
                    hwnds = []
                    outs = {}
                    for i in range( num_hwnds ):
                        hwnds.append( self.img_coords[i][4] )
                        self.profiler.mark( "got_hwnd" )
                        batch.append( np.ascontiguousarray(self.img_shared[self.img_coords[i][1]:self.img_coords[i][3],self.img_coords[i][0]:self.img_coords[i][2]]) )
                        self.profiler.mark( "got_batch" )
                        outs[ self.img_coords[i][4] ] = {}

                    self.profiler.mark( "presizes" )
                    for size in _sizes:
                        model_for_size = self.get_model_for_size( size )
                        if model_for_size is None:
                            continue
                        if self.env == 'tensorrt': # tensorrt needs to have engine files designed for batching
                            output = [ model_for_size.predict( x, imgsz=size, verbose=False, half=self.use_fp16 )[0] for x in batch ]
                        else:
                            output = model_for_size.predict( batch, imgsz=size, verbose=False, half=self.use_fp16 )
                        #if random.randint(0,100) <2:
                            #raise Exception( "test throw" )
                        self.profiler.mark( "predict" )
                        for i in range( num_hwnds ):
                            outs[ hwnds[ i ] ][size] = output[i].boxes.cpu().numpy()
                            self.profiler.mark( "append_out" )

                    self.profiler.mark( "done_predict" )
                    write_time_ns = time.perf_counter_ns()
                    inference_latency_ns = write_time_ns - detect_started_ns
                    i=0
                    for hwnd in outs:
                        j=0
                        for size in outs[hwnd]:
                            for box in outs[hwnd][size]:
                                raw_class_index = int( box.cls[0].item() )
                                mapped_class_index = self.model_class_indices.get( size, {} ).get( raw_class_index )
                                if mapped_class_index is None:
                                    # Ignore classes we do not know how to map back into the
                                    # legacy BetaChip class list instead of corrupting indices.
                                    continue
                                self.boxes_np[i][j] = (sstime,mapped_class_index,box.xyxy[0][0].item(),box.xyxy[0][1].item(),box.xyxy[0][2].item(),box.xyxy[0][3].item(),1,size)
                                j = j+1
                        self.profiler.mark( "copied_box" )
                        self.box_hwnds_np[i]=(hwnd,j,self.img_coords[i][2]-self.img_coords[i][0],self.img_coords[i][3]-self.img_coords[i][1])
                        self.profiler.mark( "wrote_hwnds" )
                        i=i+1
                    self._write_box_info(
                        sstime=sstime,
                        num_hwnds=i,
                        sizes_key=nn.sizes_to_key(_sizes),
                        write_time_ns=write_time_ns,
                        inference_latency_ns=inference_latency_ns
                    )

                    self.profiler.mark( "done_outs" )
                else:
                    write_time_ns = time.perf_counter_ns()
                    self._write_box_info(
                        sstime=sstime,
                        num_hwnds=0,
                        sizes_key=nn.sizes_to_key(_sizes),
                        write_time_ns=write_time_ns,
                        inference_latency_ns=write_time_ns - detect_started_ns
                    )

                self.last_t = sstime

            fps_limit_sleep = 1/self.fps_limit - ( time.perf_counter() - self.last_detect_finish )
            if fps_limit_sleep > 0:
                time.sleep( fps_limit_sleep )
            self.profiler.mark( "done_sleep" )

            self.last_detect_finish = time.perf_counter()

            n = n+1
            if n == 100:
                t_end = time.perf_counter()
                print( "100 detections in %.2f seconds, or %.1ffps"%(t_end - t_start, 100 / ( t_end - t_start ) ) )
                t_start = t_end
                n = 0
            self.profiler.mark( "done" )

        for shm in (self.img_shm, self.img_coords_shm, self.img_ref_shm, self.boxes_shm, self.box_hwnds_shm, self.box_info_shm):
            try:
                shm.close()
            except Exception:
                pass

class profiler:
    times = {}
    n = 0
    last = 0

    def initialize( self, report_freq, time_threshold ):
        self.n = 0
        self.times = {}
        self.report_start = time.perf_counter()
        self.report_freq = report_freq
        self.last = time.perf_counter()
        self.loop_start = time.perf_counter()
        self.time_threshold = time_threshold

    def mark( self, label ):
        new = time.perf_counter()
        elapsed = new - self.last
        self.last = new

        if label in self.times:
            if label in self.seen_in_loop:
                self.times[label][-1] = self.times[label][-1] + elapsed
            else:
                self.times[label].append( elapsed )
                self.seen_in_loop[label]=None
        else:
            self.times[label] = [ elapsed ]

    def loop( self ):
        self.n = self.n + 1
        if time.perf_counter() - self.report_start > self.report_freq:
            label_sum = {}
            for label in self.times:
                label_sum[ label ] = sum( self.times[label] )
                if max(self.times[label]) > self.time_threshold:
                    print( label.ljust(15), 'avg %.1fms max %.1fms std %.1fms count %d/%d'%(
                        label_sum[label]/self.n*1000,
                        max(self.times[label])*1000,
                        statistics.stdev( self.times[label] )*1000 if len(self.times[label])>1 else 0,
                        len(self.times[label]),
                        self.n
                        ), '%.1f fps'%(self.n/(time.perf_counter()-self.report_start),) if label=='__loop__' else '' )

            self.times = {}
            self.report_start = time.perf_counter()
            self.loop_start = time.perf_counter()
            self.n = 0

        else:
            if '__loop__' in self.times:
                self.times['__loop__'].append( time.perf_counter() - self.loop_start )
            else:
                self.times['__loop__'] = [ time.perf_counter() - self.loop_start ]
            self.loop_start = time.perf_counter()

        self.seen_in_loop = {}

class mmc_realtime:

    def initialize( self ):
        _disable_windows_quick_edit()
        self.model_settings = mmc_config.get_model_settings()
        self.model_profile = mmc_const.normalize_model_profile(
            os.getenv( 'mmcModelProfile', self.model_settings.get( 'default-profile', mmc_const.model_profile_medium ) ),
            mmc_const.model_profile_medium
        )
        os.environ['mmcModelProfile'] = self.model_profile
        self.perf_settings = mmc_config.get_perf_settings()
        _set_process_affinity(self.perf_settings.get('capture-gui-affinity-cores', []))
        self._hi_res_timer_active = _begin_high_precision_timer()
        self.reset_count = 0

        # Warn when Hardware-Accelerated GPU Scheduling (HAGS) is disabled.
        # HAGS significantly reduces GPU command-queue latency on RTX 30-series
        # cards and should be enabled in Windows Display settings.
        hags = _check_hags_enabled()
        if hags is False:
            print( 'WARNING: Hardware-Accelerated GPU Scheduling (HAGS) is DISABLED. '
                   'Enable it in Windows Settings → Display → Graphics → Default GPU '
                   'settings for lower latency on RTX 30-series GPUs.' )
        elif hags is True:
            print( 'HAGS: enabled.' )

        # Check Windows power plan.  Balanced / Power Saver plans throttle CPU
        # frequency and can reduce GPU boost clocks, causing variable performance
        # between boots depending on which plan Windows restores at startup.
        self._power_plan = _check_windows_power_plan()   # (guid, name) or None

        self.hud_enabled = bool(self.perf_settings.get('hud-enabled', True))
        self.sync_warning_ms = float(self.perf_settings.get('sync-warning-ms', 250))
        self.latest_inference_latency_ns = 0
        self.latest_processing_delay_ns = 0
        self.latest_detection_write_ns = 0
        self.latest_detection_snap_ns = 0
        self._ema_processing_delay_ns = 0.0   # smoothed delay used only for auto-reset logic
        self._hud_sync_ema_ms = 0.0           # smoothed SYNC value shown in the HUD
        self.display_fps = 0.0
        self._last_display_tick_ns = 0
        self._cached_vram_text = 'VRAM: n/a'
        self._last_vram_query_ns = 0
        self._vram_query_timeout_s = float(self.perf_settings.get('vram-query-timeout-s', 0.8))
        self._future_frame_warned = False

        self.sc = mmc_screencap()
        self.sc.initialize()
        self.ready = False
        self.running = False
        self.hwnds = []
        self.cv_title_template = 'mmcensor-%d-%%d'%random.randint(0,100000 )
        self.profiler = profiler()
        self.profiler.initialize( 5, 0.0001 )
        self.sc.profiler = self.profiler
        self.threaded_screenshot = True
        self.decorators = []
        self.open_windows = {}

        self.size_detection_timings = {}
        self.size_delays = {}
        self._rolling_timings = {}          # delay_key -> deque(maxlen=_N_ROLLING)
        self._rolling_recal_counters = {}   # delay_key -> int (samples since last recal)
        self.auto_reset_sync_s = float(self.perf_settings.get('auto-reset-sync-s', 30))
        self._sync_warn_since_ns = None
        self.boxes_shm_name    = 'boxes_shm_name_%d'%random.randint(0,10000000)     # [ [ t, cls, x1, y1, x2, y2, prob, size ] ]
        self.box_hwnds_shm_name    = 'box_hwnds_shm_name_%d'%random.randint(0,10000000)     # [ [ hwnd, numboxes ] ]
        self.box_info_shm_name = 'box_info_shm_name_%d'%random.randint(0,10000000)  # [ t, numhwnds, t ]

        self.boxes_shm = shared_memory.SharedMemory( name=self.boxes_shm_name, create=True, size = 8 * 20 * 9 * 500 ) # 8 bytes times twenty hwnds times nine fields times 500 boxes
        self.box_hwnds_shm = shared_memory.SharedMemory( name=self.box_hwnds_shm_name, create=True, size = 10000 ) 
        self.box_info_shm = shared_memory.SharedMemory( name = self.box_info_shm_name, create=True, size = 10000 )

        self.boxes_np = np.ndarray( (20,500,8), dtype = np.int64, buffer = self.boxes_shm.buf        )
        self.box_hwnds_np = np.ndarray( ( 50, 4), dtype = np.int64, buffer = self.box_hwnds_shm.buf        )
        self.box_info_np = np.ndarray( (8,),      dtype = np.int64, buffer = self.box_info_shm.buf )
        self.box_info_np[:] = 0
        self.box_hwnds_np[:][:]=0

        self.boxes = np.ndarray( (50,20000,8), dtype=np.int64 )
        self.boxes_hwnd_index = {}
        self.hwnd_times = {} # hwnd: [ t, first_index, last_index ]
        self.last_detection_found = 0
        self.last_detection_delay_key = None

        self.detector_async = mmc_detect_loop_async()
        self.detector_async.initialize( self.sc.img_shm_name, self.sc.img_coords_name, self.sc.img_ref_name, self.sc.img_shape, [], self.boxes_shm_name, self.box_hwnds_shm_name, self.box_info_shm_name )
        self.sizes = self.detector_async.sizes
        self.to_show = {}

        self.on_gray_callback = None
        self.off_gray_callback = None
        self.gray_state = False

        self.recording = False
        self.video_writers = {}      # hwnd -> cv2.VideoWriter
        self.video_writer_dims = {}  # hwnd -> (w, h); VideoWriter.get(CAP_PROP_FRAME_WIDTH/HEIGHT) returns 0, so stored separately
        self.recording_path = None
        self._rec_hwnd_paths = {}    # hwnd -> file path (set at writer-creation time)
        self._rec_next_idx = 0       # stable counter for multi-window filename suffixes
        self._audio_thread   = None  # background thread capturing WASAPI loopback audio
        self._audio_tmp_path = None  # temp WAV file path written by _audio_thread

    def take_screenshot( self ):
        for hwnd in self.to_show:
            if self.to_show[hwnd] is not None:
                now = datetime.today().strftime('%Y%m%d%H%M%S%f')
                cv2.imwrite( '../screenshots/%s.jpg'%now, self.to_show[hwnd] )

    def start_recording( self, path ):
        """Begin recording censored output.  Writers are created lazily on the first frame."""
        self.recording_path    = path
        self.video_writers     = {}
        self.video_writer_dims = {}
        self._rec_hwnd_paths   = {}
        self._rec_next_idx     = 0      # stable counter for multi-window filename suffixes
        self._audio_tmp_path   = None
        self._audio_thread     = None
        self._start_audio_capture()
        self.recording = True

    def _start_audio_capture( self ):
        """Start a background thread that records WASAPI loopback (system audio) to a temp WAV."""
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            print( 'WARNING: pyaudiowpatch not installed; recording without audio' )
            return
        try:
            tmp_fd, tmp_path = tempfile.mkstemp( suffix='.wav' )
            os.close( tmp_fd )
            self._audio_tmp_path = tmp_path
            self._audio_thread   = threading.Thread(
                target=self._audio_record_loop,
                kwargs={ 'pyaudio_mod': pyaudio, 'tmp_path': tmp_path },
                daemon=True
            )
            self._audio_thread.start()
        except Exception as exc:
            print( 'WARNING: could not start audio capture: %s' % exc )
            self._audio_tmp_path = None
            self._audio_thread   = None

    def _audio_record_loop( self, pyaudio_mod, tmp_path ):
        """Background thread: capture WASAPI loopback audio and write it to a WAV file."""
        try:
            with pyaudio_mod.PyAudio() as pa:
                wasapi_info = pa.get_host_api_info_by_type( pyaudio_mod.paWASAPI )
                speakers    = pa.get_device_info_by_index( wasapi_info['defaultOutputDevice'] )
                # Prefer the dedicated loopback device if the default output is not already one
                if not speakers.get( 'isLoopbackDevice', False ):
                    for loopback in pa.get_loopback_device_info_generator():
                        if speakers['name'] in loopback['name']:
                            speakers = loopback
                            break
                rate     = int( speakers['defaultSampleRate'] )
                channels = speakers['maxInputChannels'] or 2   # loopback devices set maxInputChannels; fall back to stereo
                chunk    = 512
                stream   = pa.open(
                    format            = pyaudio_mod.paInt16,
                    channels          = channels,
                    rate              = rate,
                    frames_per_buffer = chunk,
                    input             = True,
                    input_device_index= speakers['index'],
                )
                with wave.open( tmp_path, 'wb' ) as wf:
                    wf.setnchannels( channels )
                    wf.setsampwidth( pa.get_sample_size( pyaudio_mod.paInt16 ) )
                    wf.setframerate( rate )
                    while self.recording:
                        data = stream.read( chunk, exception_on_overflow=False )
                        wf.writeframes( data )
                stream.stop_stream()
                stream.close()
        except Exception as exc:
            print( 'WARNING: audio recording failed: %s' % exc )

    # Path to the bundled ffmpeg downloaded by check-prereqs.bat during setup.
    # Falls back to whatever 'ffmpeg' is on the system PATH.
    _FFMPEG_BUNDLED = os.path.normpath(
        os.path.join( os.path.dirname( os.path.abspath(__file__) ),
                      '..', '..', '..', 'tools', 'ffmpeg.exe' ) )

    def _mux_audio_into_video( self, video_path, audio_path ):
        """Use ffmpeg to mux audio_path into video_path, replacing the file in-place."""
        ffmpeg = self._FFMPEG_BUNDLED if os.path.isfile( self._FFMPEG_BUNDLED ) else 'ffmpeg'
        base, ext = os.path.splitext( video_path )
        tmp_out   = base + '_withaudio' + ext
        try:
            result = subprocess.run(
                [ ffmpeg, '-y',
                  '-i', video_path,
                  '-i', audio_path,
                  '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                  tmp_out ],
                capture_output=True, timeout=120
            )
            if result.returncode == 0:
                os.replace( tmp_out, video_path )
            else:
                print( 'WARNING: ffmpeg mux failed (code %d); video saved without audio\n%s'
                       % ( result.returncode,
                           result.stderr.decode( errors='replace' ) ) )
                try: os.unlink( tmp_out )
                except OSError: pass
        except FileNotFoundError:
            print( 'WARNING: ffmpeg not found; video saved without audio' )
        except subprocess.TimeoutExpired:
            print( 'WARNING: ffmpeg mux timed out; video saved without audio' )
            try: os.unlink( tmp_out )
            except OSError: pass

    def stop_recording( self ):
        """Stop recording, mux captured audio into each video file, release resources."""
        self.recording = False
        # Wait for the audio thread to finish flushing its last chunk to the WAV
        if self._audio_thread is not None:
            self._audio_thread.join( timeout=5.0 )
            if self._audio_thread.is_alive():
                print( 'WARNING: audio capture thread did not finish cleanly within timeout' )
        # Flush and close all video writers
        for wr in self.video_writers.values():
            wr.release()
        # Mux audio into every recorded video file
        if self._audio_tmp_path and os.path.exists( self._audio_tmp_path ):
            for video_path in self._rec_hwnd_paths.values():
                if os.path.exists( video_path ):
                    self._mux_audio_into_video( video_path, self._audio_tmp_path )
            try: os.unlink( self._audio_tmp_path )
            except OSError: pass
        self.video_writers     = {}
        self.video_writer_dims = {}
        self._rec_hwnd_paths   = {}
        self._rec_next_idx     = 0
        self._audio_tmp_path   = None
        self._audio_thread     = None

    def _query_vram_usage( self ):
        now_ns = time.perf_counter_ns()
        if now_ns - self._last_vram_query_ns < VRAM_QUERY_CACHE_NS:
            return self._cached_vram_text
        self._last_vram_query_ns = now_ns
        try:
            result = subprocess.run(
                [ 'nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits' ],
                capture_output=True,
                timeout=self._vram_query_timeout_s
            )
            if result.returncode != 0:
                self._cached_vram_text = 'VRAM: n/a (nvidia-smi error)'
                return self._cached_vram_text
            lines = result.stdout.decode(errors='replace').strip().splitlines()
            if not lines:
                self._cached_vram_text = 'VRAM: n/a'
                return self._cached_vram_text
            first_line = lines[0]
            used_s, total_s = [x.strip() for x in first_line.split(',')[:2]]
            used = int(used_s)
            total = int(total_s)
            pct = 100.0 * used / total if total else 0.0
            self._cached_vram_text = f'VRAM: {used}/{total} MB ({pct:.0f}%)'
        except Exception:
            self._cached_vram_text = 'VRAM: n/a (nvidia-smi unavailable)'
        return self._cached_vram_text

    def _draw_hud( self, img, frame_time_ns, waiting=False ):
        if not self.hud_enabled or img is None or img.size == 0:
            return
        if waiting:
            # Frame pixel data is not yet available — show a clear placeholder
            # so the user knows the system is alive and waiting for the AI.
            msg = 'Waiting for frame...'
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cy = img.shape[0] // 2
            cx = (img.shape[1] - tw) // 2
            cv2.putText(img, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, msg, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            return
        now_ns = time.perf_counter_ns()
        if frame_time_ns > now_ns and not self._future_frame_warned:
            self._future_frame_warned = True
            print('WARNING: frame timestamp is ahead of local clock; clamping frame age to 0ms')
        frame_age_ms = max(0.0, (now_ns - frame_time_ns) / 1_000_000.0)
        detector_delay_ms = max(0.0, self.latest_processing_delay_ns / 1_000_000.0)
        sync_delay_ms = max(frame_age_ms, detector_delay_ms)
        # Smooth the displayed SYNC value with a light EMA so single-frame
        # GPU spikes don't make the readout alarm unnecessarily.  The raw
        # value is still used for all logic; this is cosmetic only.
        if self._hud_sync_ema_ms == 0.0:
            self._hud_sync_ema_ms = sync_delay_ms
        else:
            self._hud_sync_ema_ms = ((1.0 - _HUD_SYNC_EMA_ALPHA) * self._hud_sync_ema_ms
                                     + _HUD_SYNC_EMA_ALPHA * sync_delay_ms)
        warning = sync_delay_ms > self.sync_warning_ms
        status_text = f"SYNC: {'WARN' if warning else 'OK'} ({self._hud_sync_ema_ms:.1f} ms)"
        _HIGH_PERF_GUIDS = {
            '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c',
            'e9a42b02-d5df-448d-aa00-03f14749eb61',
        }
        _plan = getattr(self, '_power_plan', None)
        _plan_warn = _plan is not None and _plan[0] not in _HIGH_PERF_GUIDS
        lines = [
            f'FPS: {self.display_fps:.1f}',
            f'Infer: {self.latest_inference_latency_ns / 1_000_000.0:.1f} ms',
            self._query_vram_usage(),
            status_text,
            'Timer: 1ms' if self._hi_res_timer_active else 'Timer: default',
            f'Resets: {self.reset_count}',
        ]
        if _plan_warn:
            lines.append(f'Power: {_plan[1]} !')
        color = (0, 0, 255) if warning else (0, 255, 0)
        for idx, text in enumerate(lines):
            y = 22 + idx * 24
            line_color = (0, 0, 255) if (idx == 3 and warning) or (idx == len(lines) - 1 and _plan_warn) else (255, 255, 255)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 1, cv2.LINE_AA)

    def shutdown( self ):
        self.running = False
        if getattr(self, '_hi_res_timer_active', False):
            _end_high_precision_timer()
            self._hi_res_timer_active = False
        try:
            self.stop_recording()
        except Exception:
            pass
        try:
            self.detector_async.shutdown()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        for shm in (getattr(self, 'boxes_shm', None), getattr(self, 'box_hwnds_shm', None), getattr(self, 'box_info_shm', None)):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
        if getattr(self, 'sc', None) is not None:
            self.sc.shutdown()

    def update_sizes( self, sizes ):
        while( len( self.sizes ) ):
            self.sizes.pop(0)
        self.sizes.extend( sizes )

    def set_model_profile( self, profile ):
        self.model_profile = mmc_const.normalize_model_profile( profile, self.model_profile )
        os.environ['mmcModelProfile'] = self.model_profile

    def make_ready( self ):
        self.time_safety_ns = mmc_config.get_time_settings()['time-safety'] * 1000 * 1000 * 1000
        self.detector_async.start()
        self.ready = True

    def reset_runtime( self ):
        """Hot-reset the inference worker without restarting the application.

        Gracefully terminates the existing detector process, clears the shared
        detection buffers and all timing state, re-applies CPU affinity for the
        capture/GUI process, then immediately respawns the inference worker so
        that detection resumes with a clean state.  The screencap shared-memory
        segments are not touched; capture continues uninterrupted during the
        reset.

        Called from the main rendering loop when the user presses 'R'.
        """
        print( 'Hot reset #%d: shutting down detector...' % (self.reset_count + 1,) )

        # Capture the active sizes NOW, while the Manager is still alive.
        # self.sizes is a Manager proxy list — after shutdown() the proxy
        # becomes invalid and list(self.sizes) raises a connection error.
        try:
            current_sizes = list( self.sizes )
        except Exception:
            current_sizes = []

        # Gracefully stop the existing inference worker.
        try:
            self.detector_async.shutdown()
        except Exception as exc:
            print( 'Hot reset: detector shutdown error: %s' % exc )

        # Zero out the detection shared-memory buffers so stale data is not
        # rendered before the new worker produces its first result.
        try:
            self.box_info_np[:] = 0
            self.box_hwnds_np[:] = 0
        except Exception:
            pass

        # Reset all detection/timing state.
        self.last_detection_found = 0
        self.last_detection_delay_key = None
        self.hwnd_times = {}
        self.boxes_hwnd_index = {}
        self.boxes = np.ndarray( (50, 20000, 8), dtype=np.int64 )
        self.size_detection_timings = {}
        self.size_delays = {}
        self._rolling_timings = {}
        self._rolling_recal_counters = {}
        self._sync_warn_since_ns = None
        self._ema_processing_delay_ns = 0.0
        self._hud_sync_ema_ms = 0.0
        self._last_display_tick_ns = 0
        self._future_frame_warned = False

        # Re-apply CPU affinity for the capture/GUI process so the OS
        # does not migrate it to low-power efficiency cores during the pause.
        _set_process_affinity( self.perf_settings.get('capture-gui-affinity-cores', []) )

        # Respawn the inference worker, reusing the same shared-memory segments.
        print( 'Hot reset: respawning detector...' )
        self.detector_async = mmc_detect_loop_async()
        self.detector_async.initialize(
            self.sc.img_shm_name, self.sc.img_coords_name, self.sc.img_ref_name,
            self.sc.img_shape, current_sizes,
            self.boxes_shm_name, self.box_hwnds_shm_name, self.box_info_shm_name,
        )
        # Point self.sizes at the new manager's list so update_sizes() keeps working.
        self.sizes = self.detector_async.sizes
        self.detector_async.start()

        self.reset_count += 1
        print( 'Hot reset complete (reset #%d)' % self.reset_count )

    def go_decorate( self ):
        if not self.ready:
            return

        self.running = True
        img_buffer = []
        self.hwnd_pos = {}

        n = 0
        t_fps = time.perf_counter()

        self.delay_key_print_history = {}

        # Pump the OpenCV event queue from the rendering thread.
        # startWindowThread() is a no-op on platforms that don't need it and is
        # harmless on those that do; catch AttributeError for older OpenCV builds.
        try:
            cv2.startWindowThread()
        except AttributeError:
            pass

        # Persistent snap thread reference so each iteration can check whether
        # the previous snap has finished before starting a new one.  This
        # prevents the render loop from blocking indefinitely when the OS
        # screenshot API (DWM / mss) is slow — which happens, e.g., when the
        # user scrolls a background window while the GUI is in the foreground.
        # A completed dummy thread is used as the initial sentinel so that the
        # first real loop iteration always starts a snap immediately.
        _sentinel = threading.Thread(target=lambda: None)
        _sentinel.start()
        _sentinel.join()
        t1 = _sentinel

        while( True ):
            self.profiler.loop()

            if not self.detector_async.P1.is_alive():
                print( "DETECTOR THREAD FAILED.  EXITING.  PLEASE REPORT ANY ERRORS PRINTED ABOVE." )
                sys.exit()

            # Only start a new snap if the previous snap thread has finished.
            # If the previous snap is still running (e.g. DWM/mss slow during
            # scroll), we skip starting a new one and reuse the last captured
            # frame for this render iteration.  This prevents the render loop
            # from stalling and keeps cv2.waitKey pumping the message queue.
            if not t1.is_alive():
                t1 = threading.Thread( target=self.sc.snap_hwnds, args = [ self.hwnds ] )
                if self.threaded_screenshot:
                    t1.start()
                else:
                    t1.run()

            self.profiler.mark('after_snap')
            num_snapped = self.sc.img_ref[1]
            t_snapped = self.sc.img_ref[0]

            self.profiler.mark('got_ref')

            coords = self.sc.img_coords.copy()

            self.profiler.mark('copied_coords')

            # grab the snapped images
            img_collection = {}
            for i in range(num_snapped):
                hwnd = coords[i][4]
                img_collection[ hwnd ] = [
                    self.sc.img_shared[coords[i][1]:coords[i][3],coords[i][0]:coords[i][2]].copy(), 
                    [ coords[i][0], coords[i][1], coords[i][2], coords[i][3] ] 
                    ]

            self.profiler.mark( 'copied_img' )

            img_buffer.append( [ t_snapped, img_collection ] )

            self.profiler.mark( 'appended_buffer' )

            _local_sizes = list(self.sizes)  # snapshot Manager proxy once per frame to avoid repeated IPC calls
            delay_key = ( len( self.hwnds ), nn.sizes_to_key( _local_sizes ) )
            if delay_key[1] == 0:
                # No nets active — the detector has no inference to run so the
                # only pipeline overhead is IPC latency (~1 ms).  A small fixed
                # delay is sufficient to satisfy the straddle check without
                # inflating the displayed sync unnecessarily.
                delay = 30_000_000  # 30 ms
            elif delay_key in self.size_delays:
                delay = self.size_delays[ delay_key ]
            else:
                if delay_key in self.size_detection_timings:
                    avg = sum( self.size_detection_timings[delay_key] ) / len( self.size_detection_timings[delay_key] )
                    # 1.5× average interval + 20 ms fixed jitter buffer.
                    # The straddle check only requires delay > one inference
                    # cycle; 1.5× provides a 50 % margin for timing variance
                    # without inflating sync by the full time_safety window.
                    calibrated_delay = 1.5 * avg + 20_000_000
                    if len( self.size_detection_timings[ delay_key ] ) > 15 or ( len(self.size_detection_timings[delay_key] ) > 4 and sum( self.size_detection_timings[delay_key] ) > 4 * 1000000000 ):
                        print( self.size_detection_timings[ delay_key ] )
                        delay = calibrated_delay
                        self.size_delays[delay_key] = delay
                        print( 'delay set to %.3fs'%(delay/1000000000,) )
                    else:
                        # Use the running average of collected samples as a live
                        # estimate so sync never spikes to 3 s during calibration.
                        delay = calibrated_delay
                        if delay_key not in self.delay_key_print_history or len(self.delay_key_print_history[ delay_key ]) != len(self.size_detection_timings[ delay_key ]):
                            print( "calculating delay....", self.size_detection_timings[delay_key] )
                            self.delay_key_print_history[ delay_key ] = self.size_detection_timings[ delay_key ].copy()
                else:
                    if delay_key not in self.delay_key_print_history or self.delay_key_print_history[ delay_key ] != []:
                        self.delay_key_print_history[ delay_key ] = []
                        print( "calculating delay...." )
                    # Use time_safety_ns × 2 as the cold-start minimum, but raise
                    # the floor to (_infer_ns + time_safety_ns) once the first
                    # inference completes.  For slow nets (1280/1920/2560) the
                    # 300 ms default is shorter than the actual inference time,
                    # causing the buffer to not hold a frame old enough for the
                    # straddle check even after the first detection arrives.
                    delay = max(self.time_safety_ns * 2,
                                max(self.latest_inference_latency_ns, self.time_safety_ns) + self.time_safety_ns)

            self.profiler.mark( 'popped_old' )

            # eliminate old detections
            if not img_buffer:
                continue
            # Compute the detection-window width first so it can inform the display
            # frame selection below.
            _infer_ns = max(self.latest_inference_latency_ns, self.time_safety_ns)
            # Show the newest frame that is at least (_infer_ns + 10 ms) old.
            # That margin guarantees there is a detection snap before the display
            # frame (straddle condition) while keeping display latency — and
            # therefore the SYNC readout — as low as possible.
            # 10 ms (down from 20 ms) is still well above IPC round-trip latency
            # on modern hardware and reduces steady-state frame_age_ms directly.
            display_delay = min(max(_infer_ns + 10_000_000, 50_000_000), delay)
            _target_time = time.perf_counter_ns() - display_delay
            display_idx = 0
            for _di in range(len(img_buffer)):
                if img_buffer[_di][0] <= _target_time:
                    display_idx = _di
                else:
                    break
            # Trim stale frames from the front immediately after selecting the
            # display frame.  Frames older than display_idx will never be shown
            # again (because _target_time only advances).  The previous strategy
            # trimmed by `delay` (up to 1.5 × inference_interval old), keeping
            # up to ~90 frames × frame_size in memory for slow nets — this caused
            # Python GC to collect hundreds of MB of numpy arrays periodically,
            # producing the light freezes / stutters.  With this trim the buffer
            # holds only the few frames between display_idx and the latest snap.
            if display_idx > 0:
                del img_buffer[:display_idx]
                display_idx = 0
            # Hard cap: never buffer more than _MAX_IMG_BUF frames regardless of
            # timing, to bound worst-case memory use during cold-start or snap stalls.
            _MAX_IMG_BUF = 120
            if len(img_buffer) > _MAX_IMG_BUF:
                del img_buffer[:len(img_buffer) - _MAX_IMG_BUF]
            to_show_time_ns = img_buffer[display_idx][0]
            oldest_detection = to_show_time_ns - _infer_ns
            latest_detection = to_show_time_ns + self.time_safety_ns
            for hwnd in self.hwnd_times:
                popped = False
                while( len( self.hwnd_times[hwnd] )>1 and self.hwnd_times[hwnd][1][0] < oldest_detection ):
                    self.hwnd_times[hwnd].pop(0)
                    popped = True
                if popped:
                    index = self.boxes_hwnd_index[hwnd]
                    first_index_to_keep = self.hwnd_times[hwnd][0][1]
                    last_index_to_keep = self.hwnd_times[hwnd][-1][2]
                    self.boxes[index][:last_index_to_keep-first_index_to_keep+1]=self.boxes[index][first_index_to_keep:last_index_to_keep+1]
                    for elt in self.hwnd_times[hwnd]:
                        elt[1] = elt[1] - first_index_to_keep
                        elt[2] = elt[2] - first_index_to_keep

            self.profiler.mark( 'copied_Q' )

            detection_time = self.box_info_np[0]
            if detection_time > self.last_detection_found:
                self.latest_detection_write_ns = int(self.box_info_np[3])
                self.latest_inference_latency_ns = int(self.box_info_np[4])
                self.latest_processing_delay_ns = int(self.box_info_np[5])
                self.latest_detection_snap_ns = int(self.box_info_np[6])
                # Update the EMA used by the auto-reset guard.  We initialise
                # to the raw value on the first sample to avoid a cold-start
                # bias toward 0 triggering spurious resets during warm-up.
                _raw_delay = float(self.latest_processing_delay_ns)
                if self._ema_processing_delay_ns == 0.0:
                    self._ema_processing_delay_ns = _raw_delay
                else:
                    self._ema_processing_delay_ns = (
                        (1.0 - _AUTO_RESET_EMA_ALPHA) * self._ema_processing_delay_ns
                        + _AUTO_RESET_EMA_ALPHA * _raw_delay
                    )
                for i in range(self.box_info_np[1]):
                    hwnd = self.box_hwnds_np[i][0]
                    num_boxes = self.box_hwnds_np[i][1]
                    if hwnd not in self.boxes_hwnd_index:
                        self.boxes_hwnd_index[hwnd] = len(self.boxes_hwnd_index)
                        self.hwnd_times[ hwnd ] = []
                    if len( self.hwnd_times[hwnd] ):
                        new_first_index = self.hwnd_times[hwnd][-1][2] + 1
                        new_last_index = self.hwnd_times[hwnd][-1][2] + num_boxes
                    else:
                        new_first_index = 0
                        new_last_index = num_boxes-1
                    if num_boxes:
                        self.boxes[self.boxes_hwnd_index[hwnd]][new_first_index:new_last_index+1]=self.boxes_np[i][0:num_boxes]
                    self.hwnd_times[hwnd].append( [ detection_time, new_first_index, new_last_index ] )
                detected_delay_key = (self.box_info_np[1], self.box_info_np[2])
                if self.last_detection_found > 0:
                    # Only record same-key intervals; cross-key gaps include warmup
                    # time and would inflate the calibration estimate.
                    if self.last_detection_delay_key == detected_delay_key:
                        interval = detection_time - self.last_detection_found
                        if detected_delay_key not in self.size_delays:
                            # Pre-calibration: collect for initial calibration.
                            self.size_detection_timings.setdefault(detected_delay_key, []).append(interval)
                        else:
                            # Post-calibration: rolling re-calibration (capped buffer).
                            if detected_delay_key not in self._rolling_timings:
                                self._rolling_timings[detected_delay_key] = deque(maxlen=_N_ROLLING)
                                self._rolling_recal_counters[detected_delay_key] = 0
                            self._rolling_timings[detected_delay_key].append(interval)
                            self._rolling_recal_counters[detected_delay_key] += 1
                            _rbuf = self._rolling_timings[detected_delay_key]
                            # Use the 75th percentile of the rolling buffer instead of
                            # mean × 1.5.  The percentile ignores the top-25 % of GPU
                            # spikes while still covering the typical-slow case, so the
                            # calibrated delay is tighter and SYNC stays lower.
                            _rolling_p75 = float(np.percentile(list(_rbuf), 75))
                            _rolling_avg = sum(_rbuf) / len(_rbuf)
                            _cur_delay = self.size_delays[detected_delay_key]
                            _new_delay = 1.2 * _rolling_p75 + 15_000_000  # + 15 ms fixed jitter buffer
                            if _rolling_avg > 1.5 * _cur_delay:
                                # Immediate update: rolling mean has grown >50 % above current delay.
                                self.size_delays[detected_delay_key] = _new_delay
                                self._rolling_recal_counters[detected_delay_key] = 0
                                print('[rolling-recal] immediate update: delay %.3fs -> %.3fs (%s)' % (
                                    _cur_delay / 1e9, _new_delay / 1e9,
                                    datetime.now().strftime('%H:%M:%S')))
                            elif self._rolling_recal_counters[detected_delay_key] >= _ROLLING_RECAL_EVERY:
                                # Periodic update every _ROLLING_RECAL_EVERY samples.
                                self.size_delays[detected_delay_key] = _new_delay
                                self._rolling_recal_counters[detected_delay_key] = 0
                                print('[rolling-recal] periodic update: delay %.3fs -> %.3fs (%s)' % (
                                    _cur_delay / 1e9, _new_delay / 1e9,
                                    datetime.now().strftime('%H:%M:%S')))
                self.last_detection_delay_key = detected_delay_key
                self.last_detection_found = detection_time

            self.profiler.mark( 'reshaped_boxes' )

            has_gray_img = False
            for hwnd in self.to_show:
                if hwnd not in img_buffer[-1][1]:
                    self.to_show[hwnd] = None
            for hwnd in img_buffer[-1][1]: # the list of things to show is whatever the *latest* list of captured coordinates is
                self.profiler.mark( 'pre_full' )
                new_xyxy = img_buffer[-1][1][hwnd][1]
                _h = new_xyxy[3]-new_xyxy[1]
                _w = new_xyxy[2]-new_xyxy[0]
                if hwnd not in self.to_show or self.to_show[hwnd] is None or self.to_show[hwnd].shape[:2] != (_h, _w):
                    self.to_show[hwnd] = np.full( (_h, _w, 3), 127, dtype=np.uint8 )
                self.profiler.mark( 'post_full' )

                # Straddle check: the oldest stored detection must be before
                # the display frame (oldest < display) and the newest must be
                # recent enough relative to the display frame.
                #
                # We use `delay` (not `_infer_ns`) as the freshness threshold.
                # `delay` = 1.5 × avg_inference_interval, so the condition only
                # fails when the newest detection is more than `delay` older than
                # the display frame — i.e. when the detector has been silent for
                # an entire display-window worth of time.  Using _infer_ns here
                # was too tight: whenever a single inference cycle ran longer
                # than `delay` (GPU load spike or just normal variance on slow
                # nets), the check failed and the screen went gray until the
                # next detection arrived, producing periodic "waiting for frame"
                # stutter every inference cycle on 1280/1920/2560 nets.
                if hwnd in img_buffer[display_idx][1] and hwnd in self.hwnd_times and self.hwnd_times[hwnd][0][0] < img_buffer[display_idx][0] and self.hwnd_times[hwnd][-1][0] > img_buffer[display_idx][0] - delay:
                    old_xyxy = img_buffer[display_idx][1][hwnd][1]
                    self.profiler.mark( 'got_old_xyxy' )
                    min_h = min( old_xyxy[3] - old_xyxy[1], new_xyxy[3] - new_xyxy[1] )
                    min_w = min( old_xyxy[2] - old_xyxy[0], new_xyxy[2] - new_xyxy[0] )

                    self.to_show[hwnd][0:min_h,0:min_w] = img_buffer[display_idx][1][hwnd][0][0:min_h,0:min_w]
                    self.profiler.mark( 'populated_show' )

                    for i in range(len(self.hwnd_times[hwnd])):
                        if self.hwnd_times[hwnd][i][0]>latest_detection:
                            break
                    last_box_index = self.hwnd_times[hwnd][i][2]

                    # you could do this faster by intersecting with window size as you
                    # go, but it's really annoying
                    # this probably isn't that slow
                    _src = self.boxes[self.boxes_hwnd_index[hwnd]][0:last_box_index+1]
                    _mask = (_src[:,2] < min_w) & (_src[:,3] < min_h)
                    relevant_boxes = _src[_mask]
                    relevant_boxes[:,4]=np.fmin(relevant_boxes[:,4],min_w)
                    relevant_boxes[:,5]=np.fmin(relevant_boxes[:,5],min_h)

                    for decorator in self.decorators:
                        self.to_show[hwnd][0:min_h,0:min_w] = decorator.decorate( self.to_show[hwnd][0:min_h,0:min_w], relevant_boxes )

                    self.profiler.mark( 'decorated' )
                    is_waiting = False
                else:
                    has_gray_img = True
                    is_waiting = True

                now_ns = time.perf_counter_ns()
                if self._last_display_tick_ns:
                    inst_fps = NS_PER_SECOND / max(MIN_FPS_INTERVAL_NS, now_ns - self._last_display_tick_ns)
                    self.display_fps = inst_fps if self.display_fps == 0 else ((1.0 - FPS_EMA_ALPHA) * self.display_fps + FPS_EMA_ALPHA * inst_fps)
                self._last_display_tick_ns = now_ns
                self.show( self.to_show[ hwnd ], hwnd, new_xyxy, to_show_time_ns, waiting=is_waiting )
                self.profiler.mark( 'showed' )

                # ── Record censored frame if recording is active ──────────
                if self.recording and self.to_show[hwnd] is not None:
                    img = self.to_show[hwnd]
                    h, w = img.shape[:2]
                    # Release writer and recreate if frame size has changed.
                    # NOTE: VideoWriter.get(CAP_PROP_FRAME_WIDTH/HEIGHT) always returns 0,
                    # so we track dimensions in video_writer_dims instead.
                    if hwnd in self.video_writers and self.video_writer_dims.get(hwnd) != (w, h):
                        self.video_writers[hwnd].release()
                        del self.video_writers[hwnd]
                        del self.video_writer_dims[hwnd]
                    if hwnd not in self.video_writers:
                        base, ext = os.path.splitext(self.recording_path)
                        if not ext:
                            ext = '.mp4'
                        if hwnd not in self._rec_hwnd_paths:
                            idx = self._rec_next_idx
                            self._rec_next_idx += 1
                            filepath = (base + ext) if idx == 0 else ('%s_%d%s' % (base, idx, ext))
                            self._rec_hwnd_paths[hwnd] = filepath
                        # Choose codec based on extension for compatibility
                        codec = 'XVID' if ext.lower() == '.avi' else 'mp4v'
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        self.video_writers[hwnd] = cv2.VideoWriter(
                            self._rec_hwnd_paths[hwnd], fourcc, 30.0, (w, h))
                        self.video_writer_dims[hwnd] = (w, h)
                        if not self.video_writers[hwnd].isOpened():
                            print( 'WARNING: could not open video writer for %s' % self._rec_hwnd_paths[hwnd] )
                    if self.video_writers[hwnd].isOpened():
                        self.video_writers[hwnd].write(img)

            # avoid deleting from dict while iterating over dict
            windows_to_close = []
            for window_hwnd in self.open_windows:
                if window_hwnd not in self.to_show or self.to_show[window_hwnd] is None:
                    windows_to_close.append( window_hwnd )
            for window_hwnd in windows_to_close:
                cv2.destroyWindow( self.open_windows[window_hwnd] )
                del self.open_windows[ window_hwnd ]
                del self.hwnd_pos[ window_hwnd ]

            if self.gray_state == True and has_gray_img == False and self.off_gray_callback is not None:
                self.off_gray_callback()
                self.gray_state = False

            if self.gray_state == False and has_gray_img == True and self.on_gray_callback is not None:
                self.on_gray_callback()
                self.gray_state = True

            self.profiler.mark('closed_windows')

            # --- Auto-reset on chronic SYNC WARN ---
            if self.auto_reset_sync_s > 0 and self.hwnds:
                _now_ns = time.perf_counter_ns()
                _frame_age_ms = max(0.0, (_now_ns - to_show_time_ns) / 1_000_000.0)
                # Use the EMA-smoothed processing delay instead of the raw
                # instantaneous value so that a single slow GPU inference cycle
                # does not start the auto-reset countdown.  Only a sustained
                # degradation (many consecutive slow frames) will trigger it.
                _det_delay_ms = max(0.0, self._ema_processing_delay_ns / 1_000_000.0)
                _is_warn = max(_frame_age_ms, _det_delay_ms) > self.sync_warning_ms
                if _is_warn:
                    if self._sync_warn_since_ns is None:
                        self._sync_warn_since_ns = _now_ns
                    elif (_now_ns - self._sync_warn_since_ns) >= self.auto_reset_sync_s * NS_PER_SECOND:
                        print('[auto-reset] chronic SYNC WARN for %.0fs — resetting runtime (%s)' % (
                            self.auto_reset_sync_s, datetime.now().strftime('%H:%M:%S')))
                        if t1.is_alive():
                            t1.join(timeout=0.5)
                        self.reset_runtime()
                        self._sync_warn_since_ns = None
                else:
                    self._sync_warn_since_ns = None
            if n == 100:
                elapsed = time.perf_counter() - t_fps
                print( '100 frames in %.3fs, or %.1fps'%( elapsed, 100/elapsed ))
                n = 0
                t_fps = time.perf_counter()

            self.profiler.mark( 'post_fps' )

            key = cv2.waitKey(1)

            # 'R' / 'r' — hot-reset the inference worker at runtime.
            if key == ord('r') or key == ord('R'):
                # Wait for the current snap thread before resetting so it
                # cannot write into shared memory mid-reset.  Use a timeout
                # so that a stalled snap thread (e.g. DWM busy during scroll)
                # does not block the reset indefinitely.
                if t1.is_alive():
                    t1.join( timeout=0.5 )
                self.reset_runtime()

            if( key == ord('q') or self.running == False ):
                cv2.destroyAllWindows()
                self.open_windows = {}
                if t1.is_alive():
                    t1.join( timeout=0.5 )
                self.stop_recording()
                break

            self.profiler.mark( 'post_wait' )

            # Wait for the snap thread to finish before starting the next
            # iteration's snap.  Use a short timeout so the render loop never
            # blocks indefinitely if the snap thread is stuck.  If the timeout
            # expires, the next iteration will detect t1.is_alive() == True
            # and skip starting a new snap, reusing the previous frame instead.
            if t1.is_alive():
                t1.join( timeout=0.3 )

            self.profiler.mark( 'post_join' )

    def show( self, img, real_hwnd, new_xyxy, frame_time_ns, waiting=False ):
        cv_title = self.cv_title_template%real_hwnd
        self.open_windows[ real_hwnd ] = cv_title
        self._draw_hud( img, frame_time_ns, waiting=waiting )
        cv2.imshow( cv_title, img )
        #cv2.imshow( 'rec', img )
        self.profiler.mark( 'show_call')
        if real_hwnd not in self.hwnd_pos or self.hwnd_pos[real_hwnd] != new_xyxy:
            hwnd = win32gui.FindWindow(None, cv_title )
            self.profiler.mark( 'show_find_hwnd')
            if hwnd:
                # Get window style and perform a 'bitwise or' operation to make the style layered and transparent, achieving
                # the clickthrough property
                ctypes.windll.user32.SetWindowDisplayAffinity( hwnd, 0x00000011 )
                self.profiler.mark( 'show_affinity')
                l_ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                self.profiler.mark( 'show_ex_style')
                l_ex_style |= win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, l_ex_style)
                self.profiler.mark( 'show_set_long')

                # Set the window to be transparent and appear always on top
                win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 255, win32con.LWA_ALPHA)  # transparent
                self.profiler.mark( 'show_transparent')
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, self.sc.visible_bounds[0] + new_xyxy[0], self.sc.visible_bounds[1] + new_xyxy[1], new_xyxy[2]-new_xyxy[0], new_xyxy[3]-new_xyxy[1], 0 )
                self.profiler.mark( 'show_pos1')
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOP,     self.sc.visible_bounds[0] + new_xyxy[0], self.sc.visible_bounds[1] + new_xyxy[1], new_xyxy[2]-new_xyxy[0], new_xyxy[3]-new_xyxy[1], 0 )
                self.profiler.mark( 'show_pos2')

                GWL_STYLE = -16

                currentStyle = win32gui.GetWindowLong(hwnd, GWL_STYLE)
                self.profiler.mark( 'show_getlonggwl')

                #  remove titlebar elements
                currentStyle = currentStyle & ~(0x00C00000)  #  WS_CAPTION
                currentStyle = currentStyle & ~(0x00080000)  #  WS_SYSMENU
                currentStyle = currentStyle & ~(0x00040000)  #  WS_THICKFRAME
                currentStyle = currentStyle & ~(0x20000000)  #  WS_MINIMIZE
                currentStyle = currentStyle & ~(0x00010000)  #  WS_MAXIMIZEBOX

                #  apply new style
                win32gui.SetWindowLong(hwnd, GWL_STYLE, currentStyle)
                self.profiler.mark( 'show_setlonggwl')
                self.hwnd_pos[ real_hwnd ] = new_xyxy
                
