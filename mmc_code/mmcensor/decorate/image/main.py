import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import random
from mmcensor.decorate.decorator_utils import feature_selector
import mmcensor.geo as geo

_SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}

# Default folder relative to the working directory (mmc_code/)
_DEFAULT_FOLDER = 'mmcensor/decorate/image_overlays'


def _load_images_from_folder(folder):
    """Return a list of BGR numpy arrays loaded from *folder*.

    Only files with a supported image extension are loaded.  Returns an empty
    list when the folder does not exist or contains no readable images.
    """
    images = []
    if not folder or not os.path.isdir(folder):
        return images
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in _SUPPORTED_EXTS:
            continue
        path = os.path.join(folder, name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


def _fit_and_crop(image, target_w, target_h):
    """Scale *image* to cover (*target_w*, *target_h*) then center-crop.

    The image aspect ratio is preserved; the image is scaled up or down so
    that it is *at least* as large as the target in both dimensions, then the
    excess is cropped from the centre.  The result is always exactly
    (*target_w*, *target_h*) pixels.
    """
    if target_w <= 0 or target_h <= 0:
        return None
    ih, iw = image.shape[:2]
    scale = max(target_w / iw, target_h / ih)
    new_w = max(int(round(iw * scale)), target_w)
    new_h = max(int(round(ih * scale)), target_h)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA
                         if scale < 1 else cv2.INTER_LINEAR)
    # centre-crop
    y0 = (new_h - target_h) // 2
    x0 = (new_w - target_w) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]


class decorator:

    def initialize(self, known_classes):
        self.known_classes = known_classes
        self.classes = []
        self.folder = _DEFAULT_FOLDER
        self._images = _load_images_from_folder(self.folder)

    # ── core decoration ────────────────────────────────────────────────────

    def decorate(self, img, boxes):
        if not self._images:
            return img

        condensed = geo.condense_boxes_single(boxes)

        for feature in condensed:
            if self.known_classes[feature] not in self.classes:
                continue
            for box in condensed[feature]:
                x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                # clip box to image boundaries
                ih, iw = img.shape[:2]
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(iw, x2)
                y2c = min(ih, y2)
                bw = x2c - x1c
                bh = y2c - y1c
                if bw <= 0 or bh <= 0:
                    continue
                src = random.choice(self._images)
                patch = _fit_and_crop(src, bw, bh)
                if patch is None:
                    continue
                img[y1c:y2c, x1c:x2c] = patch

        return img

    # ── settings persistence ───────────────────────────────────────────────

    def export_settings(self):
        return {'classes': self.classes, 'folder': self.folder}

    def import_settings(self, settings):
        self.classes = settings.get('classes', [])
        self.folder = settings.get('folder', _DEFAULT_FOLDER)
        self._images = _load_images_from_folder(self.folder)

    # ── UI helpers ─────────────────────────────────────────────────────────

    def short_desc(self):
        n = len(self._images)
        return '%d classes, %d image%s' % (len(self.classes), n, '' if n == 1 else 's')

    def populate_config_frame(self, frame):
        tk.Label(frame, text="Image folder:").grid(row=0, column=0, sticky='w')

        self._folder_var = tk.StringVar(value=self.folder)
        folder_entry = tk.Entry(frame, textvariable=self._folder_var, width=40)
        folder_entry.grid(row=0, column=1, padx=4, sticky='ew')

        def _browse():
            chosen = filedialog.askdirectory(title='Select image folder')
            if chosen:
                self._folder_var.set(chosen)

        tk.Button(frame, text='Browse…', command=_browse).grid(row=0, column=2, padx=4)

        self._img_count_label = tk.Label(
            frame, text=self._img_count_text())
        self._img_count_label.grid(row=1, column=0, columnspan=3, sticky='w', pady=(2, 6))

        self.feature_selector = feature_selector()
        class_frame = tk.Frame(frame)
        self.feature_selector.populate_frame(class_frame, self.known_classes, self.classes)
        class_frame.grid(row=2, column=0, columnspan=3)

    def apply_config_from_config_frame(self):
        self.classes = self.feature_selector.get_selected_classes()
        new_folder = self._folder_var.get().strip()
        if new_folder != self.folder:
            self.folder = new_folder
            self._images = _load_images_from_folder(self.folder)
        if hasattr(self, '_img_count_label'):
            self._img_count_label.config(text=self._img_count_text())

    def destroy_config_frame(self):
        return 0

    def _img_count_text(self):
        n = len(self._images)
        return 'Loaded: %d image%s' % (n, '' if n == 1 else 's')
