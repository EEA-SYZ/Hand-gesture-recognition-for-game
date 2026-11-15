# -*- coding: utf-8 -*-
"""A small, dependency-light reimplementation of PaddleX 1.x ``Predictor``.

Only classification inference models are supported.  The preprocessing
pipeline is rebuilt from the ``Transforms`` section of ``model.yml`` exactly
like PaddleX 1.3's ``build_transforms`` does (BGR -> RGB, RandomCrop /
CenterCrop / ResizeByShort / Normalize, HWC -> CHW).
"""

from __future__ import absolute_import

import math
import os
import os.path as osp
import warnings

import cv2
import numpy as np
import paddle
import yaml


def _random_crop(im, crop_size=224, lower_scale=0.08, lower_ratio=3. / 4,
                 upper_ratio=4. / 3):
    """PaddleX ``random_crop`` (training-time augmentation used by this model)."""
    height, width = im.shape[:2]
    aspect_ratio = math.sqrt(np.random.uniform(lower_ratio, upper_ratio))
    w_ratio = aspect_ratio
    h_ratio = 1. / aspect_ratio
    bound = min((float(height) / width) / (h_ratio ** 2),
                (float(width) / height) / (w_ratio ** 2))
    scale_max = min(1.0, bound)
    scale_min = min(lower_scale, bound)
    target_area = height * width * np.random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_area)
    crop_w = max(1, int(target_size * w_ratio))
    crop_h = max(1, int(target_size * h_ratio))
    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)
    top = np.random.randint(0, height - crop_h + 1)
    left = np.random.randint(0, width - crop_w + 1)
    im = im[top:top + crop_h, left:left + crop_w, :]
    return cv2.resize(im, (crop_size, crop_size))


def _center_crop(im, crop_size=224):
    height, width = im.shape[:2]
    top = max(0, (height - crop_size) // 2)
    left = max(0, (width - crop_size) // 2)
    im = im[top:top + crop_size, left:left + crop_size, :]
    if im.shape[0] != crop_size or im.shape[1] != crop_size:
        im = cv2.resize(im, (crop_size, crop_size))
    return im


def _resize_by_short(im, short_size=256, max_size=-1):
    height, width = im.shape[:2]
    short = min(height, width)
    long_side = max(height, width)
    scale = float(short_size) / short
    if max_size > 0 and round(scale * long_side) > max_size:
        scale = float(max_size) / long_side
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    return cv2.resize(im, (resized_w, resized_h),
                      interpolation=cv2.INTER_LINEAR)


def _normalize(im, mean, std):
    mean = np.array(mean, dtype="float32")[np.newaxis, np.newaxis, :]
    std = np.array(std, dtype="float32")[np.newaxis, np.newaxis, :]
    im = im.astype("float32") / 255.0
    im = (im - mean) / std
    return im.astype("float32")


def _softmax(x):
    x = np.asarray(x, dtype="float64")
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype("float32")


class Predictor(object):
    """Drop-in replacement for ``paddlex.deploy.Predictor`` (classifier only).

    Args:
        model_dir (str): directory containing ``__model__``, ``__params__``
            and ``model.yml``.
        use_gpu (bool): request GPU inference; silently falls back to CPU
            when the installed Paddle build has no usable CUDA device.
        cpu_threads (int): CPU inference thread count.
        Others (``gpu_mem``, ``use_mkl``, ...): accepted for API compatibility
            and ignored when they do not apply.
    """

    def __init__(self,
                 model_dir,
                 use_gpu=False,
                 gpu_mem=8000,
                 cpu_threads=2,
                 use_mkl=False,
                 **kwargs):
        model_dir = str(model_dir)
        if not osp.isdir(model_dir):
            raise Exception("[ERROR] Path {} not exist.".format(model_dir))
        model_file = osp.join(model_dir, "__model__")
        params_file = osp.join(model_dir, "__params__")
        yml_file = osp.join(model_dir, "model.yml")
        for path in (model_file, params_file, yml_file):
            if not osp.exists(path):
                raise Exception("[ERROR] {} not found.".format(path))

        with open(yml_file, "r", encoding="utf-8") as f:
            self.info = yaml.safe_load(f.read())

        self.model_dir = model_dir
        self.status = self.info.get("status", "Infer")
        attributes = self.info.get("_Attributes", {})
        self.model_type = attributes.get("model_type", "classifier")
        self.model_name = self.info.get("Model", "Unknown")
        self.num_classes = int(attributes.get("num_classes", 0))
        self.labels = list(attributes.get("labels", []))
        if self.model_type != "classifier":
            raise Exception("[ERROR] Only classifier models are supported, "
                            "got '{}'.".format(self.model_type))
        if not self.labels:
            raise Exception("[ERROR] No labels found in model.yml.")
        if self.num_classes and self.num_classes != len(self.labels):
            raise Exception("[ERROR] num_classes != len(labels) in model.yml.")
        self.transforms = list(self.info.get("Transforms") or [])
        self.transforms_mode = self.info.get("TransformsMode", "RGB")
        self._transforms_desc = [list(t.keys())[0] for t in self.transforms]

        self.use_gpu = bool(use_gpu) and self._cuda_available()
        if use_gpu and not self.use_gpu:
            warnings.warn(
                "[paddlex-compat] use_gpu=True was requested but this Paddle "
                "build has no usable CUDA device; running on CPU.")
        self.predictor = self._create_predictor(
            use_gpu=self.use_gpu,
            cpu_threads=int(cpu_threads or 2),
            use_mkl=bool(use_mkl))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _cuda_available():
        try:
            if not paddle.device.is_compiled_with_cuda():
                return False
            return paddle.device.cuda.device_count() > 0
        except Exception:
            return False

    def _create_predictor(self, use_gpu, cpu_threads, use_mkl):
        last_error = None
        # IR optimisation is normally safe; retry without it for old models.
        for ir_optim in (True, False):
            try:
                config = paddle.inference.Config(
                    osp.join(self.model_dir, "__model__"),
                    osp.join(self.model_dir, "__params__"))
                config.disable_glog_info()
                config.enable_memory_optim()
                config.switch_ir_optim(ir_optim)
                if use_gpu:
                    config.enable_use_gpu(100, 0)
                else:
                    config.disable_gpu()
                    config.set_cpu_math_library_num_threads(
                        max(1, min(cpu_threads, 16)))
                    if use_mkl:
                        try:
                            config.enable_mkldnn()
                        except Exception:
                            pass
                return paddle.inference.create_predictor(config)
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc
                if not ir_optim:
                    break
        raise RuntimeError(
            "[ERROR] Failed to load Paddle inference model: {}".format(
                last_error))

    # -------------------------------------------------------------- transforms
    def _preprocess(self, image):
        if isinstance(image, str):
            im = cv2.imread(image)
            if im is None:
                raise Exception("[ERROR] Cannot read image {}".format(image))
        elif isinstance(image, np.ndarray):
            im = image
        else:
            raise TypeError("image must be a path or a numpy array")
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        im = im.astype("float32")
        # PaddleX Compose always feeds RGB to the network.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            name, attr = next(iter(op.items()))
            attr = attr or {}
            if name == "RandomCrop":
                # model.yml stores the *training* pipeline (RandomCrop).  At
                # prediction time PaddleX's evaluation pipeline is
                # ResizeByShort(crop_size * 1.14) + CenterCrop(crop_size),
                # which is also how the reported 91.6% accuracy was measured.
                # Using the random crop here would make the game direction
                # flicker between frames.
                crop_size = int(attr.get("crop_size", 224))
                im = _resize_by_short(im, int(crop_size * 1.14))
                im = _center_crop(im, crop_size)
            elif name == "CenterCrop":
                im = _center_crop(im, attr.get("crop_size", 224))
            elif name == "ResizeByShort":
                im = _resize_by_short(im, attr.get("short_size", 256),
                                      attr.get("max_size", -1))
            elif name == "Resize":
                target = attr.get("target_size", 224)
                if isinstance(target, (list, tuple)):
                    im = cv2.resize(im, (int(target[0]), int(target[1])))
                else:
                    im = cv2.resize(im, (int(target), int(target)))
            elif name == "Normalize":
                im = _normalize(
                    im, attr.get("mean", [0.485, 0.456, 0.406]),
                    attr.get("std", [0.229, 0.224, 0.225]))
            elif name in ("RandomHorizontalFlip", "RandomVerticalFlip"):
                # Augmentations are not used for prediction; ignore them so
                # the controller does not flicker between frames.
                continue
            else:
                warnings.warn(
                    "[paddlex-compat] Unsupported transform '{}' ignored."
                    .format(name))

        if im.ndim != 3:
            raise Exception("[ERROR] Preprocessed image is not HWC.")
        im = np.transpose(im, (2, 0, 1)).astype("float32")
        return np.expand_dims(im, axis=0)

    # ---------------------------------------------------------------- predict
    def raw_predict(self, im):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        input_handle.copy_from_cpu(np.ascontiguousarray(im, dtype="float32"))
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        output_name = "predict" if "predict" in output_names else output_names[0]
        output = self.predictor.get_output_handle(output_name).copy_to_cpu()
        return np.asarray(output)

    def predict(self, image, topk=1, transforms=None, **kwargs):
        """Run classification on one BGR image (path or ``numpy.ndarray``).

        Returns:
            list[dict]: top-k entries with ``category_id``, ``category`` and
            ``score`` keys, sorted by score descending.
        """
        if transforms is not None:
            raise NotImplementedError(
                "Custom transforms are not supported by this compatibility layer.")
        batched = self._preprocess(image)
        output = self.raw_predict(batched)
        scores = np.asarray(output)
        if scores.ndim == 2:
            scores = scores[0]
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        if scores.size != len(self.labels):
            raise Exception(
                "[ERROR] Model returned {} scores for {} labels.".format(
                    scores.size, len(self.labels)))
        if not np.allclose(scores.sum(), 1.0, atol=1e-3):
            scores = _softmax(scores)

        true_topk = max(1, min(int(topk or 1), len(self.labels)))
        order = np.argsort(scores)[::-1][:true_topk]
        return [{
            "category_id": int(idx),
            "category": self.labels[int(idx)],
            "score": float(scores[int(idx)]),
        } for idx in order]

    def batch_predict(self, image_list, topk=1, transforms=None, **kwargs):
        return [self.predict(image, topk=topk) for image in image_list]

    # ------------------------------------------------------------------- misc
    def __repr__(self):
        return ("Predictor(model={!r}, labels={!r}, device={!r})".format(
            self.model_name, self.labels,
            "gpu" if self.use_gpu else "cpu"))
