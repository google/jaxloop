"""Minimal Imagenet dataset pipeline using tf.data for ResNet."""

import functools
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from google3.third_party.cloud_tpu.models.resnet import resnet_preprocessing


def training_transform(
    batch: Dict[str, tf.Tensor],
    img_size: Tuple[int, int],
) -> Dict[str, tf.Tensor]:
  """ImageNet training preprocessing."""
  batch['image'] = resnet_preprocessing.preprocess_for_train(
      batch['image'],
      use_bfloat16=False,
      image_size=img_size[0],
  )
  del batch['file_name']  # Not supported by partitioning.
  return batch


def validation_transform(
    batch: Dict[str, tf.Tensor],
    img_size: Tuple[int, int],
) -> Dict[str, tf.Tensor]:
  """ImageNet validation preprocessing."""
  batch['image'] = resnet_preprocessing.preprocess_for_eval(
      batch['image'], use_bfloat16=False, image_size=img_size[0]
  )
  del batch['file_name']  # Not supported by partitioning.
  return batch


def imagenet_datasets(
    input_size: Tuple[int, int],
    batch_size: int,
    data_dir: str,
):
  """Creates a training and validation dataset that produces batches of a given shape."""
  ds_builder = tfds.builder('imagenet2012:5.1.0', data_dir=data_dir)
  train_len = ds_builder.info.splits['train'].num_examples // batch_size
  ds_train = (
      ds_builder.as_dataset(
          split=tfds.Split.TRAIN,
          shuffle_files=True,
          decoders={'image': tfds.decode.SkipDecoding()},
      )
      .shuffle(max(1024, batch_size * 4))
      .map(
          functools.partial(training_transform, img_size=input_size),
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
      )
      .repeat()
      .batch(batch_size, drop_remainder=True)
      .prefetch(tf.data.experimental.AUTOTUNE)
  )
  ds_val = (
      ds_builder.as_dataset(
          split=tfds.Split.VALIDATION,
          decoders={'image': tfds.decode.SkipDecoding()},
      )
      .map(
          functools.partial(validation_transform, img_size=input_size),
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
      )
      .batch(batch_size)
      .prefetch(tf.data.experimental.AUTOTUNE)
  )
  return train_len, iter(tfds.as_numpy(ds_train)), tfds.as_numpy(ds_val)
