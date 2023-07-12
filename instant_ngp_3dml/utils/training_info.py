#!/usr/bin/python3
"""Training info."""

from dataclasses import dataclass
from typing import List


@dataclass
class StepInfo:
    """Step Info"""
    step: int
    loss: float
    time: float  # Result from time.monotonic()
    depth_supervision_lambda: float


@dataclass
class TrainingInfo:
    """NeRF Training Info"""
    begin_time: float  # Result from time.monotonic()
    end_time: float  # Result from time.monotonic()
    steps_info: List[StepInfo]
    n_steps: int
    enable_depth_supervision: bool
