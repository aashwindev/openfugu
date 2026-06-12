"""Fugu fast-path router package."""

from openfugu.router.inference import FuguRouter, RouteDecision
from openfugu.router.multistep import MultiStepRouter, MultiStepResult

__all__ = ["FuguRouter", "RouteDecision", "MultiStepRouter", "MultiStepResult"]
