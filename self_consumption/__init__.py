"""
Energy consumption analysis and visualization package.

This package provides tools for analyzing energy consumption patterns,
PV production simulation, and battery storage performance.
"""

__version__ = "0.1.0"

from .self_consumption_analysis import (
    simulate_day,
    simulate_year,
    summarize_result,
    typical_pv_day_profile,
    typical_consumption_profile,
    intervals_per_day,
    model_error_low,
    model_error_high,
    target_annual_load_kWh
)

__all__ = [
    'simulate_day',
    'simulate_year',
    'summarize_result',
    'typical_pv_day_profile',
    'typical_consumption_profile',
    'intervals_per_day',
    'model_error_low',
    'model_error_high',
    'target_annual_load_kWh',
    '__version__'
]
