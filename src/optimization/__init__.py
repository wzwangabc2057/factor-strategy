"""优化模块"""

from .anti_fixation import (
    AntiFixationEngine,
    SoftDiversifyPenalty,
    AdaptiveThresholdAdjuster,
    diagnose_list_fixation,
    run_ablation_study
)

__all__ = [
    'AntiFixationEngine',
    'SoftDiversifyPenalty',
    'AdaptiveThresholdAdjuster',
    'diagnose_list_fixation',
    'run_ablation_study'
]
