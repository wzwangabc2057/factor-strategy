"""
配置验证器 v3.0 (Gate-1)

验证YAML配置文件的:
- 必需字段存在
- 字段值范围合法
- 逻辑一致性

使用方法:
    from src.config.validator import ConfigValidator, validate_config_file

    # 验证单个配置文件
    errors = validate_config_file('config/strategy_params.yaml')

    # 使用验证器类
    validator = ConfigValidator()
    errors = validator.validate_strategy_config(config_dict)
"""

import os
import yaml
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证错误"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"配置验证失败: {len(errors)} 个错误")


class ConfigValidator:
    """配置验证器"""

    # 策略配置验证规则
    STRATEGY_RULES = {
        'required_fields': [
            'strategy.name',
            'strategy.version',
            'factors.weights',
            'rebalance.frequency',
            'rebalance.turnover_limit',
            'constraints.min_weight',
            'constraints.max_position_count',
        ],
        'value_ranges': {
            'rebalance.turnover_limit': (0.0, 0.20, "月换手上限应在0-20%之间"),
            'constraints.min_weight': (0.0, 0.05, "最小权重应在0-5%之间"),
            'constraints.max_position_count': (10, 200, "最大持仓数应在10-200之间"),
            'weight_tilt.top_10_pct_boost': (1.0, 2.0, "Top10%加成应在1.0-2.0之间"),
            'weight_tilt.bottom_10_pct_penalty': (0.0, 1.0, "Bottom10%低配应在0-1.0之间"),
            'reversal.roe_threshold': (0.0, 20.0, "ROE阈值应在0-20%之间"),
        },
        'factor_weights_sum': {
            'field': 'factors.weights',
            'min': 0.8,
            'max': 1.2,
            'message': "因子权重和应在0.8-1.2之间 (允许±20%误差)"
        }
    }

    # 风控配置验证规则
    RISK_RULES = {
        'required_fields': [
            'risk_control.enabled',
            'risk_control.tier1_trend.action.position_ratio',
            'risk_control.tier1_trend.action.max_single_weight',
            'risk_control.tier2_volatile.action.position_ratio',
            'risk_control.tier2_volatile.action.max_single_weight',
            'risk_control.tier3_crash.action.position_ratio',
            'risk_control.tier3_crash.action.max_single_weight',
            'risk_control.tier3_crash.action.cash_ratio_min',
        ],
        'value_ranges': {
            'risk_control.tier1_trend.action.position_ratio': (0.8, 1.0, "Tier1仓位比例应在80-100%之间"),
            'risk_control.tier1_trend.action.max_single_weight': (0.05, 0.15, "Tier1单股上限应在5-15%之间"),
            'risk_control.tier1_trend.action.cash_ratio_min': (0.0, 0.10, "Tier1现金比例应在0-10%之间"),
            'risk_control.tier2_volatile.action.position_ratio': (0.6, 0.9, "Tier2仓位比例应在60-90%之间"),
            'risk_control.tier2_volatile.action.max_single_weight': (0.04, 0.12, "Tier2单股上限应在4-12%之间"),
            'risk_control.tier2_volatile.action.cash_ratio_min': (0.05, 0.20, "Tier2现金比例应在5-20%之间"),
            'risk_control.tier3_crash.action.position_ratio': (0.3, 0.7, "Tier3仓位比例应在30-70%之间"),
            'risk_control.tier3_crash.action.max_single_weight': (0.02, 0.08, "Tier3单股上限应在2-8%之间"),
            'risk_control.tier3_crash.action.cash_ratio_min': (0.30, 0.60, "Tier3现金比例应在30-60%之间"),
            'turnover_control.monthly_limit': (0.0, 0.15, "月换手上限应在0-15%之间"),
        },
        'logic_checks': [
            {
                'name': 'tier_position_decreasing',
                'condition': 'tier1_action_position >= tier2_action_position >= tier3_action_position',
                'message': "仓位比例应递减: Tier1 >= Tier2 >= Tier3"
            },
            {
                'name': 'tier_max_weight_decreasing',
                'condition': 'tier1_max_weight >= tier2_max_weight >= tier3_max_weight',
                'message': "单股上限应递减: Tier1 >= Tier2 >= Tier3"
            },
            {
                'name': 'tier_cash_increasing',
                'condition': 'tier1_cash <= tier2_cash <= tier3_cash',
                'message': "现金比例应递增: Tier1 <= Tier2 <= Tier3"
            }
        ]
    }

    # 成本模型配置验证规则
    COST_MODEL_RULES = {
        'required_fields': [
            'cost_model.enabled',
            'cost_model.base_cost.commission.buy',
            'cost_model.base_cost.commission.sell',
            'cost_model.base_cost.slippage.default',
            'cost_model.impact_cost.enabled',
        ],
        'value_ranges': {
            'cost_model.base_cost.commission.buy': (0.0, 0.001, "买入佣金应在0-0.1%之间"),
            'cost_model.base_cost.commission.sell': (0.0, 0.003, "卖出佣金应在0-0.3%之间"),
            'cost_model.base_cost.slippage.default': (0.0001, 0.01, "默认滑点应在0.01%-1%之间"),
            'cost_model.impact_cost.coefficient': (0.0, 1.0, "冲击成本系数应在0-1之间"),
        },
        'min_values': {
            'cost_model.base_cost.slippage.default': {
                'min': 0.001,
                'message': "滑点必须>=0.1% (红线要求)"
            }
        },
        'logic_checks': [
            {
                'name': 'sell_cost_higher_than_buy',
                'condition': 'commission_sell > commission_buy',
                'message': "卖出佣金应高于买入(含印花税)"
            }
        ]
    }

    def __init__(self, strict: bool = True):
        """
        初始化验证器

        Args:
            strict: 严格模式，验证失败抛出异常
        """
        self.strict = strict
        self.errors = []

    def _get_nested_value(self, config: Dict, path: str, default=None) -> Any:
        """获取嵌套字典值"""
        keys = path.split('.')
        value = config
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def _validate_required_fields(self, config: Dict, rules: Dict) -> List[str]:
        """验证必需字段"""
        errors = []
        for field in rules.get('required_fields', []):
            value = self._get_nested_value(config, field)
            if value is None:
                errors.append(f"缺少必需字段: {field}")
        return errors

    def _validate_value_ranges(self, config: Dict, rules: Dict) -> List[str]:
        """验证值范围"""
        errors = []
        for field, (min_val, max_val, message) in rules.get('value_ranges', {}).items():
            value = self._get_nested_value(config, field)
            if value is not None:
                if not (min_val <= value <= max_val):
                    errors.append(f"{field}={value}: {message}")
        return errors

    def _validate_min_values(self, config: Dict, rules: Dict) -> List[str]:
        """验证最小值"""
        errors = []
        for field, rule in rules.get('min_values', {}).items():
            value = self._get_nested_value(config, field)
            if value is not None:
                min_val = rule['min']
                message = rule.get('message', f"{field}应>={min_val}")
                if value < min_val:
                    errors.append(f"{field}={value}: {message}")
        return errors

    def _validate_logic_checks(self, config: Dict, rules: Dict) -> List[str]:
        """验证逻辑一致性"""
        errors = []
        for check in rules.get('logic_checks', []):
            name = check['name']
            message = check['message']

            # 简化的逻辑检查
            if name == 'tier_position_decreasing':
                tier1 = self._get_nested_value(config, 'risk_control.tier1_trend.action.position_ratio', 1.0)
                tier2 = self._get_nested_value(config, 'risk_control.tier2_volatile.action.position_ratio', 1.0)
                tier3 = self._get_nested_value(config, 'risk_control.tier3_crash.action.position_ratio', 1.0)
                if not (tier1 >= tier2 >= tier3):
                    errors.append(message)

            elif name == 'tier_max_weight_decreasing':
                tier1 = self._get_nested_value(config, 'risk_control.tier1_trend.action.max_single_weight', 1.0)
                tier2 = self._get_nested_value(config, 'risk_control.tier2_volatile.action.max_single_weight', 1.0)
                tier3 = self._get_nested_value(config, 'risk_control.tier3_crash.action.max_single_weight', 1.0)
                if not (tier1 >= tier2 >= tier3):
                    errors.append(message)

            elif name == 'tier_cash_increasing':
                tier1 = self._get_nested_value(config, 'risk_control.tier1_trend.action.cash_ratio_min', 0.0)
                tier2 = self._get_nested_value(config, 'risk_control.tier2_volatile.action.cash_ratio_min', 0.0)
                tier3 = self._get_nested_value(config, 'risk_control.tier3_crash.action.cash_ratio_min', 0.0)
                if not (tier1 <= tier2 <= tier3):
                    errors.append(message)

            elif name == 'sell_cost_higher_than_buy':
                buy = self._get_nested_value(config, 'cost_model.base_cost.commission.buy', 0)
                sell = self._get_nested_value(config, 'cost_model.base_cost.commission.sell', 0)
                if sell <= buy:
                    errors.append(message)

        return errors

    def _validate_factor_weights_sum(self, config: Dict, rules: Dict) -> List[str]:
        """验证因子权重和"""
        errors = []
        sum_rule = rules.get('factor_weights_sum')
        if sum_rule:
            weights = self._get_nested_value(config, sum_rule['field'])
            if weights and isinstance(weights, dict):
                total = sum(weights.values())
                min_sum = sum_rule['min']
                max_sum = sum_rule['max']
                if not (min_sum <= total <= max_sum):
                    errors.append(f"{sum_rule['field']}总和={total:.2f}: {sum_rule['message']}")
        return errors

    def validate_strategy_config(self, config: Dict) -> List[str]:
        """验证策略配置"""
        errors = []
        errors.extend(self._validate_required_fields(config, self.STRATEGY_RULES))
        errors.extend(self._validate_value_ranges(config, self.STRATEGY_RULES))
        errors.extend(self._validate_factor_weights_sum(config, self.STRATEGY_RULES))
        self.errors.extend(errors)
        return errors

    def validate_risk_config(self, config: Dict) -> List[str]:
        """验证风控配置"""
        errors = []
        errors.extend(self._validate_required_fields(config, self.RISK_RULES))
        errors.extend(self._validate_value_ranges(config, self.RISK_RULES))
        errors.extend(self._validate_logic_checks(config, self.RISK_RULES))
        self.errors.extend(errors)
        return errors

    def validate_cost_config(self, config: Dict) -> List[str]:
        """验证成本模型配置"""
        errors = []
        errors.extend(self._validate_required_fields(config, self.COST_MODEL_RULES))
        errors.extend(self._validate_value_ranges(config, self.COST_MODEL_RULES))
        errors.extend(self._validate_min_values(config, self.COST_MODEL_RULES))
        errors.extend(self._validate_logic_checks(config, self.COST_MODEL_RULES))
        self.errors.extend(errors)
        return errors

    def validate_config_file(self, file_path: str, config_type: str) -> List[str]:
        """
        验证配置文件

        Args:
            file_path: 配置文件路径
            config_type: 配置类型 ('strategy', 'risk', 'cost')

        Returns:
            错误列表
        """
        if not os.path.exists(file_path):
            return [f"配置文件不存在: {file_path}"]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return [f"YAML解析错误: {e}"]

        if config_type == 'strategy':
            return self.validate_strategy_config(config)
        elif config_type == 'risk':
            return self.validate_risk_config(config)
        elif config_type == 'cost':
            return self.validate_cost_config(config)
        else:
            return [f"未知配置类型: {config_type}"]

    def get_all_errors(self) -> List[str]:
        """获取所有错误"""
        return self.errors

    def clear_errors(self):
        """清空错误"""
        self.errors = []

    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0

    def raise_if_strict(self):
        """严格模式下抛出异常"""
        if self.strict and self.errors:
            raise ConfigValidationError(self.errors)


def validate_config_file(file_path: str, config_type: str, strict: bool = True) -> List[str]:
    """
    便捷函数: 验证配置文件

    Args:
        file_path: 配置文件路径
        config_type: 配置类型 ('strategy', 'risk', 'cost')
        strict: 严格模式

    Returns:
        错误列表
    """
    validator = ConfigValidator(strict=strict)
    errors = validator.validate_config_file(file_path, config_type)
    if strict and errors:
        raise ConfigValidationError(errors)
    return errors


def validate_all_configs(config_dir: str = 'config', strict: bool = True) -> Dict[str, List[str]]:
    """
    验证所有配置文件

    Args:
        config_dir: 配置目录
        strict: 严格模式

    Returns:
        {配置文件名: 错误列表}
    """
    validator = ConfigValidator(strict=strict)

    results = {}
    config_files = [
        ('strategy_params.yaml', 'strategy'),
        ('risk_control.yaml', 'risk'),
        ('cost_model.yaml', 'cost'),
    ]

    for filename, config_type in config_files:
        file_path = os.path.join(config_dir, filename)
        if os.path.exists(file_path):
            errors = validator.validate_config_file(file_path, config_type)
            results[filename] = errors
        else:
            results[filename] = [f"配置文件不存在: {file_path}"]

    if strict and validator.has_errors():
        validator.raise_if_strict()

    return results


if __name__ == '__main__':
    # 测试
    import sys

    print("=" * 60)
    print("配置验证器测试")
    print("=" * 60)

    results = validate_all_configs(config_dir='config', strict=False)

    for filename, errors in results.items():
        print(f"\n{filename}:")
        if errors:
            for err in errors:
                print(f"  ❌ {err}")
        else:
            print("  ✓ 验证通过")

    # 统计
    total_errors = sum(len(errs) for errs in results.values())
    print(f"\n总计: {total_errors} 个错误")
