"""
过滤原因标签器 v3.0

为被过滤的股票打上原因标签，支持审计和贡献度分析

使用方法:
    from src.reliability.filter_tagger import FilterTagger

    tagger = FilterTagger()
    reasons = tagger.tag(codes, scores, factors, price_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class FilterTagger:
    """过滤原因标签器"""

    # 标签定义
    TAGS = {
        'missing_factor': '因子数据缺失',
        'low_liquidity': '流动性不足',
        'suspended': '停牌',
        'limit_up': '涨停',
        'limit_down': '跌停',
        'st_stock': 'ST股票',
        'new_listing': '新股',
        'blacklisted': '黑名单',
        'missing_price': '价格缺失',
        'data_quality_issue': '数据质量问题',
        'below_threshold': '未达筛选阈值',
    }

    def __init__(self, config: Dict = None):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.filter_stats = {tag: 0 for tag in self.TAGS}

    def tag(self,
            all_codes: List[str],
            filtered_codes: List[str],
            scores: pd.Series = None,
            factors: pd.DataFrame = None,
            price_data: pd.DataFrame = None,
            amount_data: pd.Series = None,
            blacklist: Set[str] = None,
            min_adv: float = 10000000,
            min_listed_days: int = 120) -> Dict[str, List[str]]:
        """
        为被过滤的股票打标签

        Args:
            all_codes: 全部候选代码
            filtered_codes: 过滤后代码
            scores: 得分数据
            factors: 因子数据
            price_data: 价格数据
            amount_data: 成交额数据
            blacklist: 黑名单
            min_adv: 最小成交额
            min_listed_days: 最小上市天数

        Returns:
            {code: [tag1, tag2, ...]}
        """
        removed_codes = set(all_codes) - set(filtered_codes)
        reasons = {}

        for code in removed_codes:
            code_reasons = []

            # 检查因子缺失
            if factors is not None:
                if 'code' in factors.columns:
                    if code not in factors['code'].values:
                        code_reasons.append('missing_factor')
                else:
                    if code not in factors.index:
                        code_reasons.append('missing_factor')

            # 检查得分缺失
            if scores is not None:
                if code not in scores.index:
                    code_reasons.append('data_quality_issue')

            # 检查价格缺失
            if price_data is not None:
                if 'code' in price_data.columns:
                    if code not in price_data['code'].values:
                        code_reasons.append('missing_price')
                else:
                    if code not in price_data.index:
                        code_reasons.append('missing_price')

            # 检查流动性
            if amount_data is not None:
                adv = amount_data.get(code, 0)
                if adv < min_adv:
                    code_reasons.append('low_liquidity')

            # 检查黑名单
            if blacklist and code in blacklist:
                code_reasons.append('blacklisted')

            # 检查ST
            if code.startswith('ST') or code.startswith('*ST'):
                code_reasons.append('st_stock')

            # 如果没有找到具体原因，标记为below_threshold
            if not code_reasons:
                code_reasons.append('below_threshold')

            reasons[code] = code_reasons

            # 更新统计
            for tag in code_reasons:
                self.filter_stats[tag] = self.filter_stats.get(tag, 0) + 1

        return reasons

    def get_stats(self) -> Dict[str, int]:
        """获取过滤统计"""
        return self.filter_stats.copy()

    def get_summary(self) -> str:
        """获取过滤摘要"""
        total = sum(self.filter_stats.values())
        if total == 0:
            return "无过滤"

        summary = []
        for tag, count in sorted(self.filter_stats.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / total * 100
                desc = self.TAGS.get(tag, tag)
                summary.append(f"{desc}: {count}只 ({pct:.1f}%)")

        return "\n".join(summary)


def tag_filter_reasons(all_codes: List[str],
                       filtered_codes: List[str],
                       **kwargs) -> Dict[str, List[str]]:
    """
    便捷函数：打标签

    Args:
        all_codes: 全部候选代码
        filtered_codes: 过滤后代码
        **kwargs: 其他参数

    Returns:
        {code: [tag1, tag2, ...]}
    """
    tagger = FilterTagger()
    return tagger.tag(all_codes, filtered_codes, **kwargs)


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("过滤标签器测试")
    print("=" * 60)

    tagger = FilterTagger()

    all_codes = [f'{i:06d}' for i in range(1, 101)]
    filtered_codes = all_codes[:70]  # 过滤30只

    # 模拟数据
    scores = pd.Series(np.random.uniform(30, 90, 80), index=all_codes[:80])

    reasons = tagger.tag(all_codes, filtered_codes, scores=scores)

    print(f"\n被过滤股票数: {len(reasons)}")
    print(f"过滤统计:")
    print(tagger.get_summary())
