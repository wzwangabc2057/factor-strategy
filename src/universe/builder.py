"""
动态股票池构建器 v3.0

解决"固定名单"问题：
- 支持多种股票池来源：index/liquidity_top/file
- 流动性筛选、上市天数过滤
- 白名单/黑名单支持
- 降级模式可见性

使用方法:
    from src.universe.builder import UniverseBuilder

    builder = UniverseBuilder(config_path='config/universe.yaml')
    universe = builder.build(price_df, date='2024-01-01')
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class UniverseBuilder:
    """动态股票池构建器"""

    def __init__(self, config_path: str = None, config: Dict = None):
        """
        初始化构建器

        Args:
            config_path: 配置文件路径
            config: 直接传入配置字典
        """
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f).get('universe', {})
        else:
            self.config = self._default_config()

        # 解析配置
        self.source = self.config.get('source', 'liquidity_top')
        self.liquidity_config = self.config.get('liquidity', {})
        self.index_config = self.config.get('index', {})
        self.file_config = self.config.get('file', {})
        self.filters = self.config.get('filters', {})
        self.whitelist = self.config.get('whitelist', [])
        self.blacklist = self.config.get('blacklist', [])

        # 降级状态跟踪
        self.degradation_events = []
        self.is_degraded = False
        self.degradation_reason = None

        # 统计信息
        self.stats = {
            'universe_size': 0,
            'source': self.source,
            'is_degraded': False,
            'degradation_reason': None,
            'filtered_by_liquidity': 0,
            'filtered_by_listed_days': 0,
            'filtered_by_blacklist': 0,
        }

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'source': 'liquidity_top',
            'liquidity': {
                'top_n': 800,
                'adv_window': 20,
                'min_adv': 10000000,
            },
            'filters': {
                'min_listed_days': 120,
                'exclude_st': True,
            },
            'whitelist': [],
            'blacklist': [],
            'degradation': {
                'log_warning': True,
                'output_metric': True,
            }
        }

    def build(self,
              price_df: pd.DataFrame,
              date: str = None,
              amount_df: pd.DataFrame = None,
              list_date_df: pd.DataFrame = None,
              custom_blacklist: List[str] = None) -> List[str]:
        """
        构建股票池

        Args:
            price_df: 价格数据，需包含 date, code, close 列
            date: 目标日期（默认使用最新日期）
            amount_df: 成交额数据，需包含 date, code, amount 列
            list_date_df: 上市日期数据，需包含 code, list_date 列
            custom_blacklist: 额外黑名单

        Returns:
            股票代码列表
        """
        self.degradation_events = []
        self.is_degraded = False
        self.degradation_reason = None

        # 重置统计
        self.stats = {
            'universe_size': 0,
            'source': self.source,
            'is_degraded': False,
            'degradation_reason': None,
            'filtered_by_liquidity': 0,
            'filtered_by_listed_days': 0,
            'filtered_by_blacklist': 0,
        }

        # 确定目标日期
        if date is None:
            date = price_df['date'].max()

        # 根据来源构建初始池
        if self.source == 'file':
            universe = self._build_from_file(date)
        elif self.source == 'index':
            universe = self._build_from_index(date, price_df)
        else:  # liquidity_top
            universe = self._build_from_liquidity(date, price_df, amount_df)

        # 应用通用过滤
        universe = self._apply_filters(universe, date, list_date_df)

        # 应用黑名单
        all_blacklist = set(self.blacklist)
        if custom_blacklist:
            all_blacklist.update(custom_blacklist)

        before_blacklist = len(universe)
        universe = [c for c in universe if c not in all_blacklist]
        self.stats['filtered_by_blacklist'] = before_blacklist - len(universe)

        # 应用白名单（白名单股票始终包含）
        if self.whitelist:
            universe = list(set(universe) | set(self.whitelist))

        # 更新统计
        self.stats['universe_size'] = len(universe)
        self.stats['is_degraded'] = self.is_degraded
        self.stats['degradation_reason'] = self.degradation_reason

        if len(universe) == 0:
            logger.warning(f"[Universe] 股票池为空! date={date}, source={self.source}")
            self._record_degradation('empty_universe', '股票池为空')

        return universe

    def _build_from_file(self, date: str) -> List[str]:
        """从文件构建股票池"""
        if not self.file_config.get('allow_file_pool', False):
            logger.warning("[Universe] 文件池模式未启用 (allow_file_pool=false)")
            self._record_degradation('file_pool_disabled', 'allow_file_pool=false')
            # 降级到 liquidity_top
            return []

        file_path = self.file_config.get('path', 'data/universe.csv')
        if not os.path.exists(file_path):
            logger.warning(f"[Universe] 文件不存在: {file_path}")
            self._record_degradation('file_not_found', file_path)
            return []

        try:
            df = pd.read_csv(file_path)
            required_cols = self.file_config.get('required_columns', ['code'])
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"[Universe] 文件缺少必需列: {col}")
                    self._record_degradation('missing_column', col)
                    return []

            codes = df['code'].astype(str).str.zfill(6).tolist()
            logger.info(f"[Universe] 从文件加载 {len(codes)} 只股票: {file_path}")
            return codes
        except Exception as e:
            logger.warning(f"[Universe] 文件读取失败: {e}")
            self._record_degradation('file_read_error', str(e))
            return []

    def _build_from_index(self, date: str, price_df: pd.DataFrame) -> List[str]:
        """从指数成分构建股票池"""
        index_name = self.index_config.get('name', 'hs300')

        # 检查是否有指数成分数据
        # 这里预留接口，实际需要指数成分数据源
        has_index_data = False  # 目前默认没有

        if not has_index_data:
            self._record_degradation('no_index_data', f'index={index_name}')
            logger.warning(f"[Universe] 无指数成分数据: {index_name}, 降级到 liquidity_top")
            # 降级处理
            return self._build_from_liquidity(date, price_df, None)

        # TODO: 实现指数成分获取
        return []

    def _build_from_liquidity(self,
                               date: str,
                               price_df: pd.DataFrame,
                               amount_df: pd.DataFrame = None) -> List[str]:
        """从流动性构建股票池"""
        top_n = self.liquidity_config.get('top_n', 800)
        adv_window = self.liquidity_config.get('adv_window', 20)
        min_adv = self.liquidity_config.get('min_adv', 10000000)

        # 获取日期范围内的数据
        dates = sorted(price_df['date'].unique())
        if date not in dates:
            # 找最近的日期
            dates_before = [d for d in dates if d <= date]
            if dates_before:
                date = dates_before[-1]
            else:
                date = dates[0]

        date_idx = dates.index(date)
        start_idx = max(0, date_idx - adv_window)
        window_dates = dates[start_idx:date_idx + 1]

        # 筛选窗口期数据
        window_df = price_df[price_df['date'].isin(window_dates)]

        if amount_df is not None and 'amount' in amount_df.columns:
            # 使用成交额数据
            window_amount = amount_df[amount_df['date'].isin(window_dates)]
            adv_df = window_amount.groupby('code')['amount'].mean().reset_index()
            adv_df.columns = ['code', 'adv']

            # 过滤最小成交额
            adv_df = adv_df[adv_df['adv'] >= min_adv]
            self.stats['filtered_by_liquidity'] = len(adv_df)

            # 取 Top N
            adv_df = adv_df.nlargest(top_n, 'adv')
            codes = adv_df['code'].astype(str).str.zfill(6).tolist()

            logger.info(f"[Universe] 流动性筛选: {len(codes)} 只股票 (adv_top_{top_n})")
            return codes

        else:
            # 降级：使用成交量或市值
            fallback_field = self.liquidity_config.get('fallback_field', 'volume')
            self._record_degradation('no_amount_data', f'fallback={fallback_field}')

            if fallback_field == 'volume' and 'volume' in window_df.columns:
                vol_df = window_df.groupby('code')['volume'].mean().reset_index()
                vol_df = vol_df.nlargest(top_n, 'volume')
                codes = vol_df['code'].astype(str).str.zfill(6).tolist()
                logger.warning(f"[Universe] 无成交额数据，降级使用成交量: {len(codes)} 只")
                return codes
            elif 'close' in window_df.columns:
                # 最终降级：使用收盘价排序（不合理但保证有输出）
                latest = window_df[window_df['date'] == date]
                latest = latest.nlargest(top_n, 'close')
                codes = latest['code'].astype(str).str.zfill(6).tolist()
                logger.warning(f"[Universe] 严重降级：使用价格排序: {len(codes)} 只")
                return codes
            else:
                logger.error("[Universe] 无法构建股票池：缺少必要数据")
                # 返回所有股票
                all_codes = price_df['code'].unique().tolist()
                return [str(c).zfill(6) for c in all_codes[:top_n]]

    def _apply_filters(self,
                        codes: List[str],
                        date: str,
                        list_date_df: pd.DataFrame = None) -> List[str]:
        """应用通用过滤"""
        if not codes:
            return codes

        filtered_codes = set(codes)

        # 上市天数过滤
        min_listed_days = self.filters.get('min_listed_days', 0)
        if min_listed_days > 0 and list_date_df is not None:
            date_dt = pd.to_datetime(date)
            list_date_df['list_date'] = pd.to_datetime(list_date_df['list_date'])
            list_date_df['listed_days'] = (date_dt - list_date_df['list_date']).dt.days

            valid_codes = set(list_date_df[
                list_date_df['listed_days'] >= min_listed_days
            ]['code'].astype(str).str.zfill(6))

            before_count = len(filtered_codes)
            filtered_codes = filtered_codes & valid_codes
            self.stats['filtered_by_listed_days'] = before_count - len(filtered_codes)

        # ST过滤（预留，需要ST状态数据）
        if self.filters.get('exclude_st', False):
            # TODO: 实现ST过滤
            pass

        return list(filtered_codes)

    def _record_degradation(self, reason: str, detail: str = None):
        """记录降级事件"""
        self.is_degraded = True
        self.degradation_reason = reason

        event = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'detail': detail
        }
        self.degradation_events.append(event)

        if self.config.get('degradation', {}).get('log_warning', True):
            logger.warning(f"[Universe] 降级: {reason} - {detail}")

    def get_stats(self) -> Dict:
        """获取构建统计"""
        return self.stats.copy()

    def get_degradation_events(self) -> List[Dict]:
        """获取降级事件列表"""
        return self.degradation_events.copy()


def build_universe(price_df: pd.DataFrame,
                   config_path: str = 'config/universe.yaml',
                   **kwargs) -> Tuple[List[str], Dict]:
    """
    便捷函数：构建股票池

    Args:
        price_df: 价格数据
        config_path: 配置路径
        **kwargs: 传递给 UniverseBuilder.build 的参数

    Returns:
        (股票列表, 统计信息)
    """
    builder = UniverseBuilder(config_path=config_path)
    universe = builder.build(price_df, **kwargs)
    return universe, builder.get_stats()


def get_universe_stats(builder: UniverseBuilder) -> Dict:
    """获取股票池统计（用于报告）"""
    stats = builder.get_stats()
    stats['degradation_events'] = builder.get_degradation_events()
    return stats


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("Universe Builder 测试")
    print("=" * 60)

    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30).astype(str)
    codes = [f'{i:06d}' for i in range(1, 1001)]

    data = []
    for date in dates:
        for code in codes:
            data.append({
                'date': date,
                'code': code,
                'close': np.random.uniform(10, 100),
                'amount': np.random.uniform(1e7, 1e9)
            })

    price_df = pd.DataFrame(data)

    # 构建
    builder = UniverseBuilder(config_path='config/universe.yaml')
    universe = builder.build(price_df, date='2024-01-30')

    print(f"股票池大小: {len(universe)}")
    print(f"统计信息: {builder.get_stats()}")
