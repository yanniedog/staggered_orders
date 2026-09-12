import ast
import unittest
from pathlib import Path

import numpy as np
from gui_charts import (ProfitDistributionChart, HistoricalTouchFrequencyChart,
                        RiskReturnProfileChart, TouchVsTimeChart,
                        FitQualityDashboardChart)


class ChartContractTests(unittest.TestCase):
    def chart(self, kind):
        return kind()

    def test_targets_are_percentages_not_realized_dollars(self):
        fig = self.chart(ProfitDistributionChart).create({'actual_profits': np.array([2., 5.])})
        self.assertEqual(list(fig.data[0].y), [2., 5.])
        self.assertEqual(fig.data[0].name, 'Target profit (%)')
        self.assertIn('%{y:.2f}%', fig.data[0].hovertemplate)
        self.assertNotIn('$', fig.data[0].hovertemplate)
        self.assertNotIn('Actual', fig.data[0].name)

    def test_modeled_fields_do_not_fabricate_observations_or_risk(self):
        modeled = {'buy_touch_probs': np.array([.8]), 'actual_profits': np.array([2.]),
                   'weibull_params': {'buy': {'theta': 1., 'p': 2.}}}
        for kind in [HistoricalTouchFrequencyChart, RiskReturnProfileChart,
                     TouchVsTimeChart, FitQualityDashboardChart]:
            with self.subTest(kind=kind):
                fig = self.chart(kind).create(modeled)
                self.assertEqual(len(fig.data), 0)
                self.assertTrue(fig.layout.annotations)

    def test_callback_definitions_are_unique(self):
        tree = ast.parse(Path(__file__).with_name('gui_app.py').read_text(encoding='utf-8'))
        gui = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'InteractiveLadderGUI')
        for name in ['update_all_visualizations', '_get_cached_kpis']:
            self.assertEqual(sum(isinstance(n, ast.FunctionDef) and n.name == name
                                 for n in gui.body), 1)


if __name__ == '__main__':
    unittest.main()
