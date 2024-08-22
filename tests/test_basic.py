import unittest
import pandas as pd
import sys
sys.path.insert(0, '/10 A KAI 2/week 0/EDA_PROJECT/')
from src.basic_stastic import BasicStas 
class Testbasic(unittest.TestCase):
        def setUp(self):
              data = {
                'Timestamp': ['8/9/2021 0:01', '8/9/2021 0:02', '8/9/2021 0:03'],
                'GHI': [-1.2, -1.1, -1.1],
                'DNI': [-0.2, -0.2, -0.2],
                'DHI' : [-1.1, -1.1, -1.1],
                'ModA': [0, 0, 0],
                'ModB': [0, 0, 0],
                'Tamb': [26.2, 26.2, 26.2],
                'RH': [93.4, 93.6, 93.7],
                'WS': [0, 0, 0.3],
                'WSgust': [0.4, 0, 1.1],
                'WSstdev': [0.1, 0, 0.5],
                'WD': [122.1, 0, 124.6],
                'WDstdev': [0, 0, 1.5],
                'BP': [998, 998, 997],
                'Cleaning': [0, 0, 0],
                'Precipitation': [0, 0, 0],
                'TModA': [26.3, 26.3, 26.4],
                'TModB': [26.2, 26.2, 26.2],
                'Comments': [None, None, None]
        }
              self.df = pd.DataFrame(data)
              self.eda = BasicStas(self.df)
        def test_basic_info(self):
               result = self.eda.basic_stats()
                 # Assert the expected output
               expected_stats = self.df.describe()
               self.assertTrue(result.equals(expected_stats))
               