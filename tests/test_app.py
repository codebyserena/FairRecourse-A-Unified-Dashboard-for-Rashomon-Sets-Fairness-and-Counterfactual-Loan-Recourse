import unittest

from app import app, load_data


class DashboardSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_load_data_reads_required_artifacts(self):
        data = load_data()
        self.assertIsNotNone(data)
        self.assertFalse(data['metrics'].empty)
        self.assertFalse(data['rashomon'].empty)
        self.assertFalse(data['applicants'].empty)

    def test_dashboard_routes_render(self):
        for route in ['/', '/rashomon', '/recourse', '/fairness', '/summary']:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'FairRecourse Dashboard', response.data)

    def test_download_model_metrics(self):
        response = self.client.get('/download/model-metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/csv', response.content_type)
        response.close()


if __name__ == '__main__':
    unittest.main()
