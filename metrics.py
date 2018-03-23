import sys
class Metrics():
    def __init__(self, beta = 0.9):
        self.metrics_dict = {}
        self.beta = 0.9

    def update_metric(self, metric, value):
        if value is None:
            value = 0.
        if metric in self.metrics_dict:
            self.metrics_dict[metric] = self.beta * self.metrics_dict[metric] + (1. - self.beta) * value
        else:
            self.metrics_dict[metric] = float(value)

    def update_metrics(self, metrics, values):
        if values is None:
            values = [0.] * len(metrics)
        for metric, value in zip(metrics, values):
            self.update_metric(metric, value)

    def write_metrics(self, metrics = None):
        """This function writes out metrics_dict in certain formats for FloydHub Parser to Parse
        and generates figures, See https://docs.floydhub.com/guides/jobs/metrics_dict/ for more
        information"""
        if metrics is None:
            for metric in self.metrics_dict:
                sys.stdout.write('{"metric": "%s", "value": %f}\n' % (metric, self.metrics_dict[metric]))
        else:
            for metric in self.metrics_dict:
                if (metric in metrics) or (metric == metrics):
                    sys.stdout.write('{"metric": "%s", "value": %f}\n' % (metric, self.metrics_dict[metric]))

    def get_metrics_dict(self):
        return self.metrics_dict

    def get_metrics_names(self):
        return list(self.metrics_dict.keys)