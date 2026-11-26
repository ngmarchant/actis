import numpy as np

from experiments.runners import ACTISRunner
from experiments.scenarios import Population


class TestACTISRunnerIntegration:
    def test_auto_resolution_small_dataset(self):
        rng = np.random.default_rng(999)
        N = 500
        scores = rng.beta(2, 5, size=N)
        labels = (scores > 0.3).astype(int)

        runner = ACTISRunner(
            name="test_actis_auto",
            conf_seq="asymptotic",
            adaptive=True,
            initial_sample_size="auto",
            min_positives="auto",
            v_0="auto",
        )

        pop = Population(scores=scores, labels=labels)
        trial_result = runner.run_trial(
            population=pop,
            gamma_R=0.8,
            gamma_P=0.8,
            delta=0.05,
            rng=rng,
        )

        assert trial_result.cost["oracle"]["num_calls"] > 0
        assert trial_result.cost["oracle"]["num_calls"] <= N
        assert 0.0 <= trial_result.recall <= 1.0
        assert 0.0 <= trial_result.precision <= 1.0
        assert trial_result.tau_upper is not None
        assert trial_result.tau_lower is not None


def test_actis_runner_asymptotic_protection_flag():
    runner_on = ACTISRunner(name="actis_on", enable_asymptotic_protection=True)
    runner_off = ACTISRunner(name="actis_off", enable_asymptotic_protection=False)
    assert runner_on.enable_asymptotic_protection is True
    assert runner_off.enable_asymptotic_protection is False
