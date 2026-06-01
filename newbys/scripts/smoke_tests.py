import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tests.test_api as api_tests
import tests.test_feature_builder as feature_tests
import tests.test_subjective_bayes as bayes_tests


def main():
    bayes_tests.test_strong_short_term_evidence_produces_higher_t1_probability()
    bayes_tests.test_market_cycle_changes_dynamic_prior_before_evidence()
    bayes_tests.test_result_explains_profit_and_loss_likelihoods()
    feature_tests.test_feature_builder_infers_evidence_categories_from_quote_snapshot()
    feature_tests.test_manual_evidence_overrides_inferred_categories()
    api_tests.test_analysis_api_returns_subjective_bayes_result()
    api_tests.test_manual_analysis_accepts_explicit_evidence()
    print("all smoke tests passed")


if __name__ == "__main__":
    main()
