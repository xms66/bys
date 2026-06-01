import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tests.test_api as api_tests
import tests.test_feature_builder as feature_tests
import tests.test_llm_advisor as llm_tests
import tests.test_plan_api as plan_api_tests
import tests.test_subjective_bayes as bayes_tests
import tests.test_tickerlab_data_source as tickerlab_tests
import tests.test_trade_planner as trade_planner_tests


def main():
    bayes_tests.test_strong_short_term_evidence_produces_higher_t1_probability()
    bayes_tests.test_market_cycle_changes_dynamic_prior_before_evidence()
    bayes_tests.test_result_explains_profit_and_loss_likelihoods()
    feature_tests.test_feature_builder_infers_evidence_categories_from_quote_snapshot()
    feature_tests.test_manual_evidence_overrides_inferred_categories()
    api_tests.test_analysis_api_returns_subjective_bayes_result()
    api_tests.test_manual_analysis_accepts_explicit_evidence()
    api_tests.test_analysis_defaults_to_top5_and_includes_llm_advice()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        llm_tests.test_load_env_file_reads_freechat_config_without_exporting_secret(Path(tmp))
    llm_tests.test_build_llm_payload_uses_bayesian_system_prompt_and_top5_inputs()
    llm_tests.test_llm_advisor_posts_openai_compatible_request()
    tickerlab_tests.test_tickerlab_hot_rank_parses_top_entries_and_preserves_rank()
    tickerlab_tests.test_tickerlab_source_uses_quote_data_and_returns_ranked_snapshots()
    trade_planner_tests.test_next_trade_dates_use_calendar_days_for_initial_version()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        trade_planner_tests.test_store_creates_plan_and_today_actions(path)
        trade_planner_tests.test_today_actions_roll_sell_then_buy(path)
        trade_planner_tests.test_mark_execution_updates_buy_and_sell_prices(path)
        plan_api_tests.test_generate_plan_saves_llm_decision(path)
        plan_api_tests.test_today_actions_and_mark_execution_api(path)
    print("all smoke tests passed")


if __name__ == "__main__":
    main()
