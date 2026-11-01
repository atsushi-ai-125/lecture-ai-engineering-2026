"""比較実験のテスト。すべてのGeminiクライアントをモックに置き換える。"""

import csv
from dataclasses import replace
import json
from unittest.mock import MagicMock, Mock

import pytest
from google.genai import errors, types
from pydantic import ValidationError

from app import benchmark


@pytest.fixture
def experiment():
    return benchmark.Experiment.model_validate_json(benchmark.DEFAULT_CONFIG.read_text())


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text="回答です")]),
            finish_reason="STOP",
        )],
        model_version="test-version",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=11, candidates_token_count=17,
            thoughts_token_count=5, total_token_count=33,
        ),
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr(benchmark.genai, "Client", constructor)
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key")
    monkeypatch.setattr(benchmark.time, "sleep", Mock())
    return constructor, client


def test_six_independent_requests_and_csv(experiment, fake_client, tmp_path):
    constructor, client = fake_client
    output = tmp_path / "results.csv"

    results = benchmark.run_experiment(experiment, output, "test-secret-key")

    calls = client.models.generate_content.call_args_list
    assert len(results) == len(calls) == 6
    assert [(call.kwargs["model"], call.kwargs["contents"]) for call in calls] == [
        (model, question.text)
        for question in experiment.questions for model in experiment.models
    ]
    assert all(call.kwargs["config"] == benchmark.generation_config(experiment) for call in calls)
    assert benchmark.time.sleep.call_count == 5
    benchmark.time.sleep.assert_called_with(15)
    http = constructor.call_args.kwargs["http_options"]
    assert http.retry_options.attempts == 1
    assert http.timeout == 120_000
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert rows[0]["answer"] == "回答です"
    assert rows[0]["input_tokens"] == "11"
    assert rows[0]["output_tokens"] == "17"
    assert rows[0]["thinking_tokens"] == "5"
    assert rows[0]["total_tokens"] == "33"
    assert rows[0]["model_version"] == "test-version"
    assert rows[0]["finish_reason"] == "STOP"
    assert rows[0]["error"] == ""


def test_timer_and_missing_metadata(experiment, fake_client, monkeypatch, tmp_path):
    _, client = fake_client
    experiment.models = experiment.models[:1]
    experiment.questions = experiment.questions[:1]
    client.models.generate_content.return_value.usage_metadata = None
    clock = Mock(side_effect=[100.0, 102.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", clock)

    rows = benchmark.run_experiment(experiment, tmp_path / "results.csv", "test-secret-key")

    assert clock.call_count == 2
    assert rows[0].elapsed_seconds == 2.5
    assert all(getattr(rows[0], field) is None for field in benchmark.TOKEN_FIELDS)
    with (tmp_path / "results.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["input_tokens"] == row["total_tokens"] == ""


def test_partial_metadata_does_not_infer_total(experiment, fake_client, tmp_path):
    _, client = fake_client
    client.models.generate_content.return_value.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10, candidates_token_count=0,
    )
    results = benchmark.run_experiment(experiment, tmp_path / "partial.csv", "test-secret-key")
    assert results[0].output_tokens == 0
    assert results[0].total_tokens is None
    assert results[0].thinking_tokens is None


def test_csv_preserves_multiline_answer_and_redacts_key(experiment, fake_client, tmp_path, capsys):
    _, client = fake_client
    experiment.models = experiment.models[:1]
    experiment.questions = experiment.questions[:1]
    client.models.generate_content.return_value.candidates[0].content.parts[0].text = (
        '回答, "引用"\nキー: test-secret-key'
    )
    path = tmp_path / "multiline.csv"
    benchmark.run_experiment(experiment, path, "test-secret-key")
    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["answer"] == '回答, "引用"\nキー: [REDACTED]'
    assert "test-secret-key" not in capsys.readouterr().out


@pytest.mark.parametrize("code", [429, 403, 503])
def test_api_error_saved_and_stops_without_retry(experiment, fake_client, tmp_path, capsys, code):
    _, client = fake_client
    client.models.generate_content.side_effect = [
        client.models.generate_content.return_value,
        errors.APIError(code, {
            "status": "RESOURCE_EXHAUSTED" if code == 429 else "ERROR",
            "message": "failure with test-secret-key",
        }),
    ]
    path = tmp_path / "error.csv"

    results = benchmark.run_experiment(experiment, path, "test-secret-key")

    assert client.models.generate_content.call_count == 2
    assert len(results) == 2
    assert results[1].status == "api_error"
    assert results[1].error_code == code
    assert results[1].input_tokens is None
    assert results[1].elapsed_seconds >= 0
    assert "[REDACTED]" in results[1].error
    assert "test-secret-key" not in path.read_text()
    assert "test-secret-key" not in capsys.readouterr().out
    with path.open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 2


def test_transport_error_does_not_leak_details(experiment, fake_client, tmp_path):
    _, client = fake_client
    client.models.generate_content.side_effect = TimeoutError("URL containing test-secret-key")
    rows = benchmark.run_experiment(experiment, tmp_path / "error.csv", "test-secret-key")
    assert len(rows) == 1
    assert rows[0].status == "client_error"
    assert "TimeoutError" in rows[0].error
    assert "test-secret-key" not in rows[0].error


def test_repeat_and_interrupt_keep_completed_rows(experiment, fake_client, tmp_path):
    _, client = fake_client
    experiment.repeat = 2
    client.models.generate_content.side_effect = [
        *[client.models.generate_content.return_value] * 7, KeyboardInterrupt(),
    ]
    path = tmp_path / "interrupted.csv"
    rows = benchmark.run_experiment(experiment, path, "test-secret-key")
    assert len(rows) == 7
    assert rows[-1].repeat == 2
    with path.open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 7


def test_empty_response_and_summary_exclude_missing_and_errors(experiment, fake_client, tmp_path, capsys):
    _, client = fake_client
    client.models.generate_content.return_value = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"),
    )
    results = benchmark.run_experiment(experiment, tmp_path / "empty.csv", "test-secret-key")
    assert results[0].status == "empty"
    assert results[0].finish_reason == "SAFETY"
    capsys.readouterr()
    success = replace(results[0], status="ok", elapsed_seconds=2, input_tokens=10, output_tokens=0)
    missing = replace(success, elapsed_seconds=4, input_tokens=None)
    failed = replace(success, status="api_error", elapsed_seconds=100, input_tokens=1000)
    benchmark.print_summary([success, missing, failed, results[0]])
    output = capsys.readouterr().out
    assert "成功=2/4" in output
    assert "elapsed_seconds=3.000 (n=2)" in output
    assert "input_tokens=10.000 (n=1)" in output
    assert "output_tokens=0.000 (n=2)" in output
    assert "total_tokens=未取得 (n=0)" in output


def test_preview_never_asks_for_key_or_constructs_client(fake_client, monkeypatch, capsys):
    constructor, _ = fake_client
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr("sys.argv", ["app.benchmark"])
    monkeypatch.setattr(benchmark, "get_api_key", Mock(side_effect=AssertionError("key read")))
    benchmark.main()
    constructor.assert_not_called()
    assert "最大6リクエスト" in capsys.readouterr().out


def test_default_cli_displays_and_saves_all_six_distinct_answers(
    experiment, fake_client, monkeypatch, tmp_path, capsys,
):
    """実際のCLI入口から、各モデルの回答・計測・平均・CSVまで確認する。"""
    _, client = fake_client
    responses = []
    expected = []
    for question in experiment.questions:
        for model in experiment.models:
            answer = f"テスト回答: {model} / {question.id}"
            response = client.models.generate_content.return_value.model_copy(deep=True)
            response.candidates[0].content.parts[0].text = answer
            responses.append(response)
            expected.append((model, question.id, answer))
    client.models.generate_content.side_effect = responses
    # 各リクエストを1秒とする。実測値ではなく計測・平均表示の検証用。
    monkeypatch.setattr(benchmark.time, "perf_counter", Mock(side_effect=range(12)))
    path = tmp_path / "six_answers.csv"
    monkeypatch.setattr("sys.argv", ["app.benchmark", "--run", "--output", str(path)])
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    benchmark.main()

    output = capsys.readouterr().out
    assert "完了: 6/6リクエスト" in output
    for index, (model, question_id, answer) in enumerate(expected, start=1):
        assert f"[{index}] {model} / {question_id} / ok" in output
        assert output.count(answer) == 1
    assert output.count("時間=1.000秒") == 6
    assert "平均（model別" in output
    assert "平均（question_id別" in output
    assert "平均（model × question_id別" in output
    assert "elapsed_seconds=1.000 (n=3)" in output
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(row["model"], row["question_id"], row["answer"]) for row in rows] == expected
    assert all(float(row["elapsed_seconds"]) == 1.0 for row in rows)
    assert all(row["total_tokens"] == "33" for row in rows)
    assert client.models.generate_content.call_count == 6


def test_cli_reports_partial_results_when_second_model_hits_rate_limit(
    fake_client, monkeypatch, tmp_path, capsys,
):
    _, client = fake_client
    client.models.generate_content.side_effect = [
        client.models.generate_content.return_value,
        errors.APIError(429, {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}),
    ]
    path = tmp_path / "rate_limit.csv"
    monkeypatch.setattr("sys.argv", ["app.benchmark", "--run", "--output", str(path)])
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert output.count("回答です") == 1
    assert "RESOURCE_EXHAUSTED" in output
    assert "完了: 2/6リクエスト" in output
    assert client.models.generate_content.call_count == 2
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["status"] for row in rows] == ["ok", "api_error"]
    assert rows[1]["error_code"] == "429"


@pytest.mark.parametrize("answer", ["no", "", EOFError(), KeyboardInterrupt()])
def test_declining_confirmation_never_calls_api(fake_client, monkeypatch, answer):
    constructor, _ = fake_client
    monkeypatch.setattr("sys.argv", ["app.benchmark", "--run"])
    mock_input = Mock(side_effect=answer) if isinstance(answer, BaseException) else Mock(return_value=answer)
    monkeypatch.setattr("builtins.input", mock_input)
    benchmark.main()
    constructor.assert_not_called()


def test_confirmed_cli_overrides_and_thinking_setting(fake_client, monkeypatch, tmp_path, experiment):
    _, client = fake_client
    experiment.thinking_level = "LOW"
    config = tmp_path / "config.json"
    config.write_text(experiment.model_dump_json())
    output = tmp_path / "results.csv"
    monkeypatch.setattr("sys.argv", [
        "app.benchmark", "--config", str(config), "--run", "--models", "test-model",
        "--repeat", "2", "--interval", "0", "--output", str(output),
    ])
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    benchmark.main()
    assert client.models.generate_content.call_count == 6
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["config"].thinking_config.thinking_level == "LOW"
    benchmark.time.sleep.assert_called_with(0)


def test_existing_csv_is_not_overwritten(experiment, fake_client, tmp_path):
    constructor, _ = fake_client
    path = tmp_path / "existing.csv"
    path.write_text("previous experiment")
    with pytest.raises(FileExistsError):
        benchmark.run_experiment(experiment, path, "test-secret-key")
    assert path.read_text() == "previous experiment"
    constructor.assert_not_called()


@pytest.mark.parametrize("questions", [["桃の特徴は？"], ["日本の首都は？", "過学習とは？", "3×7はいくつ？"]])
def test_interactive_questions_reach_each_model_and_csv(
    experiment, fake_client, monkeypatch, tmp_path, capsys, questions,
):
    _, client = fake_client
    output = tmp_path / "interactive.csv"
    monkeypatch.setattr("sys.argv", [
        "app.benchmark", "--interactive", "--run", "--output", str(output),
    ])
    inputs = iter([*(f" {question} " for question in questions), "", "yes"])

    def answer_prompt(prompt):
        # 質問入力と実行確認が終わるまで、APIを呼ばない。
        client.models.generate_content.assert_not_called()
        return next(inputs)

    monkeypatch.setattr("builtins.input", answer_prompt)
    benchmark.main()

    expected = [(model, question) for question in questions for model in experiment.models]
    assert [
        (call.kwargs["model"], call.kwargs["contents"])
        for call in client.models.generate_content.call_args_list
    ] == expected
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(row["model"], row["question"]) for row in rows] == expected
    assert {row["question_id"] for row in rows} == {
        f"input_{index}" for index in range(1, len(questions) + 1)
    }
    assert all(row["category"] == "自由入力" for row in rows)
    assert f"完了: {len(expected)}/{len(expected)}リクエスト" in capsys.readouterr().out


def test_interactive_preview_needs_no_key(fake_client, monkeypatch, capsys):
    constructor, _ = fake_client
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr("sys.argv", ["app.benchmark", "--interactive", "--repeat", "2"])
    monkeypatch.setattr("builtins.input", Mock(side_effect=["私の質問", ""]))
    benchmark.main()
    constructor.assert_not_called()
    output = capsys.readouterr().out
    assert "最大4リクエスト" in output
    assert "私の質問" in output
    assert "計画表示のみ" in output


@pytest.mark.parametrize("inputs", [
    ["   "], ["exit"], ["質問", " QUIT "], ["質問", "終了"],
    [EOFError()], [KeyboardInterrupt()], ["質問", EOFError()],
    ["質問", KeyboardInterrupt()], ["質問", "", "no"],
])
def test_interactive_cancel_never_calls_api_or_creates_csv(
    fake_client, monkeypatch, tmp_path, inputs,
):
    constructor, _ = fake_client
    output = tmp_path / "cancelled.csv"
    monkeypatch.setattr("sys.argv", [
        "app.benchmark", "--interactive", "--run", "--output", str(output),
    ])
    monkeypatch.setattr("builtins.input", Mock(side_effect=inputs))
    benchmark.main()
    constructor.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("change", [
    {"repeat": 0}, {"repeat": True}, {"interval_seconds": -1},
    {"interval_seconds": float("nan")}, {"max_output_tokens": 0},
    {"models": []}, {"models": [" "]}, {"models": ["same", "same"]},
    {"questions": []}, {"temperature": 3}, {"thinking_level": "UNKNOWN"},
])
def test_invalid_settings_are_rejected(experiment, change):
    settings = experiment.model_dump()
    settings.update(change)
    with pytest.raises(ValidationError):
        benchmark.Experiment.model_validate(settings)


def test_invalid_json_does_not_call_api(fake_client, monkeypatch, tmp_path):
    constructor, _ = fake_client
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(["not an object"]))
    monkeypatch.setattr("sys.argv", ["app.benchmark", "--config", str(path), "--run"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    constructor.assert_not_called()
