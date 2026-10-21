"""質問スクリプトのテスト（API呼び出しとキー取得をモックに置き換える）。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import errors

from app import ask


@pytest.fixture
def generate_content(monkeypatch):
    generate = Mock(return_value=SimpleNamespace(
        text="回答です", usage_metadata=None,
    ))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=SimpleNamespace(
        models=SimpleNamespace(generate_content=generate),
    )))
    return generate


def test_conversation_sends_previous_questions_and_answers(monkeypatch, generate_content, capsys):
    monkeypatch.setattr("sys.argv", ["app.ask", "私の名前は太郎です"])
    inputs = iter(["私の名前は？", "その名前を英語で書いて", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    generate_content.side_effect = [
        SimpleNamespace(text=answer, usage_metadata=None)
        for answer in ["太郎さんですね", "太郎です", "Taro"]
    ]

    ask.main()

    calls = generate_content.call_args_list
    assert len(calls) == 3
    assert [len(call.kwargs["contents"]) for call in calls] == [1, 3, 5]
    assert [
        (message.role, message.parts[0].text)
        for message in calls[2].kwargs["contents"]
    ] == [
        ("user", "私の名前は太郎です"),
        ("model", "太郎さんですね"),
        ("user", "私の名前は？"),
        ("model", "太郎です"),
        ("user", "その名前を英語で書いて"),
    ]
    assert "Taro" in capsys.readouterr().out


@pytest.mark.parametrize("exit_command", ["exit", " QUIT ", "終了"])
def test_default_question_blank_input_and_exit(monkeypatch, generate_content, exit_command):
    monkeypatch.setattr("sys.argv", ["app.ask"])
    inputs = iter(["", "   ", exit_command])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    ask.main()

    generate_content.assert_called_once()
    assert generate_content.call_args.kwargs["contents"][0].parts[0].text == ask.DEFAULT_QUESTION


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_input_interruption_exits_cleanly(monkeypatch, generate_content, exception):
    monkeypatch.setattr("sys.argv", ["app.ask", "質問"])
    monkeypatch.setattr("builtins.input", Mock(side_effect=exception))

    ask.main()

    generate_content.assert_called_once()


def test_ask_without_history_preserves_existing_usage(generate_content, capsys):
    generate_content.return_value.usage_metadata = SimpleNamespace(
        prompt_token_count=10, candidates_token_count=20,
    )

    assert ask.ask("質問") == "回答です"

    generate_content.assert_called_once_with(model=ask.MODEL, contents="質問")
    assert "[tokens] 入力: 10 / 出力: 20" in capsys.readouterr().out


def test_failed_request_does_not_change_history(generate_content):
    history = []
    ask.ask("最初の質問", history)
    previous = list(history)
    generate_content.side_effect = errors.APIError(429, {"message": "rate limit"})

    with pytest.raises(errors.APIError):
        ask.ask("次の質問", history)

    assert history == previous


def test_api_error_keeps_existing_message_and_exit_status(monkeypatch, generate_content, capsys):
    monkeypatch.setattr("sys.argv", ["app.ask", "質問"])
    generate_content.side_effect = errors.APIError(429, {"message": "rate limit"})

    with pytest.raises(SystemExit) as exc:
        ask.main()

    assert exc.value.code == 1
    assert "レート制限に達しました" in capsys.readouterr().out
