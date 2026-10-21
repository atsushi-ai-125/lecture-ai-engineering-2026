"""第1回: 素のLLMに会話履歴を渡しながら質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

回答後は続けて質問できる。exit / quit / 終了 を入力すると終了する。
会話履歴は実行中だけ保持し、次の質問と一緒に送る。ツールは使わない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.Content] | None = None) -> str:
    """質問に回答し、履歴が渡された場合は成功した会話を追記する。"""
    user_message = types.Content(role="user", parts=[types.Part(text=question)])
    contents = [*history, user_message] if history is not None else question
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=contents)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else "不明"
    output_tokens = usage.candidates_token_count if usage else "不明"
    print(f"[tokens] 入力: {input_tokens} / 出力: {output_tokens}")

    answer = response.text
    if history is not None and answer:
        history.extend([
            user_message,
            types.Content(role="model", parts=[types.Part(text=answer)]),
        ])
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに会話履歴を渡しながら質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print("終了するには exit / quit / 終了 を入力してください。")
    history: list[types.Content] = []
    question = args.question
    try:
        while True:
            question = question.strip()
            if question.lower() in {"exit", "quit", "終了"}:
                break
            if question:
                print(f"[質問] {question}\n")
                print(ask(question, history))
            question = input("\n[次の質問] ")
    except (EOFError, KeyboardInterrupt):
        print("\n対話を終了します。")
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
