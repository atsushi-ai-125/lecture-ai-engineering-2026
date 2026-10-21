"""第1回 発展: 履歴なしでモデル・質問ごとの応答時間とトークン数を比較する。

uv run python -m app.benchmark         # 計画のみ表示（APIを呼ばない）
uv run python -m app.benchmark --run   # 計画を確認し、yes と入力して実測
uv run python -m app.benchmark --interactive --run  # 質問をターミナルで入力
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import time
from typing import Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import get_api_key

DEFAULT_CONFIG = Path(__file__).with_name("benchmark.json")
TOKEN_FIELDS = {
    "input_tokens": "prompt_token_count",
    "output_tokens": "candidates_token_count",
    "total_tokens": "total_token_count",
    "thinking_tokens": "thoughts_token_count",
}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Experiment(BaseModel):
    """JSONの入力ミスを、APIを呼ぶ前に検出する。"""

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, strict=True, allow_inf_nan=False,
    )
    models: list[str] = Field(min_length=1)
    questions: list[Question] = Field(min_length=1)
    repeat: int = Field(default=1, ge=1)
    interval_seconds: float = Field(default=15, ge=0)
    max_output_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=1.0, ge=0, le=2)
    thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"] | None = None

    @field_validator("models")
    @classmethod
    def validate_models(cls, models):
        if any(not name for name in models) or len(set(models)) != len(models):
            raise ValueError("モデル名は空文字・重複を避けてください")
        return models

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions):
        if len({q.id for q in questions}) != len(questions):
            raise ValueError("質問idは重複を避けてください")
        return questions


@dataclass
class Result:
    request_index: int
    repeat: int
    model: str
    question_id: str
    category: str
    question: str
    started_at_utc: str
    generation_config: str
    interval_seconds: float
    elapsed_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    thinking_tokens: int | None = None
    model_version: str = ""
    finish_reason: str = ""
    status: str = "ok"
    error_code: int | None = None
    error: str = ""
    answer: str = ""


def redact(text: str) -> str:
    """APIエラー等に使用中のキーが含まれても表示・保存しない。"""
    key = os.getenv("GEMINI_API_KEY")
    return text.replace(key, "[REDACTED]") if key else text


def generation_config(experiment: Experiment) -> types.GenerateContentConfig:
    thinking = None
    if experiment.thinking_level is not None:
        thinking = types.ThinkingConfig(thinking_level=experiment.thinking_level)
    return types.GenerateContentConfig(
        temperature=experiment.temperature,
        max_output_tokens=experiment.max_output_tokens,
        thinking_config=thinking,
    )


def measure(client, model, question, config, experiment, index, repeat) -> Result:
    result = Result(
        request_index=index, repeat=repeat, model=model,
        question_id=question.id, category=question.category, question=question.text,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        generation_config=config.model_dump_json(exclude_none=True),
        interval_seconds=experiment.interval_seconds,
    )
    start = time.perf_counter()
    try:
        # 毎回この質問だけを送る。同じclientでも会話履歴は共有されない。
        response = client.models.generate_content(
            model=model, contents=question.text, config=config,
        )
    except errors.APIError as exc:
        result.elapsed_seconds = time.perf_counter() - start
        result.status = "api_error"
        result.error_code = exc.code
        result.error = f"{exc.status or 'APIError'}: {exc.message or '詳細なし'}"
    except Exception as exc:
        result.elapsed_seconds = time.perf_counter() - start
        result.status = "client_error"
        # 通信例外の本文・URL・ヘッダーには認証情報が含まれる可能性がある。
        result.error = f"{type(exc).__name__}: 通信またはSDK処理に失敗しました"
    else:
        result.elapsed_seconds = time.perf_counter() - start
        usage = response.usage_metadata
        for column, attribute in TOKEN_FIELDS.items():
            setattr(result, column, getattr(usage, attribute, None))
        result.answer = response.text or ""
        result.model_version = response.model_version or ""
        if response.candidates:
            reason = response.candidates[0].finish_reason
            result.finish_reason = reason.value if reason else ""
        if not result.answer:
            result.status = "empty"
            feedback = response.prompt_feedback
            reason = feedback.block_reason if feedback else None
            result.finish_reason = reason.value if reason else result.finish_reason
    # 質問・回答を含めてキーを除去してから、表示とCSV保存に渡す。
    return Result(**{
        key: redact(value) if isinstance(value, str) else value
        for key, value in asdict(result).items()
    })


def print_result(result: Result) -> None:
    print(f"\n[{result.request_index}] {result.model} / {result.question_id} / {result.status}")
    print(f"質問: {result.question}")
    print(result.answer or "（回答テキストなし）")
    tokens = " / ".join(
        f"{name}={getattr(result, name) if getattr(result, name) is not None else '未取得'}"
        for name in TOKEN_FIELDS
    )
    print(f"時間={result.elapsed_seconds:.3f}秒 / {tokens}")
    print(f"終了理由: {result.finish_reason or '未取得'}")
    if result.error:
        print(f"エラー {result.error_code or ''}: {result.error}")


def print_summary(results: list[Result]) -> None:
    """成功した回答だけを集計。欠損値を0で埋めず、項目ごとの件数も示す。"""
    for fields in [("model",), ("question_id",), ("model", "question_id")]:
        print(f"\n平均（{' × '.join(fields)}別 / 各値の n は取得件数）")
        groups = {}
        for result in results:
            key = tuple(getattr(result, field) for field in fields)
            groups.setdefault(key, []).append(result)
        for key, rows in groups.items():
            valid = [row for row in rows if row.status == "ok"]
            metrics = []
            for field in ["elapsed_seconds", *TOKEN_FIELDS]:
                values = [getattr(row, field) for row in valid if getattr(row, field) is not None]
                value = f"{mean(values):.3f} (n={len(values)})" if values else "未取得 (n=0)"
                metrics.append(f"{field}={value}")
            print(f"{' / '.join(key)}: 成功={len(valid)}/{len(rows)} | " + " | ".join(metrics))


def run_experiment(experiment: Experiment, output: Path, api_key: str) -> list[Result]:
    config = generation_config(experiment)
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    # 上書きせず、1リクエストごとに保存する。途中停止でも完了分は残る。
    with output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()
        file.flush()
        with genai.Client(
            api_key=api_key, vertexai=False,
            http_options=types.HttpOptions(
                timeout=120_000, retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client:
            try:
                for repeat in range(1, experiment.repeat + 1):
                    for question in experiment.questions:
                        for model in experiment.models:
                            if results:
                                time.sleep(experiment.interval_seconds)
                            result = measure(
                                client, model, question, config, experiment, len(results) + 1, repeat,
                            )
                            writer.writerow(asdict(result))
                            file.flush()
                            results.append(result)
                            print_result(result)
                            if result.status.endswith("error"):
                                if result.error_code == 429:
                                    print("制限に到達しました。AI Studioで割当量・使用量を確認してください。")
                                print("追加リクエスト・自動リトライを行わず停止します。")
                                return results
            except KeyboardInterrupt:
                print("\n中断しました。CSVには完了済みのリクエストのみ残ります。")
    return results


def input_questions() -> list[Question]:
    """入力した質問だけを比較対象にする。空行で完了、終了コマンドで中止。"""
    print("比較する質問を1行ずつ入力してください。空行で入力完了です。")
    print("exit / quit / 終了 で中止します。まだAPIは呼びません。")
    questions = []
    while True:
        text = input(f"[質問{len(questions) + 1}] ").strip()
        if text.lower() in {"exit", "quit", "終了"}:
            return []
        if not text:
            return questions
        questions.append(Question(
            id=f"input_{len(questions) + 1}", category="自由入力", text=text,
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", help="設定ファイルのモデル一覧を上書き")
    parser.add_argument("--repeat", type=int, help="各組み合わせの繰り返し回数")
    parser.add_argument("--interval", type=float, help="前の応答完了から次の送信までの待機秒数")
    parser.add_argument("--output", type=Path, help="保存するCSV（既存ファイルは上書きしない）")
    parser.add_argument("--interactive", action="store_true", help="JSONの質問の代わりにターミナルで質問を入力")
    parser.add_argument("--run", action="store_true", help="計画表示後、対話で確認してAPIを呼ぶ")
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("設定はJSONオブジェクトにしてください")
        for option, field in [("models", "models"), ("repeat", "repeat"), ("interval", "interval_seconds")]:
            if getattr(args, option) is not None:
                settings[field] = getattr(args, option)
        experiment = Experiment.model_validate(settings)
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(redact(str(exc)))

    if args.interactive:
        try:
            questions = input_questions()
        except (EOFError, KeyboardInterrupt):
            questions = []
        if not questions:
            print("中止しました。APIは呼んでいません。")
            return
        experiment.questions = questions

    count = len(experiment.models) * len(experiment.questions) * experiment.repeat
    print(f"計画: {len(experiment.models)}モデル × {len(experiment.questions)}質問 × "
          f"{experiment.repeat}回 = 最大{count}リクエスト（生成のみ・リトライなし）")
    print("モデル: " + redact(", ".join(experiment.models)))
    for question in experiment.questions:
        print(redact(f"  {question.id} [{question.category}]: {question.text}"))
    print(f"待機: {experiment.interval_seconds}秒 / 順序: 繰り返し → 質問 → モデル（設定順）")
    print("生成設定: " + generation_config(experiment).model_dump_json(exclude_none=True))
    print("thinking_level未指定時は各モデルの既定値です。")
    print("料金リスク: このプログラムは無料枠・利用可否・残り割当量を判定できません。")
    print("AI Studioで対象キーのプロジェクト、Free tier、モデル別制限を確認してください。")
    print("有料プロジェクトでは入力・出力（思考を含む）に課金される可能性があります。")
    if not args.run:
        print("計画表示のみ。APIは呼びません。実測する場合は --run を指定してください。")
        return
    try:
        confirmed = input("上記のモデル・回数・料金リスクを確認して実行しますか？ [yes/いいえ]: ")
    except (EOFError, KeyboardInterrupt):
        confirmed = ""
    if confirmed.strip().lower() != "yes":
        print("中止しました。APIは呼んでいません。")
        return
    output = args.output or Path("benchmark_results") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".csv"
    )
    try:
        results = run_experiment(experiment, output, get_api_key())
    except OSError as exc:
        raise SystemExit(redact(f"CSVの作成・保存に失敗しました: {exc}")) from None
    print(f"\n保存先: {output} / 完了: {len(results)}/{count}リクエスト")
    print_summary(results)
    if len(results) < count or any(row.status != "ok" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
