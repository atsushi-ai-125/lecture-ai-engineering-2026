# 第2回 解答

## 演習

- 使ったモデル・設定（`result.json` の `config` から）：
- 自分で学習を実行したか（`executed_training_myself`）：はい / いいえ（実行できた範囲：　）
- 学習前：`format_rate` =　 / `content_rate` =　
- 学習後：`format_rate` =　 / `content_rate` =　
- 退行した質問：
- 学習後に残った失敗と、その型（根拠なしの質問に断定した / 根拠ありの質問に確認要求で逃げた / 形式の崩れ）：
- 知識を問う質問（セクション 6）の結果（学習前の分類 → 学習後の分類）と気づいたこと：
- 学習で変わったこと・変わらなかったこと：

## 宿題

- 取り組んだレベル：基礎 
- 変えた設定やデータ：learning rate
- 結果（比較表や数字）：

試行１（learning rate:0.0002)
{
  "config": {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 0.0002,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {
      "do_sample": false
    }
  },
  "versions": {
    "transformers": "5.18.0",
    "peft": "0.21.1",
    "trl": "1.14.2",
    "torch": "2.11.0+cu130"
  },
  "method": "LoRA（量子化なし）",
  "before": {
    "label": "before",
    "n": 21,
    "format_rate": 0.09523809523809523,
    "content_rate": 0.0,
    "both_rate": 0.0,
    "seconds": 43.4
  },
  "after": {
    "label": "after",
    "n": 21,
    "format_rate": 1.0,
    "content_rate": 0.8571428571428571,
    "both_rate": 0.8571428571428571,
    "seconds": 45.1
  },
  "regressed": [],
  "executed_training_myself": true
}

試行２(learning rate:0.002)
{
  "config": {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 0.002,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {
      "do_sample": false
    }
  },
  "versions": {
    "transformers": "5.18.0",
    "peft": "0.21.1",
    "trl": "1.14.2",
    "torch": "2.11.0+cu130"
  },
  "method": "LoRA（量子化なし）",
  "before": {
    "label": "before",
    "n": 21,
    "format_rate": 0.09523809523809523,
    "content_rate": 0.0,
    "both_rate": 0.0,
    "seconds": 22.5
  },
  "after": {
    "label": "after",
    "n": 21,
    "format_rate": 1.0,
    "content_rate": 0.8095238095238095,
    "both_rate": 0.8095238095238095,
    "seconds": 41.8
  },
  "regressed": [],
  "executed_training_myself": true
}

試行３(learning rate:0.02)
{
  "config": {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 0.02,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {
      "do_sample": false
    }
  },
  "versions": {
    "transformers": "5.18.0",
    "peft": "0.21.1",
    "trl": "1.14.2",
    "torch": "2.11.0+cu130"
  },
  "method": "LoRA（量子化なし）",
  "before": {
    "label": "before",
    "n": 21,
    "format_rate": 0.09523809523809523,
    "content_rate": 0.0,
    "both_rate": 0.0,
    "seconds": 49.3
  },
  "after": {
    "label": "after",
    "n": 21,
    "format_rate": 0.0,
    "content_rate": 0.0,
    "both_rate": 0.0,
    "seconds": 139.4
  },
  "regressed": [],
  "executed_training_myself": true
}
- ファインチューニングが向く場面・向かない場面についての考察：
今回の3つの施行では、learning rate をそれぞれ0.0002, 0.002, 0.02と変更し、それによるモデル性能の変化を観察した。

その結果、内容正答率はそれぞれ85.7%、81.0%、0%となった。

試行１と２では、特に実際の正答数の違いは１しかなく、今回の小規模の実験では、明確な差は観測できなかったといえる。
一方で試行１，２と、試行３を比較すると、試行３での大きな性能の低下がみられた。特に0.02では形式遵守率も0%となり、過大な学習率が学習の安定性やモデルの性能に悪影響を及ぼした可能性が示唆された。

今回はlearning rateの値を大きく変更したため、このような極端な変化が得られたと考えられる。実際にモデル性能を高めるために試行を繰り返す場合は、より小さな変化量で、最適なlearning rateを探索することが望ましいと考えられる。
また、極端に理想とかけ離れたlearning rateの設定は、本来のモデルよりも性能を低下させる可能性があることも今回の観測から読み取れた。
