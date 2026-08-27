# Manifest schema

All scripts use JSONL files. Each line is one example.

```json
{
  "id": "unique_id",
  "source": "llava-video-178k",
  "benchmark": "mvbench",
  "task": "mcq",
  "media_type": "video",
  "media_path": "/data/videos/example.mp4",
  "question": "What happens first?",
  "answer": "A",
  "choices": [
    "The person opens the door.",
    "The person closes the box.",
    "The person leaves.",
    "Nothing happens."
  ]
}
```

Required fields:

- `id`: stable unique id. Feature extraction and evaluation join on this field.
- `media_type`: `video` or `image`.
- `media_path`: local file path on the training server.
- `question`: natural language question or instruction.
- `answer`: target answer. For multiple-choice benchmarks, use the option letter.

Optional fields:

- `choices`: list of option strings. If present, evaluators treat the row as multiple-choice.
- `benchmark`: benchmark name used for grouped accuracy.
- `source`: original dataset/source name.
- `task`: `caption`, `qa`, or `mcq`.

Keep train and eval manifests separate. If a benchmark appears inside a training mix, remove that source split from the training manifest.
