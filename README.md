# Windlet.Templates

The job images Windlet's catalogue lists (`Windlet.Api/src/Api/stacks.json`): what runs inside an order, on the machine of the
account that placed it. The platform never runs a model; it places the order, carries its traffic and stores what it leaves.
Each folder here is one image, built and published by CI, and listed in the catalogue by its digest.

| Folder | Image | What it is |
| --- | --- | --- |
| [`couch-serve`](couch-serve) | `ghcr.io/amarcode-studios/windlet-couch-serve` | Serves a checkpoint of couch's small GPT as OpenAI chat completions: the Playground's Models tab loads it (behavioralstate/spec#329) |

## couch-serve

Loads a checkpoint the `couch` template's training left in `/output` (`checkpoint.pt`: the weights, the model's shape, its
characters and the step) and answers `POST /v1/chat/completions` on port 8080.

```
python serve.py --checkpoint /input/checkpoint.pt --port 8080
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "To be or not"}], "max_tokens": 200, "temperature": 0.8, "top_k": 40}'
```

- `messages`: each with its `content` as text. The model is character-level and has no chat format: it reads the messages joined
  by new lines, the end of them that fits its context, and writes on from there. The reply is only what it wrote.
- `max_tokens`: 1 to 2000 characters, 500 when left out. `temperature`: 0 to 2, 0.8 when left out; 0 always picks the
  likeliest character. `top_k`: 1 or more, 40 when left out, capped at the number of characters the model knows.
- The answer is OpenAI's: `choices[0].message.content` holds the reply.
- A refusal is a 400 whose `error.message` says what to change: a character the model never saw in its training text, named;
  a setting out of its range; a request that is not JSON. A request over 1 MiB is a 413.
- One message is answered at a time; another waits its turn.
- The checkpoint is loaded as weights only, so a file that holds code is refused and the server does not start.
- `GET /health` answers once the model is loaded, with the checkpoint's step.

`model.py` and `tokeniser.py` are couch's (github.com/cameronpyne-smith/couch, at c9bdc50), kept here so the image is built
from this repository alone. A checkpoint from a later couch whose model changed needs them brought up to date, with a test.

## Building and checking

`scripts/gate.sh` runs each folder's lint (ruff) and tests (pytest) in a container, from what is committed, and ends in
`GATE GREEN`. It needs Docker and nothing else. CI runs the same gate, then builds each image and pushes it as
`ghcr.io/amarcode-studios/windlet-<folder>:<commit>`; the run's summary prints the digest the catalogue pins.

A node pulls an image without a credential, so each package is public on ghcr.io.
