# Windlet.Templates: the rules every change is held to

Windlet.Templates holds the job images Windlet's catalogue lists: what runs inside an order. How a change gets from a
requirement to production is the workspace's lifecycle,
[`Windlet.Api/SDLC.md`](https://github.com/Amarcode-Studios/Windlet.Api/blob/main/SDLC.md); these are this repository's own
rules, and what its reviewer reads with it.

## Design rules

- One folder is one image. It builds from its own folder alone: nothing is fetched at build time but its pinned
  dependencies, and nothing at run time, since an order has no network unless it asks for one.
- The catalogue (`Windlet.Api/src/Api/stacks.json`) lists an image by its digest, never a tag. A change here reaches a
  person only when the digest there changes, which is the API's change and goes through its gates.
- An image is no part of the platform: it reads no query, sends no command and holds no credential. What reaches it is what
  the order was given under `/input` and the requests the gateway carries to its port.
- It runs as a user that is not root, writes only under `/work`, `/output` and `/tmp`, and listens on `0.0.0.0` at the port
  the catalogue names.
- A file a person gives it is data: it is never unpickled as code, executed or imported. A checkpoint is loaded as weights only.
- A refusal is in words and says what to change, in the shape the caller reads (for a server, OpenAI's
  `{"error": {"message": ...}}` with a 400), and everything a request can set has a bound that the README states.
- What it prints is the order's log, which the person and whoever runs the machine read: no request's content is printed.

## Code

- Python, the version each Dockerfile names. Dependencies are pinned in the folder's `requirements.txt`.
- Tests sit beside the code as `*_test.py` and run with pytest; a behaviour has a test. `ruff check` and `ruff format --check`
  pass. No code comments, except in a file taken from another project, which keeps its own (`couch-serve/model.py`, couch's, where they explain the model).
- Docs move with the change: the README's table and the folder's section state what the image does now, not its history.
