# CLAUDE.md

How Windlet is built and shipped is [`Windlet.Api/SDLC.md`](https://github.com/Amarcode-Studios/Windlet.Api/blob/main/SDLC.md), the one lifecycle of the
Windlet repositories; this repository's own rules are
[`.github/instructions/general.instructions.md`](.github/instructions/general.instructions.md). Read them before changing anything, and keep them the current
rule when a change moves the process. What each image is and how to run it is [`README.md`](README.md). In short:

- Every change is made on `main`, one commit per piece of work, the subject one plain sentence saying what changed and
  why, with no type prefix; the one prefix is `Review:` on a commit that answers the review. Commits end with
  `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. No branch, no pull request. Open work is an issue on the
  Behavioralstate board (behavioralstate/spec), never a plan file or a `TODO`.
- Before every push: `scripts/gate.sh` must end in `GATE GREEN`, then the review, `/windlet-review`, every finding fixed on
  `main` with a test or set aside with its reason.
- Each folder is one job image the catalogue lists by digest. A job image runs on a person's own machine with their files:
  it runs as a user that is not root, loads no code from what it is given, and says in words what it refuses.
