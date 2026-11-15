import argparse
import json
import select
import signal
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from model import GPT, ModelConfig
from tokeniser import Tokeniser

MAX_BODY_BYTES = 1024 * 1024
MAX_TOKENS = 2000
MAX_TEMPERATURE = 2.0
MIN_TEMPERATURE = 0.01
MAX_TOP_K = 1_000_000
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
WRITES_FOR_SECONDS = 100
CHAT_PATH = "/v1/chat/completions"
HEALTH_PATH = "/health"
MOST_WEIGHTS = 200_000_000
LARGEST = {"vocab_size": 100_000, "block_size": 8192, "n_layer": 48, "n_head": 64, "n_embedding_dimension": 4096}


class Refused(Exception):
    pass


class NotACheckpoint(Exception):
    pass


def load_checkpoint(file_path, device):
    try:
        d = torch.load(file_path, map_location=device, weights_only=True)
        config, characters, step = d["config"], d["tokeniser"], d["step"]
        state = d["state"]
    except Exception as error:
        raise NotACheckpoint(f"{file_path} is not a checkpoint couch's training saved: {type(error).__name__}") from error
    if not isinstance(config, dict) or not isinstance(characters, list) or not all(isinstance(c, str) and len(c) == 1 for c in characters):
        raise NotACheckpoint(f"{file_path} does not hold the model's shape and its characters")
    for name, most in LARGEST.items():
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= most:
            raise NotACheckpoint(f"{file_path} says {name} is {value!r}, which must be a whole number from 1 to {most}")
    width = config["n_embedding_dimension"]
    weights = config["n_layer"] * 12 * width * width + (config["vocab_size"] + config["block_size"]) * width
    if weights > MOST_WEIGHTS:
        raise NotACheckpoint(f"{file_path} states a model of about {weights} weights, more than the {MOST_WEIGHTS} this server loads")
    tokeniser = Tokeniser("".join(characters))
    if tokeniser.vocab_size != config["vocab_size"]:
        raise NotACheckpoint(f"{file_path} holds {tokeniser.vocab_size} characters for a model of {config['vocab_size']}")
    try:
        model = GPT(ModelConfig(**config))
        model.load_state_dict(state)
    except Exception as error:
        raise NotACheckpoint(f"{file_path} holds weights that do not fit the shape it states: {type(error).__name__}") from error
    return model.to(device).eval(), tokeniser, step


def whole_number(body, name, default, low, high):
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise Refused(f"{name} must be a whole number from {low} to {high}")
    return value


def read_request(body, vocab_size):
    if not isinstance(body, dict):
        raise Refused("the request must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise Refused("messages must be a list with at least one message")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise Refused("each message must have its content as text")
    max_tokens = whole_number(body, "max_tokens", DEFAULT_MAX_TOKENS, 1, MAX_TOKENS)
    top_k = min(whole_number(body, "top_k", DEFAULT_TOP_K, 1, MAX_TOP_K), vocab_size)
    temperature = body.get("temperature", DEFAULT_TEMPERATURE)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= MAX_TEMPERATURE:
        raise Refused(f"temperature must be from 0 to {MAX_TEMPERATURE:g}")
    return messages, max_tokens, float(temperature), top_k


def prompt_of(messages, tokeniser, block_size):
    text = "\n".join(message["content"] for message in messages)
    unseen = sorted({c for c in text if c not in tokeniser.char_to_id_map})
    if unseen:
        shown = " ".join(repr(c) for c in unseen[:10])
        raise Refused(f"the model never saw these characters in its training text: {shown}")
    if not text:
        raise Refused("the messages say nothing")
    return text[-block_size:]


def complete(model, tokeniser, prompt, max_tokens, temperature, top_k, stop=lambda: False):
    device = next(model.parameters()).device
    ids = torch.tensor(tokeniser.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    written = []
    with torch.no_grad():
        for _ in range(max_tokens):
            if stop():
                break
            scores = model(ids[:, -model.config.block_size :])[:, -1, :]
            if temperature < MIN_TEMPERATURE:
                winner = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                options, _ = torch.topk(scores, top_k)
                scores[scores < options[:, -1:]] = float("-inf")
                winner = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)
            ids = torch.cat([ids, winner], dim=1)
            written.append(winner.item())
    return tokeniser.decode(written)


def completion_of(prompt, content, asked, name="couch-gpt"):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length" if len(content) == asked else "stop"}],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(content), "total_tokens": len(prompt) + len(content)},
    }


def chat_completion(model, tokeniser, body, stop=lambda: False):
    messages, max_tokens, temperature, top_k = read_request(body, tokeniser.vocab_size)
    prompt = prompt_of(messages, tokeniser, model.config.block_size)
    return completion_of(prompt, complete(model, tokeniser, prompt, max_tokens, temperature, top_k, stop), max_tokens)


def hung_up(connection):
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        return bool(readable) and connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


def handler_for(model, tokeniser, step, writes_for_seconds=WRITES_FOR_SECONDS):
    one_at_a_time = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def answer(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def refuse(self, status, message):
            self.answer(status, {"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error"}})

        def hang_up_refusing(self, status, message):
            self.close_connection = True
            self.refuse(status, message)

        def route(self):
            return self.path.split("?")[0]

        def do_GET(self):
            if self.route() == HEALTH_PATH:
                self.answer(200, {"status": "ok", "step": step, "characters": tokeniser.vocab_size, "block_size": model.config.block_size})
            else:
                self.hang_up_refusing(404, f"nothing here: POST {CHAT_PATH}")

        def other(self):
            self.hang_up_refusing(405, f"only GET {HEALTH_PATH} and POST {CHAT_PATH} are served")

        do_HEAD = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = other

        def do_POST(self):
            if self.route() != CHAT_PATH:
                self.hang_up_refusing(404, f"nothing here: POST {CHAT_PATH}")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.hang_up_refusing(411, "send a Content-Length")
                return
            if length < 0:
                self.hang_up_refusing(400, "Content-Length must not be negative")
                return
            if length > MAX_BODY_BYTES:
                self.hang_up_refusing(413, f"the request is larger than {MAX_BODY_BYTES} bytes")
                return
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw)
            except (ValueError, RecursionError):
                self.refuse(400, "the request is not JSON")
                return
            try:
                messages, max_tokens, temperature, top_k = read_request(body, tokeniser.vocab_size)
                prompt = prompt_of(messages, tokeniser, model.config.block_size)
            except Refused as refused:
                self.refuse(400, str(refused))
                return
            try:
                with one_at_a_time:
                    until = time.monotonic() + writes_for_seconds
                    content = complete(model, tokeniser, prompt, max_tokens, temperature, top_k, lambda: time.monotonic() >= until or hung_up(self.connection))
            except (RuntimeError, ValueError, IndexError, KeyError) as error:
                print(f"a completion failed: {type(error).__name__}", flush=True)
                self.hang_up_refusing(500, "the model could not write a reply; try again, or load it again")
                return
            if hung_up(self.connection):
                self.close_connection = True
            try:
                self.answer(200, completion_of(prompt, content, max_tokens))
            except OSError:
                self.close_connection = True

        def send_error(self, code, message=None, explain=None):
            self.hang_up_refusing(code if 400 <= code < 600 else 400, "the request could not be read")

        def log_message(self, format, *args):
            status = args[1] if len(args) > 1 else ""
            print(f"{getattr(self, 'command', None) or '-'} {getattr(self, 'path', '-').split('?')[0]} {status}", flush=True)

    return Handler


def serve(checkpoint, host="0.0.0.0", port=8080, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokeniser, step = load_checkpoint(checkpoint, device)
    server = ThreadingHTTPServer((host, port), handler_for(model, tokeniser, step))
    server.daemon_threads = True
    print(f"serving {checkpoint} (step {step}, {tokeniser.vocab_size} characters) on {device} at {host}:{port}{CHAT_PATH}", flush=True)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()
    try:
        server = serve(args.checkpoint, args.host, args.port)
    except NotACheckpoint as refused:
        print(str(refused), flush=True)
        sys.exit(2)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    server.serve_forever()
