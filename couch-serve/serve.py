import argparse
import json
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
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
CHAT_PATH = "/v1/chat/completions"


class Refused(Exception):
    pass


def load_checkpoint(file_path, device):
    d = torch.load(file_path, map_location=device, weights_only=True)
    model = GPT(ModelConfig(**d["config"]))
    model.load_state_dict(d["state"])
    model = model.to(device).eval()
    return model, Tokeniser("".join(d["tokeniser"])), d["step"]


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
    top_k = min(whole_number(body, "top_k", DEFAULT_TOP_K, 1, 1_000_000), vocab_size)
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


def complete(model, tokeniser, prompt, max_tokens, temperature, top_k):
    device = next(model.parameters()).device
    ids = torch.tensor(tokeniser.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    written = []
    with torch.no_grad():
        for _ in range(max_tokens):
            scores = model(ids[:, -model.config.block_size :])[:, -1, :]
            if temperature == 0:
                winner = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                options, _ = torch.topk(scores, top_k)
                scores[scores < options[:, -1:]] = float("-inf")
                winner = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)
            ids = torch.cat([ids, winner], dim=1)
            written.append(winner.item())
    return tokeniser.decode(written)


def chat_completion(model, tokeniser, body, name="couch-gpt"):
    messages, max_tokens, temperature, top_k = read_request(body, tokeniser.vocab_size)
    prompt = prompt_of(messages, tokeniser, model.config.block_size)
    content = complete(model, tokeniser, prompt, max_tokens, temperature, top_k)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(content), "total_tokens": len(prompt) + len(content)},
    }


def handler_for(model, tokeniser, step):
    one_at_a_time = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def answer(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def refuse(self, status, message):
            self.answer(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path == "/health":
                self.answer(200, {"status": "ok", "step": step, "characters": tokeniser.vocab_size, "block_size": model.config.block_size})
            else:
                self.refuse(404, f"nothing here: POST {CHAT_PATH}")

        def do_POST(self):
            if self.path != CHAT_PATH:
                self.refuse(404, f"nothing here: POST {CHAT_PATH}")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.close_connection = True
                self.refuse(411, "send a Content-Length")
                return
            if length < 0 or length > MAX_BODY_BYTES:
                self.close_connection = True
                self.refuse(413, f"the request is larger than {MAX_BODY_BYTES} bytes")
                return
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                self.refuse(400, "the request is not JSON")
                return
            try:
                with one_at_a_time:
                    completion = chat_completion(model, tokeniser, body)
            except Refused as refused:
                self.refuse(400, str(refused))
                return
            self.answer(200, completion)

        def log_message(self, format, *args):
            print(f"{self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}", flush=True)

    return Handler


def serve(checkpoint, host="0.0.0.0", port=8080, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokeniser, step = load_checkpoint(checkpoint, device)
    server = ThreadingHTTPServer((host, port), handler_for(model, tokeniser, step))
    print(f"serving {checkpoint} (step {step}, {tokeniser.vocab_size} characters) on {device} at {host}:{port}{CHAT_PATH}", flush=True)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()
    serve(args.checkpoint, args.host, args.port).serve_forever()
