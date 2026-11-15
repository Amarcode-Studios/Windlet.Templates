import dataclasses
import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from model import GPT, ModelConfig
from serve import CHAT_PATH, Refused, chat_completion, handler_for, load_checkpoint, prompt_of, read_request
from tokeniser import Tokeniser

CONFIG = ModelConfig(vocab_size=4, block_size=8, n_layer=1, n_head=1, n_embedding_dimension=8)


def save_as_training_does(model, config, tokeniser, step, path):
    torch.save({"state": model.state_dict(), "config": dataclasses.asdict(config), "tokeniser": tokeniser.chars, "step": step}, path / "checkpoint.pt")


def small():
    torch.manual_seed(0)
    return GPT(CONFIG).eval(), Tokeniser("abc\n")


def ask(messages, **settings):
    return {"messages": messages, **settings}


def test_answers_a_chat_completion_of_the_length_asked():
    model, tokeniser = small()
    answer = chat_completion(model, tokeniser, ask([{"role": "user", "content": "abc"}], max_tokens=12))
    content = answer["choices"][0]["message"]["content"]
    assert answer["object"] == "chat.completion"
    assert answer["choices"][0]["message"]["role"] == "assistant"
    assert len(content) == 12
    assert set(content) <= set("abc\n")
    assert answer["usage"] == {"prompt_tokens": 3, "completion_tokens": 12, "total_tokens": 15}


def test_temperature_zero_always_writes_the_same():
    model, tokeniser = small()
    request = ask([{"role": "user", "content": "ab"}], max_tokens=20, temperature=0)
    first = chat_completion(model, tokeniser, request)["choices"][0]["message"]["content"]
    second = chat_completion(model, tokeniser, request)["choices"][0]["message"]["content"]
    assert first == second


def test_top_k_one_always_writes_the_same_and_a_large_one_is_the_whole_vocabulary():
    model, tokeniser = small()
    request = ask([{"role": "user", "content": "ab"}], max_tokens=20, top_k=1)
    assert chat_completion(model, tokeniser, request)["choices"][0]["message"]["content"] == chat_completion(model, tokeniser, request)["choices"][0]["message"]["content"]
    assert read_request(ask([{"role": "user", "content": "a"}], top_k=1000), tokeniser.vocab_size)[3] == 4


def test_reads_the_end_of_the_conversation_that_fits_its_context():
    _, tokeniser = small()
    messages = [{"role": "user", "content": "aaaa"}, {"role": "assistant", "content": "bbbb"}, {"role": "user", "content": "cc"}]
    assert prompt_of(messages, tokeniser, 8) == "\nbbbb\ncc"
    assert prompt_of(messages, tokeniser, 256) == "aaaa\nbbbb\ncc"


@pytest.mark.parametrize(
    "body, said",
    [
        ([], "JSON object"),
        ({}, "messages"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "user"}]}, "content"),
        (ask([{"role": "user", "content": "a"}], max_tokens=0), "max_tokens"),
        (ask([{"role": "user", "content": "a"}], max_tokens=2001), "max_tokens"),
        (ask([{"role": "user", "content": "a"}], max_tokens=True), "max_tokens"),
        (ask([{"role": "user", "content": "a"}], temperature=-1), "temperature"),
        (ask([{"role": "user", "content": "a"}], temperature=2.5), "temperature"),
        (ask([{"role": "user", "content": "a"}], top_k=0), "top_k"),
        (ask([{"role": "user", "content": "xyz"}]), "never saw"),
        (ask([{"role": "user", "content": ""}]), "say nothing"),
    ],
)
def test_refuses_in_words_what_it_cannot_take(body, said):
    model, tokeniser = small()
    with pytest.raises(Refused, match=said):
        chat_completion(model, tokeniser, body)


def test_loads_what_training_saved(tmp_path):
    model, tokeniser = small()
    save_as_training_does(model, CONFIG, tokeniser, 7, tmp_path)
    loaded, loaded_tokeniser, step = load_checkpoint(tmp_path / "checkpoint.pt", "cpu")
    assert step == 7
    assert loaded_tokeniser.chars == tokeniser.chars
    assert not loaded.training
    request = ask([{"role": "user", "content": "ab"}], max_tokens=10, temperature=0)
    assert chat_completion(loaded, loaded_tokeniser, request)["choices"][0]["message"]["content"] == chat_completion(model, tokeniser, request)["choices"][0]["message"]["content"]


def test_refuses_a_checkpoint_that_holds_code(tmp_path):
    class Runs:
        def __reduce__(self):
            return (print, ("ran",))

    torch.save({"state": {}, "config": {}, "tokeniser": [], "step": 0, "extra": Runs()}, tmp_path / "checkpoint.pt")
    with pytest.raises(Exception):
        load_checkpoint(tmp_path / "checkpoint.pt", "cpu")


@pytest.fixture
def address():
    from http.server import ThreadingHTTPServer

    model, tokeniser = small()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(model, tokeniser, 3))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(url, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_serves_chat_completions_over_http(address):
    status, answer = post(address + CHAT_PATH, ask([{"role": "user", "content": "abc"}], max_tokens=5, temperature=0.8, top_k=40))
    assert status == 200
    assert len(answer["choices"][0]["message"]["content"]) == 5


def test_a_refusal_over_http_is_a_400_with_its_words_and_anything_else_a_404(address):
    status, answer = post(address + CHAT_PATH, ask([{"role": "user", "content": "a"}], max_tokens=5000))
    assert status == 400
    assert "max_tokens" in answer["error"]["message"]
    status, answer = post(address + CHAT_PATH, None, raw=b"not json")
    assert status == 400
    status, _ = post(address + "/v1/other", {})
    assert status == 404
    with urllib.request.urlopen(address + "/health") as response:
        assert json.loads(response.read())["step"] == 3
