from types import SimpleNamespace

from app import ask


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)

    assert ask.ask("質問") == "回答"
    assert "[トークン] 入力: 12 / 出力: 34" in capsys.readouterr().out


def test_ask_preserves_history_between_questions(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(usage_metadata=None, text="最初の回答"),
            SimpleNamespace(usage_metadata=None, text="続きの回答"),
        ]
    )
    requests = []

    def generate_content(**kwargs):
        requests.append(kwargs)
        return next(responses)

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    history = []

    assert ask.ask("最初の質問", history) == "最初の回答"
    assert ask.ask("続きの質問", history) == "続きの回答"

    first_contents = requests[0]["contents"]
    second_contents = requests[1]["contents"]
    assert [message.role for message in first_contents] == ["user"]
    assert [message.role for message in second_contents] == [
        "user",
        "model",
        "user",
    ]
    assert second_contents[0].parts[0].text == "最初の質問"
    assert second_contents[1].parts[0].text == "最初の回答"
    assert second_contents[2].parts[0].text == "続きの質問"


def test_ask_interactively_reuses_history_until_exit(monkeypatch, capsys):
    questions = iter(["最初の質問", "続きの質問", "exit"])
    calls = []

    def fake_ask(question, history):
        calls.append((question, history))
        return "回答"

    monkeypatch.setattr(ask, "ask", fake_ask)
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask.ask_interactively()

    assert [question for question, _ in calls] == ["最初の質問", "続きの質問"]
    assert calls[0][1] is calls[1][1]
    assert "[質問] 最初の質問" in capsys.readouterr().out
