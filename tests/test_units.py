from postpilot.config import Settings, load_platforms
from postpilot.graph.agents import DraftOut, ImageOut, normalize_draft, rule_violations
from postpilot.rag import chunk_text


def test_platforms_load_with_image_sizes():
    platforms = load_platforms()
    assert {"instagram", "linkedin", "medium"} <= set(platforms)
    ig = platforms["instagram"]
    assert ig.image.get("portrait_4_5").aspect_ratio() == "4:5"
    assert platforms["linkedin"].image.get("landscape_1_91").aspect_ratio() == "1.91:1"


def test_chunk_text_respects_size_and_keeps_everything():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(10))
    chunks = chunk_text(text, size=600, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 600 + 50 + 2 for c in chunks)
    assert all(f"Paragraph {i}" in "".join(chunks) for i in range(10))


def test_normalize_pins_image_to_allowed_size():
    ig = load_platforms()["instagram"]
    draft = DraftOut(
        title="T",
        body="hello",
        hashtags=["dev", "#Dev", "#ai-agents", ""],
        image=ImageOut(size_label="made_up", prompt="a desk", alt_text="desk"),
    )
    out = normalize_draft(ig, draft)
    assert out["hashtags"] == ["#dev", "#aiagents"]
    assert out["image"]["size_label"] == "portrait_4_5"  # unknown label falls back to the default
    assert (out["image"]["width"], out["image"]["height"]) == (1080, 1350)
    assert "1080x1350 px" in out["image"]["prompt"]


def test_hashtags_moved_out_of_body():
    li = load_platforms()["linkedin"]
    d = DraftOut(title="T", body="Line one.\n\nLine two #inline stays.\n\n#AI #Agents\n", hashtags=["#AI"])
    out = normalize_draft(li, d)
    assert out["body"] == "Line one.\n\nLine two #inline stays."
    assert out["hashtags"] == ["#AI", "#Agents"]


def test_required_image_is_added_when_model_omits_it():
    medium = load_platforms()["medium"]
    out = normalize_draft(medium, DraftOut(title="Deep dive", body="x"))
    assert out["image"]["size_label"] == "cover_16_9"


def test_rule_violations():
    li = load_platforms()["linkedin"]
    issues = rule_violations(li, {"body": "too short", "hashtags": []})
    assert any("at least 120" in i for i in issues)
    assert any("hashtags" in i for i in issues)
    ok = {"body": "word " * 150, "hashtags": ["#a", "#b", "#c"]}
    assert rule_violations(li, ok) == []


def test_email_provider_presets():
    gmail = Settings(_env_file=None, email_provider="gmail", smtp_host="ignored", smtp_port=25,
                     smtp_user="me@gmail.com", smtp_password="app-pass", email_to="a@x.com, b@x.com")
    assert (gmail.smtp_host, gmail.smtp_port, gmail.smtp_starttls) == ("smtp.gmail.com", 587, True)
    assert gmail.email_from == "me@gmail.com" and gmail.email_enabled
    assert gmail.email_recipients == ["a@x.com", "b@x.com"]
    assert not Settings(_env_file=None, email_provider="gmail", email_to="a@x.com").email_enabled

    custom = Settings(_env_file=None, email_provider="smtp", smtp_host="smtp.brevo.com", smtp_port=465,
                      smtp_ssl=True, email_from="news@me.dev", email_to="a@x.com")
    assert (custom.smtp_host, custom.smtp_port, custom.smtp_ssl) == ("smtp.brevo.com", 465, True)
    assert custom.email_from == "news@me.dev" and custom.email_enabled
