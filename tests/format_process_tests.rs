mod format_harness;

use format_harness::format::{process_request, RequestFormat, RequestProcessError};
use serde_json::{json, Value};

fn transform_config(strict_parse: bool, to: &str) -> Value {
    json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": strict_parse,
            "to": to
        }
    })
}

#[test]
fn detects_openai_chat_and_rewrites_path_for_responses_target() {
    let plan = process_request(
        &transform_config(true, "openai_responses"),
        "/proxy/openai/v1/chat/completions",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Ping"}
            ]
        }),
    )
    .expect("openai chat request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiResponses));
    assert!(plan.stream);
    assert_eq!(plan.path, "/proxy/openai/v1/responses");
    assert_eq!(plan.body["instructions"], "Be terse.");
    assert_eq!(plan.body["input"][0]["role"], "user");
    assert_eq!(plan.body["input"][0]["content"][0]["text"], "Ping");
}

#[test]
fn detects_claude_chat_from_headers_and_rewrites_path_for_openai_chat_target() {
    let plan = process_request(
        &transform_config(true, "openai_chat"),
        "/relay/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [
                {"role": "user", "content": "Summarize this"}
            ]
        }),
    )
    .expect("claude request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/relay/v1/chat/completions");
    assert_eq!(plan.body["messages"][0]["role"], "user");
    assert_eq!(plan.body["messages"][0]["content"], "Summarize this");
}

#[test]
fn detects_openai_responses_and_rewrites_path_for_claude_target() {
    let plan = process_request(
        &transform_config(true, "claude_chat"),
        "/edge/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1",
            "instructions": "Stay factual.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello"}
                    ]
                }
            ]
        }),
    )
    .expect("responses request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.path, "/edge/v1/messages");
    assert_eq!(plan.body["system"][0]["text"], "Stay factual.");
    assert_eq!(plan.body["messages"][0]["role"], "user");
    assert_eq!(plan.body["messages"][0]["content"][0]["text"], "Hello");
}

#[test]
fn detects_gemini_chat_and_rewrites_streaming_model_path() {
    let plan = process_request(
        &transform_config(true, "openai_chat"),
        "/proxy/google/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        &[],
        json!({
            "model": "gemini-2.5-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Hi"}
                    ]
                }
            ]
        }),
    )
    .expect("gemini request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::GeminiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert!(plan.stream);
    assert_eq!(plan.path, "/proxy/google/v1/chat/completions");
    assert_eq!(plan.body["messages"][0]["role"], "user");
    assert_eq!(plan.body["messages"][0]["content"], "Hi");
}

#[test]
fn strict_parse_returns_error_when_detection_fails() {
    let error = process_request(
        &transform_config(true, "openai_chat"),
        "/unknown",
        &[],
        json!({
            "foo": "bar"
        }),
    )
    .expect_err("strict mode should reject undetected format");

    match error {
        RequestProcessError::StrictParse(message) => {
            assert!(message.contains("unable to detect request format"));
        }
        other => panic!("expected strict parse error, got {other:?}"),
    }
}

#[test]
fn non_strict_mode_preserves_request_when_detection_fails() {
    let original_body = json!({
        "foo": "bar"
    });
    let plan = process_request(
        &transform_config(false, "openai_chat"),
        "/unknown",
        &[],
        original_body.clone(),
    )
    .expect("non-strict mode should pass through undetected requests");

    assert_eq!(plan.source_format, None);
    assert_eq!(plan.target_format, None);
    assert_eq!(plan.path, "/unknown");
    assert_eq!(plan.body, original_body);
    assert!(!plan.stream);
}
