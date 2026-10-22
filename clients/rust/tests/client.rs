use std::time::Duration;

use fraudgraph_client::frm::{FailMode, FrmAction, PaymentAttempt, PaymentKind, PreAuthHook};
use fraudgraph_client::{Client, Decision, Error, LabelRequest, ScoreRequest, TxnType};

const REVIEW: &str = r#"{
  "txn_id": "T1",
  "model_version": "lgbm_g1-2026.10.21-9fe27bf",
  "p_fraud": 0.412,
  "decision": "REVIEW",
  "decision_reason": "model_uncertain",
  "expected_cost": {"APPROVE": 1830.2, "REVIEW": 188.0, "BLOCK": 29.4},
  "prediction_set": ["legit", "fraud"],
  "reasons": [{"code": "DEST_HISTORY", "text": "Receiving account is new", "impact": 1.21}],
  "explanation_stable": true,
  "latency_ms": 18.7
}"#;

fn request() -> ScoreRequest {
    ScoreRequest {
        txn_id: "T1".into(),
        step: 30,
        txn_type: TxnType::Transfer,
        amount: 181_000.0,
        name_orig: "C1305486145".into(),
        name_dest: "C553264065".into(),
    }
}

fn attempt() -> PaymentAttempt {
    PaymentAttempt {
        payment_id: "T1".into(),
        amount_minor: 18_100_000,
        payer: "C1305486145".into(),
        payee: "C553264065".into(),
        kind: PaymentKind::Transfer,
        step: 30,
    }
}

#[test]
fn score_sends_the_key_and_the_contract_fields() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/score")
        .match_header("x-api-key", "fg_test")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "txn_id": "T1", "step": 30, "type": "TRANSFER", "amount": 181000.0,
            "name_orig": "C1305486145", "name_dest": "C553264065"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(REVIEW)
        .create();

    let response = Client::new(server.url(), "fg_test")
        .score(&request())
        .unwrap();
    mock.assert();
    assert_eq!(response.decision, Decision::Review);
    assert!(response.uncertain());
    assert_eq!(response.reasons[0].code, "DEST_HISTORY");
    assert_eq!(response.expected_cost["BLOCK"], 29.4);
}

#[test]
fn api_errors_keep_their_code_and_message() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/score")
        .with_status(409)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error": {"code": "out_of_order", "message": "step 3 < 30"}}"#)
        .create();

    let error = Client::new(server.url(), "k")
        .score(&request())
        .unwrap_err();
    assert_eq!(
        error,
        Error::Api {
            status: 409,
            code: "out_of_order".into(),
            message: "step 3 < 30".into()
        }
    );
}

#[test]
fn post_label_reports_whether_it_was_new() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/labels")
        .with_status(201)
        .with_header("content-type", "application/json")
        .with_body(r#"{"label": {}, "created": true}"#)
        .create();
    let label = LabelRequest {
        txn_id: "T1".into(),
        observed_fraud: true,
        step: 54,
        source: "chargeback".into(),
    };
    assert!(Client::new(server.url(), "k").post_label(&label).unwrap());
}

#[test]
fn every_decision_maps_to_one_frm_action() {
    assert_eq!(FrmAction::from(Decision::Approve), FrmAction::Continue);
    assert_eq!(FrmAction::from(Decision::StepUp), FrmAction::Challenge);
    assert_eq!(FrmAction::from(Decision::Review), FrmAction::ManualReview);
    assert_eq!(FrmAction::from(Decision::Block), FrmAction::Cancel);
}

#[test]
fn hook_converts_minor_units_and_returns_the_reasons() {
    let mut server = mockito::Server::new();
    server
        .mock("POST", "/v1/score")
        .match_body(mockito::Matcher::PartialJson(
            serde_json::json!({"amount": 181000.0, "type": "TRANSFER"}),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(REVIEW)
        .create();
    let hook = PreAuthHook::new(Client::new(server.url(), "k"), FailMode::Open);
    let verdict = hook.check(&attempt());
    assert_eq!(verdict.action, FrmAction::ManualReview);
    assert_eq!(verdict.reasons().len(), 1);
    assert!(verdict.error.is_none());
}

#[test]
fn unreachable_api_falls_back_to_the_fail_mode() {
    // Nothing listens on port 9 (discard) on CI runners; the call fails fast.
    let client = Client::with_timeout("http://127.0.0.1:9", "k", Duration::from_millis(200));
    for (mode, action) in [
        (FailMode::Open, FrmAction::Continue),
        (FailMode::Closed, FrmAction::ManualReview),
    ] {
        let verdict = PreAuthHook::new(client.clone(), mode).check(&attempt());
        assert_eq!(verdict.action, action);
        assert!(verdict.score.is_none());
        assert!(matches!(verdict.error, Some(Error::Transport(_))));
    }
}
