//! Pre-authorisation check of one payment against a running FraudGraph API.
//!
//!     FRAUDGRAPH_URL=http://localhost:8000 FRAUDGRAPH_KEY=fg_... cargo run --example frm_hook

use std::env;

use fraudgraph_client::frm::{FailMode, FrmAction, PaymentAttempt, PaymentKind, PreAuthHook};
use fraudgraph_client::Client;

fn main() {
    let url = env::var("FRAUDGRAPH_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let key = env::var("FRAUDGRAPH_KEY").unwrap_or_default();
    let hook = PreAuthHook::new(Client::new(url, key), FailMode::Closed);

    let attempt = PaymentAttempt {
        payment_id: "pay_demo_0001".into(),
        amount_minor: 18_100_000, // 181,000.00
        payer: "C1305486145".into(),
        payee: "C553264065".into(),
        kind: PaymentKind::Transfer,
        step: 30,
    };
    let verdict = hook.check(&attempt);
    match (&verdict.score, &verdict.error) {
        (Some(score), _) => println!(
            "{:?}: p_fraud {:.3}, decision {:?} ({}), model {}, {:.1} ms",
            verdict.action,
            score.p_fraud,
            score.decision,
            score.decision_reason,
            score.model_version,
            score.latency_ms
        ),
        (None, Some(error)) => println!("{:?} (fail mode): {error}", verdict.action),
        (None, None) => unreachable!(),
    }
    for reason in verdict.reasons() {
        println!("  - {} ({:+.2})", reason.text, reason.impact);
    }
    if verdict.action == FrmAction::Cancel {
        std::process::exit(2);
    }
}
