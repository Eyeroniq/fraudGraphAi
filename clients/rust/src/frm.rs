//! A pre-authorisation fraud-risk hook for a payment switch (Juspay track J4).
//!
//! A switch such as Hyperswitch asks its fraud-risk-management (FRM) connector, before
//! authorising, whether to go ahead, challenge the customer, hold the payment for review
//! or cancel it. [`PreAuthHook::check`] maps a FraudGraph decision onto those actions:
//!
//! | FraudGraph   | [`FrmAction`]          | what the switch does                   |
//! |--------------|------------------------|----------------------------------------|
//! | `APPROVE`    | `Continue`             | authorise                              |
//! | `STEP_UP`    | `Challenge`            | 3DS / OTP challenge, then authorise    |
//! | `REVIEW`     | `ManualReview`         | hold; an analyst decides               |
//! | `BLOCK`      | `Cancel`               | decline                                |
//!
//! When the API cannot be reached in time the hook does not guess: it returns the
//! configured [`FailMode`] action and says so in the verdict, so the switch can log it.

use crate::{Client, Decision, Error, Reason, ScoreRequest, ScoreResponse, TxnType};

/// What the switch should do with the payment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrmAction {
    Continue,
    Challenge,
    ManualReview,
    Cancel,
}

impl From<Decision> for FrmAction {
    fn from(decision: Decision) -> Self {
        match decision {
            Decision::Approve => FrmAction::Continue,
            Decision::StepUp => FrmAction::Challenge,
            Decision::Review => FrmAction::ManualReview,
            Decision::Block => FrmAction::Cancel,
        }
    }
}

/// Behaviour when scoring fails (timeout, API down, bad answer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailMode {
    /// Authorise anyway: availability first (the payment is still logged as unscored).
    Open,
    /// Hold for review: risk first.
    Closed,
}

impl FailMode {
    fn action(self) -> FrmAction {
        match self {
            FailMode::Open => FrmAction::Continue,
            FailMode::Closed => FrmAction::ManualReview,
        }
    }
}

/// The payment as a switch sees it before authorisation.
#[derive(Debug, Clone, PartialEq)]
pub struct PaymentAttempt {
    pub payment_id: String,
    /// Amount in minor units (paise, cents).
    pub amount_minor: i64,
    /// Paying account (customer / VPA).
    pub payer: String,
    /// Receiving account (merchant or VPA).
    pub payee: String,
    /// Peer-to-peer transfer or a cash withdrawal.
    pub kind: PaymentKind,
    /// Hour index of the payment in the model's clock.
    pub step: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaymentKind {
    Transfer,
    CashOut,
    Payment,
}

impl PaymentAttempt {
    pub fn to_score_request(&self) -> ScoreRequest {
        ScoreRequest {
            txn_id: self.payment_id.clone(),
            step: self.step,
            txn_type: match self.kind {
                PaymentKind::Transfer => TxnType::Transfer,
                PaymentKind::CashOut => TxnType::CashOut,
                PaymentKind::Payment => TxnType::Payment,
            },
            amount: self.amount_minor as f64 / 100.0,
            name_orig: self.payer.clone(),
            name_dest: self.payee.clone(),
        }
    }
}

/// The hook's answer: the action, and the evidence behind it when there is some.
#[derive(Debug, Clone, PartialEq)]
pub struct FrmVerdict {
    pub action: FrmAction,
    /// None when scoring failed and the fail mode decided.
    pub score: Option<ScoreResponse>,
    pub error: Option<Error>,
}

impl FrmVerdict {
    /// Reasons to show an analyst or the customer (empty for approvals and failures).
    pub fn reasons(&self) -> &[Reason] {
        match &self.score {
            Some(score) => score.reasons.as_slice(),
            None => &[],
        }
    }
}

/// Pre-authorisation hook around a [`Client`].
#[derive(Debug, Clone)]
pub struct PreAuthHook {
    client: Client,
    fail_mode: FailMode,
}

impl PreAuthHook {
    pub fn new(client: Client, fail_mode: FailMode) -> Self {
        PreAuthHook { client, fail_mode }
    }

    pub fn check(&self, attempt: &PaymentAttempt) -> FrmVerdict {
        match self.client.score(&attempt.to_score_request()) {
            Ok(score) => FrmVerdict {
                action: score.decision.into(),
                score: Some(score),
                error: None,
            },
            Err(error) => FrmVerdict {
                action: self.fail_mode.action(),
                score: None,
                error: Some(error),
            },
        }
    }
}
