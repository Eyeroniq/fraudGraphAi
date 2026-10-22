//! Client for the FraudGraph scoring API (Juspay track J4).
//!
//! [`Client::score`] calls `POST /v1/score` and returns the decision with its calibrated
//! probability, expected cost per action, conformal set and reasons. The [`frm`] module
//! turns that into a pre-authorisation action for a payment switch, in the style of a
//! Hyperswitch fraud-risk-management (FRM) connector.
//!
//! ```no_run
//! use fraudgraph_client::{Client, ScoreRequest, TxnType};
//!
//! let client = Client::new("http://localhost:8000", "fg_live_key");
//! let decision = client.score(&ScoreRequest {
//!     txn_id: "T1".into(),
//!     step: 30,
//!     txn_type: TxnType::Transfer,
//!     amount: 181_000.0,
//!     name_orig: "C1305486145".into(),
//!     name_dest: "C553264065".into(),
//! })?;
//! println!("{:?} p={:.3}", decision.decision, decision.p_fraud);
//! # Ok::<(), fraudgraph_client::Error>(())
//! ```

pub mod frm;

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Header that carries the API key.
pub const KEY_HEADER: &str = "X-API-Key";

/// PaySim transaction types accepted by the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TxnType {
    CashIn,
    CashOut,
    Debit,
    Payment,
    Transfer,
}

/// Body of `POST /v1/score`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreRequest {
    pub txn_id: String,
    /// Hour index of the payment (PaySim step, 1..=744).
    pub step: u32,
    #[serde(rename = "type")]
    pub txn_type: TxnType,
    pub amount: f64,
    pub name_orig: String,
    pub name_dest: String,
}

/// The action the API chose (cheapest expected cost, see the API's A6 cost model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Decision {
    Approve,
    StepUp,
    Review,
    Block,
}

/// One plain-language reason for a decision that was not an approval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reason {
    pub code: String,
    pub text: String,
    /// Contribution to the risk in log-odds.
    pub impact: f64,
}

/// Response of `POST /v1/score`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreResponse {
    pub txn_id: String,
    pub model_version: String,
    pub p_fraud: f64,
    pub decision: Decision,
    pub decision_reason: String,
    pub expected_cost: HashMap<String, f64>,
    /// Conformal set: `["legit"]`, `["fraud"]`, both (model uncertain) or empty.
    #[serde(default)]
    pub prediction_set: Option<Vec<String>>,
    #[serde(default)]
    pub reasons: Vec<Reason>,
    #[serde(default)]
    pub explanation_stable: Option<bool>,
    pub latency_ms: f64,
}

impl ScoreResponse {
    /// True when the conformal set holds both labels: the model is not sure either way.
    pub fn uncertain(&self) -> bool {
        self.prediction_set.as_ref().is_some_and(|s| s.len() == 2)
    }
}

/// Body of `POST /v1/labels`: a late label (chargeback or analyst verdict).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelRequest {
    pub txn_id: String,
    pub observed_fraud: bool,
    pub step: u32,
    pub source: String,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    code: String,
    message: String,
}

/// Everything that can go wrong with a call.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The API answered with an error (`{"error": {"code", "message"}}`).
    Api {
        status: u16,
        code: String,
        message: String,
    },
    /// No usable answer: connection refused, timeout, TLS, ...
    Transport(String),
    /// An answer that is not the expected JSON.
    Decode(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Api {
                status,
                code,
                message,
            } => write!(f, "API error {status} {code}: {message}"),
            Error::Transport(e) => write!(f, "transport error: {e}"),
            Error::Decode(e) => write!(f, "decode error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Blocking client. Cheap to clone; reuses connections.
#[derive(Debug, Clone)]
pub struct Client {
    base: String,
    key: String,
    agent: ureq::Agent,
}

impl Client {
    /// Client with a 200 ms timeout: scoring sits on the payment path.
    pub fn new(base: impl Into<String>, key: impl Into<String>) -> Self {
        Self::with_timeout(base, key, Duration::from_millis(200))
    }

    pub fn with_timeout(
        base: impl Into<String>,
        key: impl Into<String>,
        timeout: Duration,
    ) -> Self {
        let base: String = base.into();
        let agent = ureq::AgentBuilder::new().timeout(timeout).build();
        Client {
            base: base.trim_end_matches('/').to_string(),
            key: key.into(),
            agent,
        }
    }

    /// Score one transaction (`POST /v1/score`).
    pub fn score(&self, request: &ScoreRequest) -> Result<ScoreResponse, Error> {
        self.post("/v1/score", request)
    }

    /// Post a late label (`POST /v1/labels`); returns whether it was new.
    pub fn post_label(&self, label: &LabelRequest) -> Result<bool, Error> {
        let body: serde_json::Value = self.post("/v1/labels", label)?;
        let created = body.get("created").and_then(serde_json::Value::as_bool);
        Ok(created.unwrap_or(false))
    }

    fn post<B: Serialize, R: DeserializeOwned>(&self, path: &str, body: &B) -> Result<R, Error> {
        let url = format!("{}{}", self.base, path);
        let result = self
            .agent
            .post(&url)
            .set(KEY_HEADER, &self.key)
            .send_json(body);
        match result {
            Ok(response) => response
                .into_json::<R>()
                .map_err(|e| Error::Decode(e.to_string())),
            Err(ureq::Error::Status(status, response)) => {
                let text = response.into_string().unwrap_or_default();
                Err(match serde_json::from_str::<ErrorBody>(&text) {
                    Ok(body) => Error::Api {
                        status,
                        code: body.error.code,
                        message: body.error.message,
                    },
                    Err(_) => Error::Api {
                        status,
                        code: "http_error".into(),
                        message: text,
                    },
                })
            }
            Err(e) => Err(Error::Transport(e.to_string())),
        }
    }
}
