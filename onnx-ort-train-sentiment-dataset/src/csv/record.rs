use std::fmt::Display;

use serde::Deserialize;

#[derive(Debug, Deserialize, Eq, PartialEq)]
pub struct Record {
    id: Option<u64>,
    comment: String,
    sentiment: u8,
}

impl Record {
    pub fn new(id: Option<u64>, comment: String, sentiment: u8) -> Self {
        Self {
            id,
            comment,
            sentiment,
        }
    }

    pub fn comment(&self) -> &str {
        &self.comment
    }

    pub fn sentiment(&self) -> u8 {
        self.sentiment
    }

    pub fn id(&self) -> Option<u64> {
        self.id
    }
}

impl Display for Record {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{},{}",
            self.id.unwrap_or(0_u64),
            self.comment,
            self.sentiment
        )
    }
}
