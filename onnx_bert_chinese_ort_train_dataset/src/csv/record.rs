use std::fmt::Display;

use serde::Deserialize;

#[derive(Debug, Deserialize, Eq, PartialEq, Clone)]
pub struct Record {
    id: Option<u64>,
    text: String,
    label: u8,
}

impl Record {
    pub fn new(id: Option<u64>, text: String, label: u8) -> Self {
        Self {
            id,
            text,
            label,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn label(&self) -> u8 {
        self.label
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
            self.text,
            self.label
        )
    }
}
