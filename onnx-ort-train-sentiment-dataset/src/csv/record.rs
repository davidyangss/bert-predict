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
