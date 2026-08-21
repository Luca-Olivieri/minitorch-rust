use std::error::Error;
use std::fmt::self;

#[derive(Debug)]
pub struct TorchError {
    pub kind: ErrorKind,
    pub message: String
}

impl Error for TorchError {}

impl fmt::Display for TorchError {

    fn fmt(
        &self,
        f: &mut fmt::Formatter
    ) -> fmt::Result {
        // write!(f, "TorchError[kind={}], message: {}]", self.kind, self.message)
        write!(f, "{}", format!("{:?}", self))
    }
}

#[derive(Debug)]
pub enum ErrorKind {
    ValidationError
}

impl fmt::Display for ErrorKind {

    fn fmt(
        &self,
        f: &mut fmt::Formatter
    ) -> fmt::Result {
        write!(f, "{}", self)
    }
}
