use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64;
use grep_matcher::{Match, Matcher};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::Error;

use counter::CounterWriter;
use stats::Stats;

/// The configuration for the standard printer.
///
/// This is manipulated by the StandardBuilder and then referenced by the
/// actual implementation. Once a printer is build, the configuration is frozen
/// and cannot changed.
#[derive(Debug, Clone)]
struct Config {
    max_matches: Option<u64>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            max_matches: None,
        }
    }
}

/// A builder for a JSON lines printer.
///
/// The builder permits configuring how the printer behaves. The JSON printer
/// has fewer configuration options than the standard printer because it is
/// a structured format, and the printer always attempts to find the most
/// information possible.
///
/// One a printer is built, its configuration cannot be changed.
#[derive(Clone, Debug)]
pub struct JSONBuilder {
    config: Config,
}

impl JSONBuilder {
    /// Return a new builder for configuring the JSON printer.
    pub fn new() -> JSONBuilder {
        JSONBuilder { config: Config::default() }
    }

    /// Create a JSON printer that writes results to the given writer.
    pub fn build<W: io::Write>(&self, wtr: W) -> JSON<W> {
        JSON {
            config: self.config.clone(),
            wtr: CounterWriter::new(wtr),
            matches: vec![],
            stats: Stats::new()
        }
    }

    /// Set the maximum amount of matches that are printed.
    ///
    /// If multi line search is enabled and a match spans multiple lines, then
    /// that match is counted exactly once for the purposes of enforcing this
    /// limit, regardless of how many lines it spans.
    pub fn max_matches(&mut self, limit: Option<u64>) -> &mut JSONBuilder {
        self.config.max_matches = limit;
        self
    }
}

/// The JSON printer, which emits results in a JSON lines format.
#[derive(Debug)]
pub struct JSON<W> {
    config: Config,
    wtr: CounterWriter<W>,
    matches: Vec<Match>,
    stats: Stats,
}

impl<W: io::Write> JSON<W> {
    /// Return a JSON lines printer with a default configuration that writes
    /// matches to the given writer.
    pub fn new(wtr: W) -> JSON<W> {
        JSONBuilder::new().build(wtr)
    }

    /// Return an implementation of `Sink` for the JSON printer.
    ///
    /// This does not associate the printer with a file path, which means this
    /// implementation will never print a file path along with the matches.
    pub fn sink<'s, M: Matcher>(
        &'s mut self,
        matcher: M,
    ) -> JSONSink<'static, 's, M, W> {
        JSONSink {
            matcher: matcher,
            json: self,
            path: None,
            start_time: Instant::now(),
            match_count: 0,
            after_context_remaining: 0,
            binary_byte_offset: None,
        }
    }

    /// Return an implementation of `Sink` associated with a file path.
    ///
    /// When the printer is associated with a path, then it may, depending on
    /// its configuration, print the path along with the matches found.
    pub fn sink_with_path<'p, 's, M, P>(
        &'s mut self,
        matcher: M,
        path: &'p P,
    ) -> JSONSink<'p, 's, M, W>
    where M: Matcher,
          P: ?Sized + AsRef<Path>,
    {
        JSONSink {
            matcher: matcher,
            json: self,
            path: Some(path.as_ref()),
            start_time: Instant::now(),
            match_count: 0,
            after_context_remaining: 0,
            binary_byte_offset: None,
        }
    }

    /// Return a mutable reference to the underlying writer.
    pub fn get_mut(&mut self) -> &mut W {
        self.wtr.get_mut()
    }

    /// Consume this printer and return back ownership of the underlying
    /// writer.
    pub fn into_inner(self) -> W {
        self.wtr.into_inner()
    }

    /// Return a reference to the stats produced by the printer. The stats
    /// returned are cumulative over all searches performed using this printer.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
}

/// An implementation of `Sink` associated with a matcher and an optional file
/// path for the JSON printer.
#[derive(Debug)]
pub struct JSONSink<'p, 's, M: 's + Matcher, W: 's> {
    matcher: M,
    json: &'s mut JSON<W>,
    path: Option<&'p Path>,
    start_time: Instant,
    match_count: u64,
    after_context_remaining: u64,
    binary_byte_offset: Option<u64>,
}

impl<'p, 's, M: Matcher, W: io::Write> JSONSink<'p, 's, M, W> {
    /// Returns true if and only if this printer received a match in the
    /// previous search.
    ///
    /// This is unaffected by the result of searches before the previous
    /// search.
    pub fn has_match(&self) -> bool {
        self.match_count > 0
    }

    /// If binary data was found in the previous search, this returns the
    /// offset at which the binary data was first detected.
    ///
    /// The offset returned is an absolute offset relative to the entire
    /// set of bytes searched.
    ///
    /// This is unaffected by the result of searches before the previous
    /// search. e.g., If the search prior to the previous search found binary
    /// data but the previous search found no binary data, then this will
    /// return `None`.
    pub fn binary_byte_offset(&self) -> Option<u64> {
        self.binary_byte_offset
    }
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum Message {
    Begin(Begin),
    End(End),
    Summary(Summary),
    Matched(Matched),
    Context(Context),
}

#[derive(Deserialize, Serialize)]
struct Begin {
    #[serde(deserialize_with = "deser_path", serialize_with = "ser_path")]
    path: Option<PathBuf>,
}

#[derive(Deserialize, Serialize)]
struct End {
    #[serde(deserialize_with = "deser_path", serialize_with = "ser_path")]
    path: Option<PathBuf>,
    binary_offset: Option<u64>,
    stats: Stats,
}

#[derive(Deserialize, Serialize)]
struct Summary {
    stats: Stats,
}

#[derive(Deserialize, Serialize)]
struct Matched {
    #[serde(deserialize_with = "deser_path", serialize_with = "ser_path")]
    path: Option<PathBuf>,
    #[serde(deserialize_with = "deser_bytes", serialize_with = "ser_bytes")]
    lines: Vec<u8>,
    line_number: u64,
    absolute_offset: u64,
    matches: Vec<Range>,
}

#[derive(Deserialize, Serialize)]
struct Range {
    start: usize,
    end: usize,
}

#[derive(Deserialize, Serialize)]
struct Context {
    #[serde(deserialize_with = "deser_path", serialize_with = "ser_path")]
    path: Option<PathBuf>,
    #[serde(deserialize_with = "deser_bytes", serialize_with = "ser_bytes")]
    lines: Vec<u8>,
    line_number: u64,
    absolute_offset: u64,
}

/// Data represents things that look like strings, but may actually not be
/// valid UTF-8.
///
/// The happy path is valid UTF-8, which streams right through as-is, since
/// it is natively supported by JSON. When invalid UTF-8 is found, then it is
/// represented as arbitrary bytes and base64 encoded.
#[derive(Clone, Debug, Deserialize, Hash, PartialEq, Eq, Serialize)]
#[serde(untagged)]
enum Data {
    Text { text: String },
    Bytes {
        #[serde(
            deserialize_with = "from_base64",
            serialize_with = "to_base64",
        )]
        bytes: Vec<u8>,
    },
}

impl Data {
    fn from_bytes(bytes: &[u8]) -> Data {
        match str::from_utf8(bytes) {
            Ok(text) => Data::Text { text: text.to_string() },
            Err(_) => Data::Bytes { bytes: bytes.to_vec() },
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        match self {
            Data::Text { text } => text.into_bytes(),
            Data::Bytes { bytes } => bytes,
        }
    }

    #[cfg(unix)]
    fn from_path(path: &Path) -> Data {
        use std::os::unix::ffi::OsStrExt;

        match path.to_str() {
            Some(s) => Data::Text { text: s.to_string() },
            None => {
                Data::Bytes { bytes: path.as_os_str().as_bytes().to_vec() }
            }
        }
    }

    #[cfg(not(unix))]
    fn from_path(path: &Path) -> Data {
        // Using lossy conversion means some paths won't round trip precisely,
        // but it's not clear what we should actually do. Serde rejects
        // non-UTF-8 paths, and OsStr's are serialized as a sequence of UTF-16
        // code units on Windows. Neither seem appropriate for this use case,
        // so we do the easy thing for now.
        Data::Text { text: path.as_ref().to_string_lossy().into_owned() }
    }

    #[cfg(unix)]
    fn into_path_buf(&self) -> PathBuf {
        use std::os::unix::ffi::OsStrExt;

        match self {
            Data::Text { text } => PathBuf::from(text),
            Data::Bytes { bytes } => {
                PathBuf::from(OsStr::from_bytes(bytes))
            }
        }
    }

    #[cfg(not(unix))]
    fn into_path_buf(&self) -> PathBuf {
        match self {
            Data::Text { text } => PathBuf::from(text),
            Data::Bytes { bytes } => {
                PathBuf::from(String::from_utf8_lossy(&bytes).into_owned())
            }
        }
    }
}

fn to_base64<T, S>(
    bytes: T,
    ser: S,
) -> Result<S::Ok, S::Error>
where T: AsRef<[u8]>,
      S: Serializer
{
    ser.serialize_str(&base64::encode(&bytes))
}

fn from_base64<'de, D>(
    de: D,
) -> Result<Vec<u8>, D::Error>
where D: Deserializer<'de>
{
    let encoded = String::deserialize(de)?;
    let decoded = base64::decode(encoded.as_bytes())
        .map_err(D::Error::custom)?;
    Ok(decoded)
}

fn ser_bytes<T, S>(
    bytes: T,
    ser: S,
) -> Result<S::Ok, S::Error>
where T: AsRef<[u8]>,
      S: Serializer
{
    Data::from_bytes(bytes.as_ref()).serialize(ser)
}

fn deser_bytes<'de, D>(
    de: D,
) -> Result<Vec<u8>, D::Error>
where D: Deserializer<'de>
{
    Data::deserialize(de).map(|datum| datum.into_bytes())
}

fn ser_path<P, S>(
    path: &Option<P>,
    ser: S,
) -> Result<S::Ok, S::Error>
where P: AsRef<Path>,
      S: Serializer
{
    path.as_ref().map(|p| Data::from_path(p.as_ref())).serialize(ser)
}

fn deser_path<'de, D>(
    de: D,
) -> Result<Option<PathBuf>, D::Error>
where D: Deserializer<'de>
{
    Option::<Data>::deserialize(de)
        .map(|opt| opt.map(|datum| datum.into_path_buf()))
}

#[cfg(test)]
mod tests {
    use serde_json as json;
    use super::*;

    #[test]
    fn scratch() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        // let raw = OsStr::from_bytes(b"/home/and\xFFrew/rust/ripgrep");
        // let path = PathBuf::from(raw);
        let path = PathBuf::from("/home/andrew/rust/ripgrep");
        let msg = Message::Begin(Begin {
            path: Some(path),
        });
        let out = json::to_string_pretty(&msg).unwrap();
        println!("{}", out);
    }
}
