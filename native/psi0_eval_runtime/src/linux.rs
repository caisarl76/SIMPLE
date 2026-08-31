use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;

pub fn run_self_test(binary_name: &str) -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 1 || arguments[0] != "--self-test" {
        return Err("the only supported initial operation is --self-test".into());
    }
    let executable = fs::read("/proc/self/exe")?;
    let digest = hex::encode(Sha256::digest(&executable));
    let evidence = json!({
        "binary": binary_name,
        "elf_bytes": executable.len(),
        "elf_sha256": digest,
        "schema_version": 1,
        "static_expected": true,
    });
    println!("{}", serde_json::to_string(&evidence)?);
    Ok(())
}
