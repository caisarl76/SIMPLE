fn main() {
    if let Err(error) = psi0_eval_runtime::linux::run_self_test("psi0-eval-policy-relay") {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
