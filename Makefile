# BIN ?= xor
BIN = mnist

run:
	@cargo run --bin $(BIN)

run-release:
	@cargo run --release --bin $(BIN)

release:
	@cargo build --release

check:
	@cargo check
	@cargo fmt --check
	@cargo clippy -- -D warnings

fmt:
	@cargo fmt

test:
	@cargo test
