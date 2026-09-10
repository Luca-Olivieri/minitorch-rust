run:
	cargo run --release

run-debug:
	cargo run

release:
	cargo build --release

check:
	cargo check
	cargo fmt --check
	cargo clippy -- -D warnings

fmt:
	cargo fmt
