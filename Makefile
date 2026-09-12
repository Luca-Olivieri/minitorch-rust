run:
	@cargo run

run-release:
	@cargo run --release

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
