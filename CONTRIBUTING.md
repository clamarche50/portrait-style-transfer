# Contributing

Contributions must preserve the clean-room, privacy, and deterministic-computation boundaries of this project.

## Setup

1. Install Node.js 22, Python 3.12, uv, and Docker Compose.
2. Copy `.env.example` to `.env` and replace development-only secrets.
3. Run `make bootstrap`; explicitly run `make models` only when real-model work is needed.
4. Run the smallest relevant tests while developing and `make lint typecheck test build` before review.

## Rules

- Write code, schemas, logs, documentation, and tests in English.
- Do not copy code from the uploaded archive or redistribute its files.
- Do not commit portraits, user uploads, model binaries, generated outputs, signed URLs, secrets, or private fixtures.
- Use synthetic fixtures in required CI. Keep consented real-model fixtures under ignored `tests/fixtures/private/`.
- Do not add face recognition, identity search, demographic classifiers, persistent face embeddings, neural/diffusion style transfer, or request-time model downloads.
- Keep `PAPER_EXACT` the public default. Gate compatibility experiments from public API input.
- Document every algorithm deviation and add numeric tests for changed equations.
- Preserve deterministic seeds and explicit map-coordinate conventions.
- Add migrations for database changes; never edit an applied migration.
- Update OpenAPI-generated client output when the API changes.

## Pull requests

Keep changes focused, explain privacy/security implications, list commands actually run, and distinguish mocked tests from real-model tests. Include before/after numeric diagnostics for algorithm changes, never private images. New dependencies require a license review and `THIRD_PARTY_NOTICES.md` update.

Do not leave `TODO`, `pass`, `NotImplementedError`, placeholder endpoints, or fake success responses in production paths.
