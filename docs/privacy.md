# Privacy

Portraits are sensitive personal data even though this application deliberately avoids recognition and biometric templates.

## Data processed

- Uploaded input/reference/style images.
- Normalized images, masks, landmarks, crops, backgrounds, eye assets, outputs, and optional diagnostics.
- Operational metadata such as dimensions, byte count, SHA-256, quality scores, processing settings, timestamps, and safe error codes.
- Account identifier or anonymous session ownership token.

Landmarks exist only to perform the requested edit. The system stores no persistent face-recognition embedding and performs no identity lookup, face search, demographic classification, or training on uploads.

## Defaults

- Objects and styles are private unless an authorized owner explicitly publishes a style.
- Asset/job expiry is 24 hours by default.
- Every output and input has a visible deletion path.
- Downloads use short-lived signed URLs.
- EXIF and other metadata are stripped from normalized inputs and outputs.
- Client filenames are replaced with generated identifiers.
- Debug artifacts use the same owner and expiry as the job.

Deletion removes object bytes first, then soft-deletes metadata. Failed object deletions remain eligible for retry and produce an audit event/metric. Operators must define how encrypted backups age out and disclose any longer backup retention.

## User responsibilities

Users must have permission to process uploaded people and must affirm rights for style-library examples. The product is not seeded with celebrity portraits, famous photographers' work, or research-paper figures. Public-style publication requires an auditable explicit action.

## Logging

Logs may include request/job ID, hashed session ID, stage, timing, dimensions, worker/backend, algorithm version, and safe status. Logs never include pixels, email address, raw filename, object key, signed URL, or correction coordinates that could reconstruct an image.

## Operator responsibilities

Deployers must select a lawful basis, publish jurisdiction-appropriate notices, execute processor agreements, limit staff access, configure geographic/storage controls, respond to deletion/access requests, and perform a privacy impact review before real-user launch. This document describes software defaults; it is not legal advice.
