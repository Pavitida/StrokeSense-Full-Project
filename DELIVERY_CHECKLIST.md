# Delivery checklist

## Implemented in this local release

- Home, instructions/limitations, separate Face/Speech/Movement steps, Result and permission/error help.
- Responsive layout and collapsible sidebar; desktop/mobile rendering still requires visual QA.
- Native camera photo/audio capture; browser video recorder with start/stop, preview, delete and 30-second cap.
- Processing consent before camera pages; separate disabled research-consent option.
- Per-file type/content/size/duration limits; generated server filenames; no user pickle/executable upload.
- Authenticated multipart API, model paths in private temporary directories, JSON partial results.
- Inference busy lock and UI processing state; cleanup on handled errors and success.
- Preserve independent results; no fusion number when missing; no stroke-probability percentages.
- Emergency message, clear prototype disclaimer, New Test/Home actions.
- Session-scoped media/results, cancellation/reset and active-session inactivity expiry; no automatic Data Collection.
- Loopback-only binding, ephemeral API token, no access logs; source/runtime/data separation.
- Updated setup/API/privacy documentation and regression tests.

## Verification evidence

Python unittest suite: 25 tests passed with 1 legacy fixture-dependent class skipped. Live HTTP API checked with actual model inputs; missing Fusion returns no score. Mocked tests are not a substitute for hardware/browser verification.

## Still requires a real deployment or target device

- Safari macOS/iOS and physical mobile camera/microphone, orientation and media-codec checks.
- Real hardware permission denial/grant/revocation in Chrome and Safari; recorder mock tests only cover code paths.
- Full Home → camera capture → actual user result flow with the user's devices. Do not claim that recordings were made without performing this test.
- HTTPS domain/certificate, reverse proxy, authenticated multi-user access and configured crash-recovery temp cleanup.
- Research governance and retention/withdrawal process before enabling data collection (currently disabled).
- Model validation on independently held-out participants, GPU/CPU discrepancy investigation, Thai speech validation and an appropriate arm-raise model.
- Fusion training on truly paired, labeled participants; the production model is missing.

## GitHub handoff

Source changes will be placed on a reviewable branch. No datasets, personal recordings, tokens or local runtime binaries should be pushed. Push/PR completion must be confirmed from the remote, not inferred from a local commit.
