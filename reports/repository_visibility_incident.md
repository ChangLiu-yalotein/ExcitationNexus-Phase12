# Repository Visibility Incident

## Status

`RESOLVED_PRIVATE_VERIFIED`

Gate 1-A1 did not start because the repository was anonymously accessible at the required preflight check.

## Repository and initial publication

- Repository: `ChangLiu-yalotein/ExcitationNexus-Phase12`
- Initial bootstrap commit: `e5e5e7869b1162ab6b05fd575386fbb69e8910c4`
- Initial Gate tag: `gate0d-done-20260719`
- The repository was publicly accessible for at least part of the interval after the initial push.
- User-provided screenshot evidence at the time showed 0 forks, 0 stars, and 0 watching.
- Those counters do not prove that no anonymous clone or download occurred.

## Anonymous visibility verification

- Verification time (UTC): `2026-07-19T06:12:32Z`
- Unauthenticated GitHub API request: HTTP 403 due to shared-IP API rate limiting; this result is inconclusive for visibility.
- Unauthenticated repository web request: HTTP 200.
- Returned page title: `GitHub - ChangLiu-yalotein/ExcitationNexus-Phase12 · GitHub`.
- Conclusion: the repository was still public at verification time. A private repository is expected to be unavailable to an unauthenticated requester.

## Containment

- Gate 1-A1 asset discovery, preregistration, and training were not started.
- No additional research assets were committed or pushed during this Gate attempt.
- No token, Deploy Key private material, or credential was read or recorded.
- The user must change the repository visibility to Private before Gate 1-A1 can resume.

## Required re-verification

After the repository is changed to Private, repeat an unauthenticated request from a non-authenticated context. Gate 1-A1 may resume only after the repository endpoint is inaccessible anonymously (normally HTTP 404) and the verification time is appended to this report.

## Resolution

- Private re-verification time (UTC): `2026-07-19T06:14:47Z`
- Unauthenticated repository web request: HTTP 404.
- Returned page title: `Page not found · GitHub · GitHub`.
- Conclusion: anonymous access was no longer available, so the Gate 1-A1 repository-visibility precondition was satisfied.
