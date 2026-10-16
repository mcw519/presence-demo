# Security policy

## Supported versions

Only the latest release on the default branch receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use the repository's private vulnerability
reporting (GitHub: *Security → Report a vulnerability*). Include what you found, how to reproduce it and the
impact you expect. You should get a first reply within a week. Please give us a reasonable time to fix the
problem before you disclose it.

請不要用公開 issue 回報安全問題，請使用 repository 的私密漏洞回報功能（GitHub：*Security → Report a vulnerability*）。

## Scope and deployment notes

presence-demo is a demo and research tool. When you run it:

- Set `PRESENCE_TOKEN` whenever anyone else can reach the server, and serve it over HTTPS. Without a token every
  endpoint and the WebSocket are open.
- API keys (`ASSEMBLYAI_API_KEY`, `TYPESAFE_API_KEY`) are kept in process memory and shared by all connections.
- `PRESENCE_NPU_DELEGATE` loads a native library; it can only be set from the server environment.
- The optional cloud features (speech recognition, re-ID) send audio or face crops to third parties; see
  [docs/privacy.md](docs/privacy.md).

Issues we consider security bugs include: a way to read camera frames, audio, face crops, keys or the token
without authorisation; anything that writes images or audio to the server's disk; secrets appearing in logs or
responses; and ways to run code through the WebSocket or HTTP API.
