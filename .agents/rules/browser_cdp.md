# Browser Automation & Testing — Chrome CDP

## MANDATORY RULE: Never Use Playwright Downloads

Playwright CDN driver downloads fail in this environment. Never attempt to install or download Playwright browser binaries.

Instead, always use an already-running Google Chrome instance connected via **Chrome DevTools Protocol (CDP)** on port **9222**.

### Launch Command

When Chrome needs to be started for browser testing / automation:

**CMD / Batch:**
```cmd
chrome.exe --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\remote-debug-profile"
```

**PowerShell:**
```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 `
  --remote-debugging-address=127.0.0.1 `
  --user-data-dir="$env:LOCALAPPDATA\Google\Chrome\remote-debug-profile"
```

### CDP Endpoint
- **URL:** `http://127.0.0.1:9222`
- **Verification:** `http://127.0.0.1:9222/json/version`
