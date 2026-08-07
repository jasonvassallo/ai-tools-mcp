# Deploying the durable delegate queue on JVMBPro

Target: JVMBPro, user `jasonvassallo` (uid 501). All steps run ON
JVMBPro (locally or over `ssh jvmbpro`) unless noted. The service binds
`127.0.0.1:11438`; remote access is added at the edge via the existing
cloudflared tunnel + a Cloudflare Access service-token policy — the same
pattern as `ollama-mbp.djvassallo.com`.

Design doc: `docs/superpowers/specs/2026-08-07-durable-delegate-queue-design.md`.

## 1. Provision the encryption key (System keychain, secret never printed)

The service refuses to start without a 32-byte AES-256 key stored
base64-encoded in the **System** keychain (service `DELEGATE_QUEUE_KEY`,
account `jasonvassallo`).

First verify the keychain writer is the v3 zsh wrapper (NOT the retired
v2 Swift binary — v2's detached ACL write silently produced
GUI-prompting items):

```bash
head -3 ~/.local/bin/keychain-write
# expect: "#!/bin/zsh" and a "(v3, ...)" comment; if it's a Mach-O
# binary, STOP and restore the v3 wrapper first.
```

Generate and store the key in ONE pipeline so the material never
touches argv, a shell variable, or the terminal (keychain-write reads
the secret on stdin and delivers it to `security -i` on stdin;
System-keychain writes prompt for sudo):

```bash
openssl rand 32 | base64 | sudo ~/.local/bin/keychain-write \
  DELEGATE_QUEUE_KEY jasonvassallo /Library/Keychains/System.keychain
```

Verify presence WITHOUT printing the value:

```bash
security find-generic-password -s DELEGATE_QUEUE_KEY -a jasonvassallo \
  /Library/Keychains/System.keychain >/dev/null && echo present
```

## 2. Install the service + LaunchAgent

From a checkout of this repo on JVMBPro:

```bash
install -m 0755 queue_server.py ~/.local/bin/queue_server.py
install -m 0644 deploy/jvmbpro-delegate-queue/com.jasonvassallo.delegate-queue.plist \
  ~/Library/LaunchAgents/com.jasonvassallo.delegate-queue.plist
launchctl bootstrap gui/501 \
  ~/Library/LaunchAgents/com.jasonvassallo.delegate-queue.plist
```

Check it came up (fail-closed: it exits 1 and throttle-loops if the key
is missing — see `~/Library/Logs/delegate-queue.log`):

```bash
curl -s http://127.0.0.1:11438/healthz
# expect: {"status": "ok", "queued": 0, "running": 0}
```

Re-deploy after a code update:

```bash
install -m 0755 queue_server.py ~/.local/bin/queue_server.py
launchctl kickstart -k gui/501/com.jasonvassallo.delegate-queue
```

## 3. cloudflared ingress (requires Jason's sudo)

Add to `/etc/cloudflared/config.yml` on JVMBPro, in the `ingress:`
list **BEFORE the catch-all** `- service: http_status:404` rule
(ordering is how cloudflared routes):

```yaml
  - hostname: queue-mbp.djvassallo.com
    service: http://localhost:11438
```

Then restart the tunnel (system daemon, per the one-tunnel-per-machine
setup):

```bash
sudo launchctl kickstart -k system/homebrew.mxcl.cloudflared
```

## 4. DNS CNAME (Cloudflare API, token from Keychain)

Route `queue-mbp.djvassallo.com` at JVMBPro's tunnel. Uses the
`CLOUDFLARE_API_TOKEN` Keychain item (zone `djvassallo.com`, id
`d8edb0a8…` — fill from the zone list if needed); the token is read
into the process environment, never echoed:

```bash
CF_TOKEN=$(security find-generic-password -s CLOUDFLARE_API_TOKEN -w)
curl -s -X POST \
  "https://api.cloudflare.com/client/v4/zones/<ZONE_ID>/dns_records" \
  -H "Authorization: Bearer $CF_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{
    "type": "CNAME",
    "name": "queue-mbp",
    "content": "47b3a9bb-7c29-421d-b7ad-c2739652f9d2.cfargotunnel.com",
    "proxied": true
  }'
unset CF_TOKEN
```

## 5. Cloudflare Access application (reuse the existing Ollama service token)

In Zero Trust → Access → Applications, add a self-hosted application:

- Application domain: `queue-mbp.djvassallo.com`
- Policy: action **Service Auth** (`non_identity` decision), include →
  Service Token → the EXISTING Ollama Access service token (the one
  already gating `ollama-mbp.djvassallo.com`). No new token: the MCP
  client sends the same `OLLAMA_CF_ACCESS_CLIENT_ID`/`_SECRET`
  Keychain credentials for both hostnames by design.
- No browser/identity policy needed — this hostname is API-only.

## 6. Verify end to end

```bash
# Bare request is rejected at the edge:
curl -s -o /dev/null -w '%{http_code}\n' https://queue-mbp.djvassallo.com/healthz
# expect: 403

# With the service-token headers it reaches the queue (values read from
# the Keychain into variables, never typed inline):
CF_ID=$(security find-generic-password -s OLLAMA_CF_ACCESS_CLIENT_ID -w)
CF_SECRET=$(security find-generic-password -s OLLAMA_CF_ACCESS_CLIENT_SECRET -w)
curl -s https://queue-mbp.djvassallo.com/healthz \
  -H "CF-Access-Client-Id: $CF_ID" \
  -H "CF-Access-Client-Secret: $CF_SECRET"
unset CF_ID CF_SECRET
# expect: {"status": "ok", ...}
```

Then from any machine running the MCP server, a
`local_delegate(background=true)` call should return a `q`-prefixed
job_id with `"queue": "durable"` in the envelope, and
`local_delegate_result` should keep returning the result after an MCP
server restart (the durability contract the in-memory store cannot
offer).
