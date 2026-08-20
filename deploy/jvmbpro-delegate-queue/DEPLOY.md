# Deploying the durable delegate queue on JVMBPro

Target: JVMBPro, user `jasonvassallo` (uid 501). The service binds
`127.0.0.1:11438`; remote access is added at the edge via the existing
cloudflared tunnel + a Cloudflare Access service-token policy — the same
pattern as `ollama-mbp.djvassallo.com`.

Design doc: `docs/superpowers/specs/2026-08-07-durable-delegate-queue-design.md`.

**Where each step runs.** Steps are labelled **[JVMBPro]** or
**[Mini]**. Two constraints drive the labelling and are not optional:

1. **Step 1 needs a TTY.** It calls `sudo`, and `sudo` on JVMBPro has no
   NOPASSWD rule and no Touch ID, so it must read a password from a
   terminal. Run step 1 in an interactive local terminal on JVMBPro, or
   over an `ssh -t jvmbpro` session. A plain non-interactive
   `ssh jvmbpro '<command>'` has no TTY and fails with
   `sudo: a terminal is required to read the password`.
2. **JVMBPro's only checkout of this repo is its LIVE MCP server.**
   `/Users/jasonvassallo/Documents/Code/ai-tools-mcp` is the path
   `~/.claude.json` runs (`uv run <that path>/mcp_server.py`). It sits on
   `main`. **Never `git checkout`, `git switch`, `git pull`, or
   otherwise move that checkout to deploy this branch** — doing so swaps
   the live MCP server out from under every running client. Step 2 below
   installs without touching it.

## 1. Provision the encryption key (System keychain, secret never printed)

**[JVMBPro, interactive terminal — see constraint 1 above]**

The service refuses to start without a 32-byte AES-256 key stored
base64-encoded in the **System** keychain (service `DELEGATE_QUEUE_KEY`,
account `jasonvassallo`). `queue_server.py` reads it with the System
keychain path passed explicitly, so a login-keychain item of the same
name is neither used nor a substitute.

### 1a. Verify the keychain writer is the v3 zsh wrapper

Not the retired v2 Swift binary — v2's detached ACL write silently
produced GUI-prompting items:

```bash
head -3 ~/.local/bin/keychain-write
# expect: "#!/bin/zsh" and a "(v3, ...)" comment; if it's a Mach-O
# binary, STOP and restore the v3 wrapper first.
```

### 1b. Prime the sudo timestamp FIRST, as its own step

Do this before the generate-and-store pipeline. The pipeline's `sudo`
would otherwise prompt for a password *in the middle of* the pipeline,
with `openssl` already having produced key material upstream of it:

```bash
sudo -v
# enter your password; the timestamp is now valid for this terminal
```

### 1c. Generate and store the key in ONE pipeline

The material never touches argv, a shell variable, or the terminal —
`keychain-write` reads the secret on stdin and delivers it to
`security -i` on stdin:

```bash
openssl rand 32 | base64 | sudo ~/.local/bin/keychain-write \
  DELEGATE_QUEUE_KEY jasonvassallo /Library/Keychains/System.keychain
```

> **Do not blindly re-run this on failure.** Every run generates a
> *different* key. If a run fails partway (sudo timeout, typo, no TTY),
> re-running is safe only because `keychain-write` replaces the existing
> entry — but if the service has already encrypted jobs under an earlier
> key, replacing it makes every stored job undecryptable. Before a
> retry: confirm with step 1d whether an item already exists, and if the
> service has ever run with jobs on disk, stop and decide deliberately
> (rotating the key means discarding
> `~/.local/state/delegate-queue/queue.db`).

### 1d. Verify presence WITHOUT printing the value

`-w` is load-bearing and is **not** just "print the value" — the
redirect already discards output. Without `-w`, `security` returns the
item's *attributes* and never touches its data, so it never exercises
the ACL/unlock authorization that reading the secret requires: a
GUI-prompting item (the exact v2-writer regression step 1a exists to
catch) still reports `present` here, and then fails at service start
under launchd where no interactive authorization path exists. With `-w`
this is byte-for-byte the argv `queue_server.py` runs:

```bash
security find-generic-password -s DELEGATE_QUEUE_KEY -a jasonvassallo -w \
  /Library/Keychains/System.keychain >/dev/null && echo present
```

If this hangs instead of returning promptly, the item is prompting for
authorization — that is the failure, not a slow keychain. Fix it here
rather than at step 2d.

## 2. Install the service + LaunchAgent (without touching the live checkout)

Copy the two files to JVMBPro from the Mini's worktree. This is the
non-destructive route: it never runs git on JVMBPro, so the live MCP
checkout at `~/Documents/Code/ai-tools-mcp` stays exactly where it is.

### 2a. [Mini] Create the staging directory

Single-quoted so the paths expand on JVMBPro, not on the Mini:

```bash
ssh jvmbpro 'mkdir -p ~/.local/bin ~/Library/LaunchAgents ~/.cache/delegate-queue-staging'
```

### 2b. [Mini] Copy the two files from this branch's worktree

Run from the root of a checkout of `feat/durable-delegate-queue` on the
Mini:

```bash
scp queue_server.py jvmbpro:.cache/delegate-queue-staging/queue_server.py
scp deploy/jvmbpro-delegate-queue/com.jasonvassallo.delegate-queue.plist \
  jvmbpro:.cache/delegate-queue-staging/com.jasonvassallo.delegate-queue.plist
```

### 2c. [JVMBPro] Install from staging and bootstrap

```bash
install -m 0755 ~/.cache/delegate-queue-staging/queue_server.py \
  ~/.local/bin/queue_server.py
install -m 0644 \
  ~/.cache/delegate-queue-staging/com.jasonvassallo.delegate-queue.plist \
  ~/Library/LaunchAgents/com.jasonvassallo.delegate-queue.plist
launchctl bootstrap gui/501 \
  ~/Library/LaunchAgents/com.jasonvassallo.delegate-queue.plist
```

The plist runs `/opt/homebrew/bin/uv run ~/.local/bin/queue_server.py`;
`uv` resolves the inline `cryptography` dependency on first start, so
allow a few seconds before the health check.

> **Alternative to 2a–2c**, if you would rather pull on JVMBPro than
> push from the Mini: clone into a **separate throwaway directory** —
> never the live checkout —
> `gh repo clone jasonvassallo/ai-tools-mcp ~/.cache/delegate-queue-src -- -b feat/durable-delegate-queue`,
> install the two files from there, then `rm -rf ~/.cache/delegate-queue-src`.
> (`gh` so the clone authenticates the same way the rest of the fleet
> does, rather than depending on the repo being publicly readable.)

### 2d. [JVMBPro] Check it came up

Fail-closed: it exits 1 and throttle-loops if the key is missing — see
`~/Library/Logs/delegate-queue.log`.

```bash
curl -s http://127.0.0.1:11438/healthz
# expect: {"status": "ok", "queued": 0, "running": 0}
```

### 2e. Re-deploy after a code update

Repeat 2b, then **[JVMBPro]**:

```bash
install -m 0755 ~/.cache/delegate-queue-staging/queue_server.py \
  ~/.local/bin/queue_server.py
launchctl kickstart -k gui/501/com.jasonvassallo.delegate-queue
```

## Steps 3–5: build the gate before the front door

> **Order is a security control here, not a preference.** The service
> has no authentication of its own — it trusts every loopback caller by
> design, and Cloudflare Access is the *only* remote gate. The proxied
> DNS record is what makes the tunnel reachable from the internet, so it
> must be the **last** thing created. Do not create the CNAME in step 5
> until both the Access application (step 3) and the origin `access`
> block (step 4) are in place. Creating it earlier leaves a window in
> which anyone can `POST /v1/jobs` and run inference on JVMBPro,
> retrieve the result, and fill the 64-job queue so your own submits
> draw 429. The hostname is not obscure — it is compiled into this
> repo's default endpoint chain and this repo is public.

The values behind `<TUNNEL_UUID>`, `<ZONE_ID>`, `<TEAM_NAME>` and
`<AUD_TAG>` are deliberately kept out of this repo — it is public. Each
step below says how to read its value locally; do not paste the
resolved values back into this file.

## 3. Cloudflare Access application (create the gate FIRST)

An Access application is keyed on a hostname in one of your zones and
does not require that hostname's DNS record to exist yet — which is
exactly why this comes before steps 4 and 5. The dashboard may warn that
no DNS record exists; that warning is expected here, and saving is still
correct.

> **If the dashboard refuses to save without a record**, do *not*
> jump ahead to step 5 as written. Create the CNAME **unproxied**
> (grey cloud, `"proxied": false`) instead: an unproxied CNAME to
> `<TUNNEL_UUID>.cfargotunnel.com` does not route traffic, so it exposes
> nothing. Then finish this step, do step 4, and only then flip the
> record to proxied — which is what step 5's `"proxied": true` does on a
> record that already exists (use `PATCH .../dns_records/<RECORD_ID>`
> rather than `POST`).

In Zero Trust → Access → Applications, add a self-hosted application:

- Application domain: `queue-mbp.djvassallo.com`
- Policy: action **Service Auth** (`non_identity` decision), include →
  Service Token → the **same token the MCP client already sends**.

  Identify it by client-id, not by name. The `ollama-mbp` application's
  single `non_identity` policy includes **five** service tokens, and
  only one of them is the one the MCP client actually presents.
  Including any of the other four instead makes the step-6 verification
  curl return **403**, with nothing in the response pointing back at the
  cause. Print just enough of the client-id to disambiguate:

  ```bash
  security find-generic-password -s OLLAMA_CF_ACCESS_CLIENT_ID -w | cut -c1-8
  ```

  Pick the Zero Trust service token whose Client ID starts with those
  eight characters.

  No new token: the MCP client sends the same
  `OLLAMA_CF_ACCESS_CLIENT_ID`/`_SECRET` Keychain credentials for both
  hostnames by design.
- No browser/identity policy needed — this hostname is API-only.

After saving, copy the application's **Application Audience (AUD) tag**
— that is `<AUD_TAG>` in step 4. Your Zero Trust **team name** (the
`<TEAM_NAME>` in step 4) is shown in the Zero Trust dashboard under
Settings → Custom Pages, and in your team domain
`<TEAM_NAME>.cloudflareaccess.com`.

## 4. cloudflared ingress (requires Jason's sudo)

**[JVMBPro, interactive terminal]** Add to `/etc/cloudflared/config.yml`
in the `ingress:` list **BEFORE the catch-all**
`- service: http_status:404` rule (ordering is how cloudflared routes).
There is no `queue-mbp` entry yet. Confirm which tunnel the file belongs
to with `cloudflared tunnel list` if you need `<TUNNEL_UUID>` for step 5.

Add the block complete, with the real AUD tag from step 3 already filled
in — do not land a hostname whose `access` block is still a placeholder:

```yaml
  - hostname: queue-mbp.djvassallo.com
    # 127.0.0.1 literally: make_server() binds the IPv4 loopback only,
    # and a "localhost" that resolves to ::1 first would miss it.
    service: http://127.0.0.1:11438
    # Defence in depth, matching the sibling ollama-mbp entry: the origin
    # itself validates the Access JWT, so a request that reaches the
    # tunnel by any other route is still rejected.
    originRequest:
      access:
        required: true
        teamName: <TEAM_NAME>
        audTag:
          - <AUD_TAG>
```

> **Stale config warning:** a `~/.cloudflared/config.yml` can exist from an
> earlier user-level setup and is never read by the daemon. Any bare
> `cloudflared tunnel ...` invocation (no `--config`) silently validates or
> reports against that stale file instead of the live one. The daemon's
> authoritative config is whatever `--config` its LaunchDaemon plist names —
> on JVMBPro that's `/etc/cloudflared/config.yml`, per the
> `com.cloudflare.cloudflared` plist at
> `/Library/LaunchDaemons/com.cloudflare.cloudflared.plist`. Always pass
> `--config` explicitly.

Validate the edited config before restarting — as a **global tunnel flag
before the subcommand** (`ingress validate --config ...` is rejected; flag
position matters). Chain the validation, the rule check, and the restart
(system daemon, per the one-tunnel-per-machine setup — installed via
`cloudflared service install`, label `com.cloudflare.cloudflared`, running
`/opt/homebrew/bin/cloudflared tunnel --config /etc/cloudflared/config.yml run`)
so a failed check stops before anything restarts. Note `ingress rule`
exits 0 as soon as ANY rule matches, including the catch-all
`http_status:404` — a missing or mistyped hostname entry still "succeeds"
by falling through to the catch-all, so grep the output for the expected
service to actually catch that case:

```bash
cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate && \
cloudflared tunnel --config /etc/cloudflared/config.yml ingress rule https://queue-mbp.djvassallo.com \
  | grep -q 'http://127.0.0.1:11438' && \
sudo launchctl kickstart -k system/com.cloudflare.cloudflared
```

## 5. DNS CNAME (Cloudflare API, token from Keychain) — do this LAST

This is the step that publishes the hostname. Do not run it until steps
3 and 4 are both done.

Uses the `CLOUDFLARE_API_TOKEN` Keychain item; the token is piped
straight from the Keychain into curl's header reader on stdin (`-H @-`,
verified on JVMBPro's curl 8.7.1), so it never appears in any process
argument list, shell variable, or environment.

`<ZONE_ID>` is the `djvassallo.com` zone id — read it off the zone's
dashboard Overview pane, or list it with the same Keychain-piped token:

```bash
security find-generic-password -s CLOUDFLARE_API_TOKEN -w |
  sed 's/^/Authorization: Bearer /' |
  curl -s "https://api.cloudflare.com/client/v4/zones?name=djvassallo.com" \
    -H @- | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"][0]["id"])'
```

Then create the record, routing `queue-mbp.djvassallo.com` at JVMBPro's
tunnel (`<TUNNEL_UUID>` from `cloudflared tunnel list` in step 4):

```bash
security find-generic-password -s CLOUDFLARE_API_TOKEN -w |
  sed 's/^/Authorization: Bearer /' |
  curl -s -X POST \
    "https://api.cloudflare.com/client/v4/zones/<ZONE_ID>/dns_records" \
    -H @- \
    -H "Content-Type: application/json" \
    --data '{
      "type": "CNAME",
      "name": "queue-mbp",
      "content": "<TUNNEL_UUID>.cfargotunnel.com",
      "proxied": true
    }'
```

## 6. Verify end to end

```bash
# Bare request is rejected at the edge:
curl -s -o /dev/null -w '%{http_code}\n' https://queue-mbp.djvassallo.com/healthz
# expect: 403

# With the service-token headers it reaches the queue. Both header
# values are piped from the Keychain to curl's stdin header reader
# (-H @-), so neither ever appears in a process argument list, shell
# variable, or environment:
{
  security find-generic-password -s OLLAMA_CF_ACCESS_CLIENT_ID -w |
    sed 's/^/CF-Access-Client-Id: /'
  security find-generic-password -s OLLAMA_CF_ACCESS_CLIENT_SECRET -w |
    sed 's/^/CF-Access-Client-Secret: /'
} | curl -s https://queue-mbp.djvassallo.com/healthz -H @-
# expect: {"status": "ok", "queued": 0, "running": 0}
```

A 403 on the second call almost always means the Access policy in step 3
includes a *different* one of the five Ollama service tokens — recheck
it against the `OLLAMA_CF_ACCESS_CLIENT_ID` client-id prefix as step 3
describes.

Then from any machine running the MCP server, a
`local_delegate(background=true)` call should return a `q`-prefixed
job_id with `"queue": "durable"` in the envelope, and
`local_delegate_result` should keep returning the result after an MCP
server restart (the durability contract the in-memory store cannot
offer).
