---
name: lucy
description: Lucy's hyprfast operating manual — Fast Rust alternative to hypruse: persistent Hyprland IPC + direct zbus AT-SPI + CDP browser automation + Hint+Decider resolution (hint_snapshot/choose/hint_click) + Task State (task_init/status/update) + Excalidraw lightning (excalidraw_draw/diagram/export), MCP. Use whenever you need desktop/window/workspace ops, AT-SPI clicks, Brave/Chromium automation via Chrome DevTools, WhatsApp Web, multi-step todo tracking, or Excalidraw whiteboard diagrams/architecture. Consult for hyprfast's desktop/hypr/launch/ui/click_ui/pointer/keyboard/screenshot/wait_for/binds + browser_navigate/snapshot/click/type/evaluate/screenshot/tabs + hint_snapshot/choose/hint_click + task_init/status/update/clear/next + excalidraw_draw/diagram/export/fit tools. Routed via skills/opencode/SKILL.md rows 5-7.
---

> **Router:** Read `skill/browser/SKILL.md` first for browser recipes (snapshot/ref rules, Gemini Copy-button DOM). This file owns *transport* (CDP/Hyprland), not recipes.

# hyprfast (0.9.0-dev)

Single Rust binary replacing `hypruse` + `@browsermcp/mcp`: direct Unix-socket Hyprland IPC (`$XDG_RUNTIME_DIR/hypr/<sig>/.socket.sock`), persistent `zbus` AT-SPI (no `busctl` per node), CDP browser automation, Hint+Decider resolution (`hint_snapshot` → `choose`/`decide` → `hint_click`/`hint_type`), Excalidraw whiteboard control + Decider-2B perception (vision 10 / text 255).

## Quick reference

Call tools through MCP using named `key=value` parameters.

| Want to... | MCP tool call |
|---|---|---|
| Snapshot (<5ms) | `desktop` |
| Window/workspace ops | `hypr action=workspace target=3` (also `focus_window`/`move_window`/`close_window`/`fullscreen`/`toggle_floating`) |
| Launch (blocks on `openwindow`, returns address) | `launch command="brave --new-window <url>"` |
| AT-SPI tree | `ui window=0x... name="Save"` |
| Click by name via `DoAction` | `click_ui name="OK" window=0x...` |
| Mouse | `pointer action=move x=600 y=400` |
| Keyboard (focuses window first) | `keyboard action=type text="hello" window=0x...` (or `action=key keys="ctrl+k"`) |
| Screenshot (fallback only, auto-tracked) | `screenshot window=0x...` |
| Clear tracked screenshots | `clear_screenshots all=false` (or `all=true` to sweep untracked leftovers) |
| Session status | `session_status` |
| Block on compositor event | `wait_for event=window_open match=WhatsApp timeout_s=5` |
| Keybinds | `binds` |
| Browser navigate | `browser_navigate url="https://example.com"` |
| Browser snapshot (AX refs) | `browser_snapshot` |
| Browser click | `browser_click element="Submit" ref="12"` |
| Browser type | `browser_type element="Search" ref="5" text="hello" submit=true` |
| Browser eval | `browser_evaluate js="document.title"` |
| Browser screenshot (CDP) | `browser_screenshot` |
| Browser tabs | `browser_tabs` |
| Launch Brave with CDP | `browser_open url="https://..."` |
| Task init | `task_init goal="play boomshakalaka" steps=["search","click","verify"]` |
| Task status | `task_status` |
| Task update | `task_update index=0 status="completed"` |
| Task next/add/clear | `task_next` / `task_add description="new step"` / `task_clear` |
| Excalidraw open | `excalidraw_open url="https://excalidraw.com/"` |
| Excalidraw draw | `excalidraw_draw type="rectangle" x=100 y=100 width=200 height=80 label="Backend"` |
| Excalidraw batch | `excalidraw_draw_batch elements=[{...},{...}]` |
| Excalidraw diagram | `excalidraw_diagram kind="microservices" params={title,services,databases}` |
| Excalidraw scene | `excalidraw_get_scene` / `excalidraw_clear` / `excalidraw_update_scene elements=[...] mode=append\|replace` |
| Excalidraw export | `excalidraw_export format="png" scale=1 background=true` |
| Excalidraw view/fit | `excalidraw_view json="{\"scrollX\":0}"` / `excalidraw_fit` |
| Perception decide | `decide context="page" questions=[{question,options}] image="data:..."` |
| Perception find/choose/classify/detect/identify | `find query="Save button"` / `choose question="pick" options=["a","b"]` / `classify question="state?" options=["login","dash"]` |
| Perception verify/wait | `verify query="Saved?"` / `verify_element selector="button"` / `wait_until query="success" timeout_ms=5000` |
| Hint resolve | `hint_resolve instruction="click Save" target="1"` |
| Key identify | `key_identify key="Enter" target="1"` |
| Visual target | `visual_target description="Save button" candidates=[{id,rect}] image="data:..."` |

**Rules:**
- `desktop` first, act on `address`. Never screenshot to locate windows.
- `ui` > `screenshot+zoom` for native apps. Brave/Chromium needs `--force-renderer-accessibility` or its AT-SPI tree is empty — for browsers, prefer the Hint+Decider pipeline (below) over `ui`.
- `wait_for` > `sleep`.
- `click_ui` uses `DoAction(0)`, falls back to pointer.
- Prefer `address:` selector over `class:` regex; re-verify `desktop` after any focus change (`address` goes stale on window close).

---

## Lucy Agent — Primary Workflow (Hint + Decider Pipeline)

> **Lucy** is an autonomous computer-use agent. `hyprfast` is Lucy's hands and eyes.
> This section teaches Lucy **how to think** when executing any browser task.

### Priority Order (MUST follow)

```
1. Hint + Decider  (PRIMARY — deterministic, fast)
2. browser_evaluate (LAST RESORT — when hints fail or for contenteditable editors)
3. browser_screenshot (VERIFY ONLY — never for locating elements)
```

### The Hint + Decider Pipeline (Step by Step)

Every browser interaction Lucy performs follows this 4-step loop:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: hint_snapshot  →  Get all clickable elements as keys   │
│ STEP 2: choose / decide  →  Ask Decider "which key for task?"  │
│ STEP 3: hint_click / hint_type  →  Press the chosen key        │
│ STEP 4: verify / wait_until  →  Confirm the action worked      │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 1 — `hint_snapshot`: See What's Clickable

Call `hint_snapshot` (optionally with a target for a specific tab). This scans the DOM and returns every clickable element with a **letter key** (`A`, `S`, `D`, `F`, ... `AA`, `AS`, ...):

```json
{
  "hints": [
    {"label": "A", "tag": "a",      "role": "link",    "name": "Home",           "text": "Home",     "rect": {"x":50,"y":12,...}},
    {"label": "S", "tag": "input",  "role": "textbox", "name": "Search",         "text": "",         "rect": {"x":300,"y":50,...}},
    {"label": "D", "tag": "button", "role": "button",  "name": "Search Button",  "text": "Search",   "rect": {"x":620,"y":50,...}},
    {"label": "F", "tag": "a",      "role": "link",    "name": "Trending",       "text": "Trending", "rect": {"x":50,"y":90,...}},
    ...
  ],
  "count": 42,
  "via": "hint"
}
```

Each hint has:
- **`label`**: The key Lucy presses to interact with this element (e.g. `"A"`, `"S"`, `"AD"`)
- **`name`** / **`text`**: Human-readable description of what the element is
- **`tag`** / **`role`**: Element type (`input`, `button`, `a`, `textbox`, `link`, etc.)
- **`rect`**: Bounding box on screen

#### Step 2 — `choose` / `decide`: Ask Decider Which Key to Press

Build a question from the hints. Give the Decider the **task description** as the question and the **hint labels + descriptions** as options:

```json
{
  "name": "choose",
  "arguments": {
    "question": "I need to type a search query. Which element is the search input?",
    "options": ["A: Home (link)", "S: Search (textbox)", "D: Search Button (button)", "F: Trending (link)"]
  }
}
```

Decider returns:
```json
{
  "selected": "S: Search (textbox)",
  "choice": "2",
  "confidence": 0.92
}
```

→ The answer is key **`S`**.

#### Step 3 — `hint_click` / `hint_type`: Execute the Action

- **To click**: call `hint_click` with the label.
- **To type**: call `hint_type` with the label and text.
- **To click + type in one shot**: call `hint_act` with an instruction.

#### Step 4 — `verify` / `wait_until`: Confirm It Worked

After acting, verify the result before moving to the next step:
```json
{"name": "verify", "arguments": {"query": "Search results for despacito are showing"}}
```
or
```json
{"name": "wait_until", "arguments": {"query": "Video player is visible", "timeout_ms": 5000}}
```

---

### Full Example: Play "Despacito" on YouTube

Here is the complete step-by-step Lucy would execute to play "Despacito" on YouTube:

```text
TASK: Play the song "Despacito" on YouTube

─── Phase 1: Open YouTube ───────────────────────────────────────

1. Call `desktop`
   → Get window list, find if Brave is already open

2. Call `launch` with `command="brave --new-window https://www.youtube.com"`
  → Opens YouTube in Brave and returns its window address
   → Waits for window to appear, returns window address

─── Phase 2: Type "despacito" in the search box ─────────────────

3. Call `hint_snapshot` for the YouTube tab
   → Returns all clickable elements on YouTube homepage:
     {"hints": [
       {"label":"A", "tag":"a",     "name":"Home",          "role":"link"},
       {"label":"S", "tag":"input", "name":"Search",        "role":"textbox"},
       {"label":"D", "tag":"button","name":"Search",        "role":"button"},
       {"label":"F", "tag":"a",     "name":"Explore",       "role":"link"},
       {"label":"G", "tag":"a",     "name":"Shorts",        "role":"link"},
       ...
     ], "count": 35}

4. MCP choose → "I need to type a search query, which is the search input?"
     options: ["A: Home (link)", "S: Search (textbox)", "D: Search (button)", "F: Explore (link)", "G: Shorts (link)"]
   → Decider returns: selected="S: Search (textbox)", key = S

5. Call `hint_type` with `target="S"` and `text="despacito"`
   → Types "despacito" into the search input (key S)

─── Phase 3: Press the search button ────────────────────────────

6. Call `hint_snapshot` again for the YouTube tab
   → Fresh snapshot (DOM may have changed after typing — autocomplete etc.)
     {"hints": [
       {"label":"A", "tag":"a",     "name":"Home",          "role":"link"},
       {"label":"S", "tag":"input", "name":"Search",        "role":"textbox",  "text":"despacito"},
       {"label":"D", "tag":"button","name":"Search",        "role":"button"},
       ...
     ], "count": 40}

7. MCP choose → "I typed despacito, now I need to submit the search. Which is the search button?"
     options: ["A: Home (link)", "S: Search (textbox)", "D: Search (button)", ...]
   → Decider returns: selected="D: Search (button)", key = D

8. Call `hint_click` with `target="D"`
   → Clicks the search button (key D)

9. Call `wait_until` with `query="search results loaded"` and `timeout_ms=5000`
   → Waits until YouTube search results page is rendered

─── Phase 4: Click the first Despacito video ────────────────────

10. Call `hint_snapshot` again for the YouTube tab
    → Fresh snapshot of search results page:
      {"hints": [
        {"label":"A", "tag":"a", "name":"Luis Fonsi - Despacito ft. Daddy Yankee", "role":"link"},
        {"label":"S", "tag":"a", "name":"Despacito Remix",                         "role":"link"},
        {"label":"D", "tag":"a", "name":"Despacito Live Performance",              "role":"link"},
        ...
      ], "count": 28}

11. MCP choose → "Which video is the original Despacito by Luis Fonsi?"
      options: ["A: Luis Fonsi - Despacito ft. Daddy Yankee (link)",
                "S: Despacito Remix (link)",
                "D: Despacito Live Performance (link)"]
    → Decider returns: selected="A: Luis Fonsi - Despacito ft. Daddy Yankee", key = A

12. Call `hint_click` with `target="A"`
    → Clicks the first result — video page opens

13. Call `wait_until` with `query="video player is playing"` and `timeout_ms=8000`
    → Confirms the video has started playing

DONE ✓ — Despacito is now playing on YouTube.
```

### Quick-Reference: Which Hint Tool When?

| Situation | Tool | Example |
|---|---|---|
| **See all clickable elements** | `hint_snapshot` |
| **Click element by letter key** | `hint_click` |
| **Type text into element by key** | `hint_type` |
| **Auto-resolve + click in one call** | `hint_act` |
| **Batch multiple actions** | `hint_batch` |
| **Remove hint overlays** | `hint_clear` |
| **Ask Decider which key to press** | `choose` | `choose question="which is search?" options=["A: Home","S: Search"]` |
| **Verify action succeeded** | `verify` |
| **Wait for DOM state** | `wait_until` |

### When Hints Fail → Fallback Chain

If `hint_snapshot` returns **0 hints** (canvas, WebGL, PDF viewer, native app,
or a page that has not finished navigating):
1. Re-observe once with `browser_snapshot` — the DOM may not have hydrated yet.
2. Try `browser_evaluate js="document.querySelector('#search').click()"` (raw JS, only if selector known).
3. If the page is `about:blank` or the wrong site, navigate first with
   `browser_navigate url="https://..."` — never act on a blank page.

---

## Browser Automation (CDP)

No Node/`@browsermcp/mcp`/Playwright needed — `hyprfast` is the sole `browser_*` provider (uses `GET /json` discovery + WS JSON-RPC).

**Priority: Hint+Decider → Eval fallback → Snapshot → Screenshot (verify only).**
The Hint+Decider pipeline (above) is the primary approach for all browser tasks. Fall back to `browser_evaluate` only on an explicit `No action found` / `not found` / `backend resolve` error, or when the target is a `contenteditable` editor not exposed in the AX tree. Use `browser_snapshot`/`browser_click` only if hints fail. `browser_screenshot` is verification-only, never for locating elements.

Flow: `desktop` → `hypr focus_window` or `browser_open url=... workspace=N` (auto-adds CDP flags) → `hint_snapshot` + `choose` + `hint_click` → `browser_evaluate` (last resort) → `browser_screenshot` (verify).

**Env:** `HYPRFAST_CDP_HOST`/`PORT` default `127.0.0.1:9222`. `CDP unreachable` means no browser is listening — use `browser_open`, which is the single launcher. Never launch `brave`/`chromium` directly: a bare launch reuses the default profile, never opens the debugging port, and a second browser is exactly what lucy must not end up driving.

**Multi-tab targeting:** `browser_evaluate`/`browser_snapshot` can hit the wrong tab when >1 page is open. Before acting, call `browser_tabs`; if the target tab already exists, activate it (`/json/activate/<id>`) instead of opening a new one, and close stray duplicate tabs pointed at the same URL.

## No-vision fallback (no AX tree)

When `hint_snapshot` returns 0 hints (canvas, WebGL, custom-drawn UI, or a page
that has not navigated yet), stay on DOM/CDP — there is no vision grounding
fallback. `ground` / `act_fast` / `act_batch` and any `stagehand_*` tool were
removed and must never be called; a plan naming one will not resolve.

**Canonical flow:**
```text
1. Call `browser_open` with `url=<url>` and `workspace=N`.
2. Call `hint_snapshot` to list clickable elements.
3. Call `choose` with the hint labels as options to pick the target.
4. Call `hint_click` / `hint_type` with the chosen label.
5. On 0 hints, re-observe with `browser_snapshot`, or use `browser_evaluate`
   with a selector you know. If the page is `about:blank`, navigate first.
6. Verify with `verify` / `wait_until` or `browser_screenshot`.
```

**Troubleshooting:**
| Symptom | Fix |
|---|---|
| `CDP unreachable` | `browser_open url=…` — it is the only launcher, and it adds the CDP flags |
| `brave-browser exposes no accessibility tree` | Add `--force-renderer-accessibility` (already in `browser_open`'s own flags) |
| `LLM did not return actionable element` | Retry once; if persistent, fall back to `browser_evaluate` |
| `Could not find object with given id` | AX node is stale — fall back to `browser_evaluate` |

## Brave/Chromium gotchas

- Single-instance: `brave --new-window` reuses the existing process — if that process launched without `--force-renderer-accessibility`, new windows still have no AT-SPI tree. Kill and relaunch with the flag (avoid `pkill -f`, see Known gotchas).
- Window class is `brave-browser`, not `chromium`.
- For WhatsApp Web without the flag, don't use `ui`/`click_ui` — use keyboard shortcuts or `browser_evaluate` instead.

## WhatsApp Web — shortcuts

Via `keyboard action=key keys="..." window=0x...` (focuses window, then sends chord). **Use the Windows/Linux column on Omarchy/Hyprland** (model translates `Cmd`→`ctrl`).

| Action | Keys | Action | Keys |
|---|---|---|---|
| New chat | `ctrl+alt+n` | Search | `ctrl+alt+slash` (or `ctrl+k`) |
| New group | `ctrl+alt+shift+n` | Search in chat | `ctrl+alt+shift+f` |
| Archive chat | `ctrl+alt+e` | Next/prev chat | `ctrl+alt+tab` / `ctrl+alt+shift+tab` |
| Mute chat | `ctrl+alt+shift+m` | Close chat | `esc` |
| Pin chat | `ctrl+alt+shift+p` | Emoji panel | `ctrl+alt+e` |
| Mark unread | `ctrl+alt+shift+u` | GIF panel | `ctrl+alt+g` |
| Delete chat | `ctrl+alt+backspace` | Sticker panel | `ctrl+alt+s` |
| Profile | `ctrl+alt+p` | Settings | `ctrl+alt+comma` |

**DOM selectors (for `browser_evaluate` fallback):**
- Chat list item: `[data-testid="cell-frame-container"]`
- Message input: `[data-testid="conversation-compose-box-input"]`
- Send button: `[data-testid="send"]`
- Search input: `[data-testid="chat-list-search"]`

**Flow A — evaluate (fastest, ~0.5s):**
```text
{"tool":"desktop"} → {"tool":"hypr","action":"focus_window","target":"0x..."}
→ {"tool":"browser_evaluate","js":"document.querySelector('[data-testid=\"cell-frame-container\"]').click()"}
→ {"tool":"browser_evaluate","js":"const ed=document.querySelector('[data-testid=\"conversation-compose-box-input\"]');ed.focus();document.execCommand('insertText',false,'hello');document.querySelector('[data-testid=\"send\"]').click()"}
```

**Flow B — keyboard fallback (if evaluate fails):**
```text
{"tool":"desktop"} → {"tool":"hypr","action":"focus_window","target":"0x..."}
→ {"tool":"keyboard","action":"key","keys":"ctrl+alt+slash","window":"0x..."}
→ {"tool":"keyboard","action":"type","text":"Khushi","window":"0x..."} → {"tool":"keyboard","action":"key","keys":"enter"}
→ wait ~0.8s → {"tool":"keyboard","action":"type","text":"hello"} → {"tool":"keyboard","action":"key","keys":"enter"}
```
Only screenshot for visual confirm (`scale=0.5`, JPEG ~80KB). Keyboard (~50ms/key) beats a vision loop (2-4s) for WhatsApp Web.

**Window placement:** if the Brave/WhatsApp window is tiled (not fullscreen), the omnibox overlay traps keyboard focus. Before any WhatsApp keyboard flow: check `desktop` for `class==brave-browser` + title contains `WhatsApp`; if not fullscreen, move it to an empty workspace (`desktop.workspaces` where `windows==0`, else use `10`) and fullscreen it there before continuing.

## Screenshot session

Every `screenshot` call auto-appends to `$XDG_RUNTIME_DIR/hyprfast-session.json` (fallback `/tmp`).
- After a successful task: `clear_screenshots all=false` — deletes only tracked files, truncates session to `[]`.
- Full cleanup of stale leftovers: `clear_screenshots all=true`.
- Inspect: `session_status`.
- Pattern: screenshot as needed during a task → clear on success. On failure, keep shots for debugging, then call `clear_screenshots all=true`.

## Task State — persistent todo for multi-step actions

File: `$XDG_RUNTIME_DIR/hyprfast-tasks.json` (fallback `/tmp`) — single active list `{goal, steps:[{id,description,status}], progress}`. Status: `pending|in_progress|completed|failed|skipped`. Auto-clears when every step is `completed|skipped`.

| Tool | Params | When |
|---|---|---|
| `task_init` | `goal`, `steps[]` | Start: break the request into ordered steps before acting |
| `task_status` | — | Resume: check progress + next pending step after any failure/timeout, before retrying |
| `task_update` | `index` or `id`, `status` | After each step completes/fails; last completion auto-clears |
| `task_next` | — | Get next pending step |
| `task_add` | `description` | Append a step mid-flow |
| `task_clear` | — | Cancel/reset |

**Rules:**
- `task_init` before the first browser step for any request with ≥2 steps.
- `task_update` immediately after each step — don't batch.
- On retry after a timeout/failure, call `task_status` first and resume from the next pending step — don't restart from step 1, and don't re-open a tab that already exists (check `browser_tabs` and reuse/activate instead of duplicating).

## Excalidraw — whiteboard diagrams

For `https://excalidraw.com`: draws by injecting scene state directly into `excalidrawAPI` via `Runtime.evaluate` `updateScene` (~120ms for a full diagram) instead of pointer-dragging shapes (~2-4s each). Use for any architecture/flow/sequence/network/ER diagram.

| Tool | Params | When |
|---|---|---|
| `excalidraw_open` | `url?` (default `https://excalidraw.com/`) | Ensure tab before drawing |
| `excalidraw_get_scene` | — | Audit `elements.length` + `appState` |
| `excalidraw_clear` | — | Reset to `[]` |
| `excalidraw_draw` | `type,x,y,width,height,x2,y2,text,label,strokeColor,backgroundColor,fillStyle,strokeWidth,points,name` | Single primitive |
| `excalidraw_draw_batch` | `elements:[{type,...}]` | Batch N primitives in one call |
| `excalidraw_update_scene` | `elements:[...]`, `mode: append\|replace` | Raw scene / custom layout / replaying saved `.excalidraw` JSON |
| `excalidraw_diagram` | `kind: flowchart\|sequence\|microservices\|architecture\|aws\|3tier\|network\|er\|custom`, `params:{title,services,databases,participants,messages,nodes,entities,steps}` | Auto-layout template |
| `excalidraw_export` | `format: png\|svg\|clipboard, background, dark, embedScene, scale` | Export |
| `excalidraw_save` | `path?` | Trigger `.excalidraw` download |
| `excalidraw_view` / `excalidraw_fit` | `json{scrollX,scrollY,zoom}` / — | Viewport control / auto-center on bbox |

**Canonical flow:**
```text
1. excalidraw_open
2. excalidraw_clear  // skip if appending to existing scene
3. excalidraw_diagram kind="microservices" params={title:"...", services:[...], databases:[...]}
   // or excalidraw_draw_batch for a fully custom layout
4. excalidraw_fit
5. excalidraw_get_scene (verify element count) + browser_screenshot or excalidraw_export
```

## Decider-2B Perception (vision 10 / text 255)

Native local Decider server (`http://127.0.0.1:8001`, `POST /predict`) as semantic perception/decision backend.
- **Vision budget:** Max 10 candidates per image question (annotated bounding boxes with numeric IDs `1..10`).
- **Text budget:** Up to 255 options per question (hierarchical chunking via deterministic filtering + group-winners -> final set).
- **Deterministic fallback:** If Decider is offline (`DECIDER_ENABLED=0`) or confidence is low (<0.55 or margin <0.10), hyprfast falls back to deterministic DOM/AX/heuristics without failing.
- **Resolution order:** Exact -> DOM/AX -> Hint -> Heuristic -> Decider (10 vision / 255 text) -> Visual Fallback -> Uncertain.

**Environment variables:**
```bash
DECIDER_ENABLED=1             # Enable local Decider daemon routing (default: disabled, uses deterministic fallback)
DECIDER_URL=http://127.0.0.1:8001  # Daemon base URL
DECIDER_TIMEOUT_MS=5000       # HTTP request timeout
DECIDER_MAX_IMAGE_DIM=1280    # Max image dimension before downscaling
DECIDER_TEMPERATURE=1.0       # Sampling temperature
```

### Perception MCP Tool Reference

#### 1. General Decisions & Classification
* **`decide`** — **Unified all-in-one Decider-2B command**: covers single/multiple questions, options (`choose`), state classification (`classify`), boolean existence checks (`detect`), and multi-question evaluation (`batch`) over context and images (file path, base64, data URI, or `--screenshot`).

  * **CLI Usage:**
    ```bash
    # Single question with options (choose parity)
    hyprfast decide "Which button proceeds to payment?" --options "Continue Shopping, Proceed to Checkout, Cancel"

    # Presence/boolean existence check (detect parity, auto-defaults options to yes/no/uncertain)
    hyprfast decide "Is there an error banner visible?"

    # State classification (classify parity)
    hyprfast decide --options "login_screen, 2fa_challenge, dashboard"

    # With explicit image (file path, base64, or data URI)
    hyprfast decide "Which button proceeds to payment?" --options "Continue Shopping, Proceed to Checkout, Cancel" --image /tmp/shot.png
    hyprfast decide "Is dark mode enabled?" --image "data:image/png;base64,iVBORw0KGgo..."

    # With auto-captured screenshot / vision
    hyprfast decide "Is dark mode enabled?" --screenshot

    # Batch multiple questions in one request
    hyprfast decide --questions '[{"question": "Is dark mode active?", "options": ["yes", "no"]}, {"question": "Which tab is focused?", "options": ["Profile", "Security"]}]'
    ```

  * **MCP Call (Single Question / Options with `image`):**
    ```json
    {
      "name": "decide",
      "arguments": {
        "question": "Which button proceeds to payment?",
        "options": ["Continue Shopping", "Proceed to Checkout", "Cancel"],
        "context": "Cart checkout page",
        "image": "/tmp/shot.png"
      }
    }
    ```
    `image` accepts a file path (`/tmp/shot.png`), raw base64, or data URI (`data:image/png;base64,...`). Use `screenshot: true` instead for auto-capture.

  * **MCP Call (Presence / Detect Mode with `image`):**
    ```json
    {
      "name": "decide",
      "arguments": {
        "question": "Is there an error banner visible on the screen?",
        "image": "data:image/png;base64,iVBORw0KGgo...",
        "screenshot": false
      }
    }
    ```
    Either pass `image` explicitly or set `screenshot: true` for auto-capture — don't omit both for vision questions.

  * **MCP Call (Batch Multi-Question Mode with `image`):**
    ```json
    {
      "name": "decide",
      "arguments": {
        "context": "Settings modal",
        "questions": [
          {"question": "Is dark mode active?", "options": ["yes", "no"]},
          {"question": "Which tab is focused?", "options": ["Profile", "Security", "Billing"]}
        ],
        "image": "/tmp/settings-modal.png"
      }
    }
    ```

* **`batch` (`decider_batch`)** — Multiple questions evaluated against the same screenshot/context in one request.

  * **MCP Call:**
    ```json
    {
      "name": "decider_batch",
      "arguments": {
        "context": "Settings modal",
        "questions": [
          {"question": "Is dark mode active?", "options": ["yes", "no"]},
          {"question": "Which tab is focused?", "options": ["Profile", "Security", "Billing"]}
        ]
      }
    }
    ```

* **`choose`** — Select the best option from a list of up to 255 items (returns selected, runner-up, and confidence margin).

  * **MCP Call:**
    ```json
    {
      "name": "choose",
      "arguments": {
        "question": "Which action should be taken next?",
        "options": ["click_save", "discard_draft", "export_pdf"]
      }
    }
    ```

* **`classify`** — Classify the current UI view or state into predefined categories.

  * **MCP Call:**
    ```json
    {
      "name": "classify",
      "arguments": {
        "question": "What is the current page state?",
        "options": ["logged_out_landing", "login_screen", "2fa_challenge", "dashboard"]
      }
    }
    ```

* **`detect`** — Check presence/absence of an element, notification, or state (`yes` | `no` | `uncertain`).

  * **MCP Call:**
    ```json
    {
      "name": "detect",
      "arguments": {
        "query": "Is there an error banner visible on the screen?",
        "context": "Form submitted"
      }
    }
    ```

---

#### 2. Semantic Target Resolution & Actions
* **`find`** — Semantic target resolver: query -> DOM/AX/hints -> candidate filtering -> Decider -> resolved candidate metadata and bounding box.

  * **MCP Call:**
    ```json
    {
      "name": "find",
      "arguments": { "query": "Blue Submit button", "target": "1", "use_vision": false }
    }
    ```

* **`identify`** — Identify which candidate matches a description among a provided list.

  * **MCP Call:**
    ```json
    {
      "name": "identify",
      "arguments": {
        "query": "Primary download button",
        "candidates": [{"id": 1, "label": "A", "name": "Download Installer"}, {"id": 2, "label": "B", "name": "Documentation"}]
      }
    }
    ```

* **`visual-target` (`visual_target`)** — Target picker with visual budget (max 10 candidates with bounding boxes).

  * **MCP Call:**
    ```json
    {
      "name": "visual_target",
      "arguments": {
        "description": "Green confirmation pill",
        "candidates": [{"id": 1, "rect": {"x": 120, "y": 340, "width": 90, "height": 30}, "label": "A"}],
        "image": "data:image/png;base64,..."
      }
    }
    ```

* **`find-and-click` (`find_and_click`)** — Composite: semantic `find` then immediately click element via hint/browser click.

  * **MCP Call:**
    ```json
    {
      "name": "find_and_click",
      "arguments": { "query": "Accept all cookies", "target": "1" }
    }
    ```

* **`find-and-type` (`find_and_type`)** — Composite: semantic `find` then immediately type text into the input field.

  * **MCP Call:**
    ```json
    {
      "name": "find_and_type",
      "arguments": { "query": "Search documentation", "text": "async runtime", "target": "1" }
    }
    ```

---

#### 3. Hint Resolution & Virtual Keyboards
* **`hint-resolve` (`hint_resolve`)** — Unified `hint_snapshot` -> Decider candidate pick -> `hint_click` in one invocation.

  * **MCP Call:**
    ```json
    {
      "name": "hint_resolve",
      "arguments": { "instruction": "click the Login link", "target": "1" }
    }
    ```

* **`hint-resolve-batch` (`hint_resolve_batch`)** — Multiple hint resolutions against a single snapshot (max 12 steps).

  * **MCP Call:**
    ```json
    {
      "name": "hint_resolve_batch",
      "arguments": {
        "instructions": ["click Terms checkbox", "click Continue button"],
        "target": "1"
      }
    }
    ```

* **`key-identify` (`key_identify`)** — Identify target key on visual / on-screen keyboards.

  * **MCP Call:**
    ```json
    {
      "name": "key_identify",
      "arguments": { "key": "Enter", "target": "1" }
    }
    ```

---

#### 4. Verification & State Polling
* **`verify`** — DOM-first verification with visual fallback -> returns `success`, `failure`, or `uncertain`.

  * **MCP Call:**
    ```json
    { "name": "verify", "arguments": { "query": "Profile updated successfully" } }
    ```

* **`verify-element` (`verify_element`)** — Check whether a specific element is present and visible.

  * **MCP Call:**
    ```json
    { "name": "verify_element", "arguments": { "selector": "button[aria-label='Save']" } }
    ```

* **`verify-action` (`verify_action`)** — Verify that an action produced the expected text or result.

  * **MCP Call:**
    ```json
    {
      "name": "verify_action",
      "arguments": { "query": "clicked unsubscribe", "expected": "You have been unsubscribed" }
    }
    ```

* **`wait-until` (`wait_until`)** — Poll DOM/AX tree until predicate holds or timeout expires.

  * **MCP Call:**
    ```json
    {
      "name": "wait_until",
      "arguments": { "query": "Deployment complete", "timeout_ms": 10000, "interval_ms": 500 }
    }
    ```

* **`observe-state` (`observe_state`)** — Classify current state against explicit valid state options.

  * **MCP Call:**
    ```json
    {
      "name": "observe_state",
      "arguments": {
        "query": "What step is the wizard on?",
        "options": ["step_1_account", "step_2_plan", "step_3_payment", "step_4_finish"]
      }
    }
    ```

---

#### 5. Diagnostics & Metrics
* **`decider_health` (MCP)** / `GET /health` — Check if local Decider daemon is alive.
  * **MCP Call:** `{"name": "decider_health", "arguments": {}}`
* **`decider_metrics` (MCP)** — Retrieve runtime metrics (total requests, successes, errors, timeouts, avg latency).
  * **MCP Call:** `{"name": "decider_metrics", "arguments": {}}`
---

### Canonical Flows

**1. Semantic Form Automation:**
```text
1. Call browser_open with url="https://example.com/login".
2. Call find_and_type with query="Email address", text="test@example.com", and the target.
3. Call find_and_type with query="Password", text="Secret123!", and the target.
4. Call find_and_click with query="Log in button" and the target.
5. Call wait_until with query="Welcome back" and timeout_ms=5000.
```

**2. State Machine Routing with Fallback:**
```text
1. Call classify with question="Current view?" and options=["logged_out","dashboard","modal_open"].
2. If modal_open, call hint_resolve with instruction="Close modal".
3. Call verify with query="Modal closed".
```

## Safety

- Confirm recipient/content before sending (`Enter`) — same caution as `close_window`.
- Re-verify `desktop` after any focus change; a stale `address` will target the wrong window.
- Decider never executes arbitrary commands — outputs `candidate 93` → Hyprfast maps to existing `hint_click`/`browser_click` only.

## Known gotchas

- **Don't duplicate browser tabs on retry.** Before `browser_open`/`browser_navigate`, check `browser_tabs` for an existing tab at the same URL and activate it instead of opening a new one — duplicates cause `browser_evaluate`/`browser_snapshot` to target the wrong tab.
- **Don't restart a multi-step task from scratch after a timeout.** Call `task_status` (and `browser_tabs` for browser tasks) to find what's already done, then resume from the next pending step.
- **CDP eval before AT-SPI flag fix:** if `brave-browser exposes no accessibility tree` appears, don't loop on `ui`/`click_ui` — add `--force-renderer-accessibility` or switch to the Hint+Decider pipeline (`hint_snapshot` + `choose` + `hint_click`).
