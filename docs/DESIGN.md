---
version: alpha
name: presence-demo-console
description: "Presence-demo's console is a PureSound playground screen: same shell, same components, same tokens. This file records how presence maps its content onto those components; it does not invent a second design system."
source:
  repository: https://github.com/mcw519/puresound
  path: puresound/web/static
  revision: 121bf623b491cb7f18805bec65099543ecc5323f
  vendored: presence_demo/static/vendor/puresound
  license: Apache-2.0 (CSS), SIL OFL 1.1 (IBM Plex)
---

# presence-demo console — DESIGN.md

Format: [DESIGN.md](https://github.com/google-labs-code/design.md).

## Principle

**Use PureSound's design, not just its fonts.** Since v0.10.3 the HTML prefers PureSound component
classes from the vendored stylesheets. `app.css` only adds presence-specific layout (camera view,
state-machine strip, countdown bars, scopes, people list, settings module accordion) and maps a few
presence roles onto upstream tokens. It must not redefine `.button`, `.panel`, `.field`, `.control`,
`.segmented`, `.chip`, `.toast`, `.inspector`, or any `--page` / `--ink` / `--accent` token.

## Loaded stylesheets

| Href | Upstream | Role |
| --- | --- | --- |
| `/static/vendor/puresound/tokens.css` | `css/tokens.css` | colours, type, density |
| `/static/vendor/puresound/base.css` | `css/base.css` | IBM Plex, reset, focus, reduced motion |
| `/static/vendor/puresound/components.css` | `css/components.css` | button, field, control, segmented, chip, state, toggle, range, table, stat, panel, empty-state, toast, dialog |
| `/static/vendor/puresound/shell.css` | `css/shell.css` | app-shell, sidebar, page-header, workbench, inspector |
| `/static/vendor/puresound/screens-shared.css` | subset of `css/screens.css` | panel-title h2, metric-row / metric |
| `/static/app.css` | presence only | camera, state strip, scopes, people, module accordion |

Checksums of the unmodified upstream files are pinned in `tests/test_design_tokens.py`.

## Component mapping (presence → PureSound)

| Presence UI | PureSound class | Notes |
| --- | --- | --- |
| Shell | `.app-shell` `.sidebar` `.main-content` `.screen` | Same as playground |
| Nav items | `.nav-link` `.nav-icon` `.nav-group-title` | Buttons that look like playground links |
| Connection | `.status-dot.is-ok` / `.is-err` `.runtime` | Replaces custom `.dot` |
| Theme | `.nav-switch` | Auto / Light / Dark |
| Page header | `.page-header` `.page-heading` `.page-title` `.page-desc` `.page-actions` | Sticky, blur |
| Primary action (開始) | `.button.button-primary` | Ink fill, not accent |
| Secondary (設定) | `.button.button-secondary` | |
| Stop | `.button.button-danger` | Outlined danger |
| Ghost / icon close | `.button.button-ghost.button-icon.button-small` | |
| Camera / status / audio cards | `.panel.panel-pad` + `.panel-title` > `h2` + `.panel-help` | Not custom `.card` + uppercase `h3` |
| Speaking / silence / ROI / session | `.metric-row` > `.metric` > `span` + `strong` | Playground result metrics |
| State machine | `.fsm-axis` > `.fsm-node` + `.fsm-edge`; `.fsm-hist` > `.fsm-tr` (app.css) | Nodes on an axis; current node filled with its state colour, last transition's source dashed and path in accent |
| Audio signals | `.sig-group.raw` / `.sig-group.clean` (waveform + spectrogram) | One bundle per signal; mint / periwinkle edge |
| ROI editor | `.segmented#roiPick` (臉部 / 身體 / 鎖定) + drag on `#view canvas` | Face cyan dashed, body purple dotted; only the selected ROI is draggable |
| Live caption | `#view > .caption` (`.is-in` fade, `.is-final` white; `--nav` 78 % backing, `--on-nav` text) | Single bottom line over the camera, never mirrored |
| Monitor stage | `section.panel.monitor-stage` = `#view` (+ `.caption`) + `.sig-pair` (raw / output bundles) on `--nav` | One dark frame, tracks attached under the video, width from `100vh` so it fits a laptop screen |
| Layout | Left column: monitor stage, then ROI card, then monitor/record/timeline card; right column: state, ASR, P(group), countdowns | |
| Monitor off/raw/clean | `.segmented` `[role=radiogroup]` `.is-active` | |
| Live camera chip | `.chip.is-live` | Accent border + pulse |
| Recording indicator | `#recInd` (pill on `--bad-soft`) | |
| Resource tiles | `.stat` > `span` + `strong` | |
| Latency table | `.table` + `.num` | |
| Empty tracks | `.empty-state` | Dashed panel |
| Toast | `.toast.is-visible` | Always-dark nav chrome |
| Error banner | `#errBanner` (toast chrome + `--nav-bad` rule) | |
| Settings drawer | `.inspector` `.inspector-head` `.inspector-body` `.inspector-section` `.eyebrow` | Opens with `.is-open` + `.inspector-backdrop` |
| Presets | `.segmented` | |
| Advanced toggle | `.toggle-row` `.toggle-ui` | |
| Bool settings | `.toggle-row` | Built by `rowHTML` |
| Selects / text / numbers | `.control` | |
| Sliders | `.range` + `.field-label` + `output` | Accent thumb |
| Hints | `.toggle-hint` / `.tip` | |
| Module strip | `.chip` in `.flow-row` | Accent when active |
| Module blocks | `.mod` `<details>` | Presence-only accordion on inspector density |
| Scopes (wave / spectrum / timeline) | always-dark `--nav` ground | Same as playground audio players |
| Raw / clean series | `--mint` / `--periwinkle` | Same signal colours as playground |

## Role aliases in `:root` (app.css)

`--ok` → `--good`, `--idle` → `--mint-ink`, `--end` → `--ink-2`, `--viz-raw` → `--mint`,
`--viz-clean` → `--periwinkle`, `--violet` → `--magenta`. Camera overlays use the dark-theme hex
values of those tokens because the camera ground is always dark.

## Density and layout

- 4 px grid, `--panel-pad` 16, `--gap` 16, control height 32 / 26, radius 4 / 6 / pill — all from tokens.
- Workbench: main column (camera panel) + right stack at `--inspector-w` (400 px ≥ 1400 px).
- Inspector drawer 520 px; sections separated by hairlines; eyebrow labels for Presets / Data flow / Import.
- One primary button per page (開始). Accent is only for focus, selection, live, and P(group).

## Accessibility

WCAG 2.1 AA on dashboard + settings, light and dark (headless Chrome). `--ink-3` / `--on-nav-3` are
forced to `--ink-2` / `--on-nav-2` for text in `app.css` because upstream tertiary ink is below 4.5:1
in places. Focus ring and reduced-motion come from `base.css`.

## Do's and Don'ts

- Do add UI with PureSound classes first; only invent a class when no upstream component fits.
- Do keep `app.css` free of upstream token and component redefinitions (enforced by tests).
- Don't hard-code hex in components except camera overlays (documented above).
- Don't show real camera frames in docs screenshots.
