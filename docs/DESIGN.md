---
version: alpha
name: presence-demo-console
description: "Dark-first, dense-but-calm console for a real-time perception demo. Neutral graphite surfaces with hairline borders, one blue accent for interaction, and a small accessible status palette reserved for state (ACTIVE / IDLE / MAYBE_AWAY / AWAY / ENDED) and audio decisions. Numbers are monospace with tabular figures. A light theme mirrors every token."
colors:
  canvas: "#0b0d10"
  surface-1: "#12151a"
  surface-2: "#181c22"
  surface-3: "#20252d"
  hairline: "#272d36"
  hairline-strong: "#3a424e"
  ink: "#e8ebef"
  ink-muted: "#a7b0bd"
  ink-subtle: "#8b95a3"
  primary: "#7b93ff"
  on-primary: "#0b0d10"
  focus: "#9fb0ff"
  success: "#3ccf6e"
  idle: "#8aa4c8"
  warning: "#e0a43a"
  danger: "#ff6b61"
  ended: "#7d8794"
  violet: "#b48cff"
  viz-raw: "#9aa4b2"
  viz-clean: "#5aa9ff"
  recording: "#e5484d"
  light-canvas: "#f5f6f8"
  light-surface-1: "#ffffff"
  light-surface-2: "#f1f3f6"
  light-hairline: "#dde1e7"
  light-ink: "#14181d"
  light-ink-muted: "#46505d"
  light-primary: "#3552d4"
  light-success: "#1a7f37"
  light-warning: "#9a6700"
  light-danger: "#cf222e"
typography:
  title:
    fontFamily: system-ui
    fontSize: 16px
    fontWeight: 650
    lineHeight: 1.3
  section-label:
    fontFamily: system-ui
    fontSize: 12px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0.06em
  body:
    fontFamily: system-ui
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
  body-sm:
    fontFamily: system-ui
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.45
  numeric:
    fontFamily: ui-monospace
    fontSize: 13px
    fontWeight: 500
    lineHeight: 1.4
  stat:
    fontFamily: ui-monospace
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.2
rounded:
  sm: 6px
  md: 8px
  lg: 10px
  pill: 999px
spacing:
  1: 4px
  2: 8px
  3: 12px
  4: 16px
  5: 24px
components:
  card:
    backgroundColor: "{colors.surface-1}"
    rounded: "{rounded.lg}"
    padding: 14px
  button:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: 7px 12px
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.md}"
  segmented-control:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink-muted}"
    rounded: "{rounded.md}"
  badge:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: 1px 7px
  recording-indicator:
    backgroundColor: "{colors.recording}"
    textColor: "#ffffff"
    rounded: "{rounded.pill}"
    typography: "{typography.numeric}"
  module-chip:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: 4px 8px
---

# presence-demo console — DESIGN.md

Format: [DESIGN.md (Google Labs, alpha, Apache-2.0)](https://github.com/google-labs-code/design.md) — YAML tokens + prose.
The tokens are implemented as CSS custom properties at the top of `presence_demo/static/index.html`
(`--canvas`, `--surface-1`, … `--viz-clean`); charts on `<canvas>` read the same variables at draw time, so both themes stay consistent.

## Overview

A monitoring console used while people walk up to and away from a device. It must be readable at a glance
(state, who is a participant, is audio being kept or suppressed) and still show dense diagnostics
(latency tables, RTF, timelines). Principles:

- **Dark first, light available** (`☀ 淺色 / ☾ 深色` in the header, saved in the browser). Dark reduces glare next to a camera view.
- **Calm surfaces, loud only for state.** Neutral graphite layers separated by 1 px hairlines; color is reserved for meaning.
- **One accent** (blue) for interaction: primary button, selected segment, focus ring, links.
- **Numbers are monospace with tabular figures** so values do not jitter as they update.
- **Data-flow order everywhere.** Settings, docs and the module strip follow input → detection → tracking → presence → state → audio → ASR.

Inspiration (ideas only — no assets, fonts or text copied): the DESIGN.md analyses of
[Linear](https://github.com/VoltAgent/awesome-design-md/tree/main/design-md/linear.app) (near-black layered surfaces, hairline borders, a single chromatic accent used only for interaction/focus),
[Vercel](https://github.com/VoltAgent/awesome-design-md/tree/main/design-md/vercel) (monospace captions for technical values, semantic success/warning/error scale) and
[Sentry](https://github.com/VoltAgent/awesome-design-md/tree/main/design-md/sentry) (monitoring dashboards: dense tables, status color discipline),
from [VoltAgent/awesome-design-md](https://github.com/VoltAgent/awesome-design-md) (MIT). Those files describe third-party brands;
their brand fonts and logos are not used. All values here are our own.

## Colors

| Role | Dark | Light | Use |
|---|---|---|---|
| canvas / surface-1 / surface-2 / surface-3 | `#0b0d10` / `#12151a` / `#181c22` / `#20252d` | `#f5f6f8` / `#fff` / `#f1f3f6` / `#e7eaef` | page / cards / controls, stats / gate rows, badges |
| hairline / hairline-strong | `#272d36` / `#3a424e` | `#dde1e7` / `#c3c9d2` | 1 px borders, grid lines |
| ink / ink-muted / ink-subtle | `#e8ebef` / `#a7b0bd` / `#8b95a3` | `#14181d` / `#46505d` / `#5b6573` | text levels (all ≥ 4.5:1 on surface-1) |
| primary | `#7b93ff` | `#3552d4` | interaction, selected, focus |
| success (ACTIVE, agree, speech) | `#3ccf6e` | `#1a7f37` | |
| idle (IDLE) | `#8aa4c8` | `#4f6a8f` | |
| warning (MAYBE_AWAY, 聲學誤判-漏收) | `#e0a43a` | `#9a6700` | |
| danger (AWAY, 漏音, errors) | `#ff6b61` | `#cf222e` | |
| ended / silence | `#7d8794` | `#6e7781` | |
| violet (聲學誤判-超出範圍, pose ROI) | `#b48cff` | `#8250df` | |
| viz-raw / viz-clean | `#9aa4b2` / `#5aa9ff` | `#6e7781` / `#2f6fdb` | raw vs clean audio series |
| recording | `#e5484d` | `#cf222e` | recording indicator only |

Status is never color-only: every state chip, decision and counter also has a text label.
Overlays drawn on the live video keep fixed high-contrast hues (independent of theme) because the background is the camera image.

## Typography

System font stacks only (no web-font download): sans `-apple-system, "Segoe UI", Roboto, "PingFang TC", "Noto Sans TC", "Microsoft JhengHei"`;
mono `ui-monospace, SFMono-Regular, Menlo, Consolas, "JetBrains Mono"` (JetBrains Mono, OFL, is used only if already installed).
Scale: title 16/650, section label 12/600 uppercase +0.06em, body 14, small 12, numeric mono 13, stat mono 18/600.
`font-variant-numeric: tabular-nums` on stats, tables and key/value values.

## Layout

- Header (sticky, translucent canvas): connection dot, title, recording indicator, camera/mic pill, start/stop, theme, settings.
- Two columns ≥ 900 px: left = video with overlays (ROI, tracks, distance zone) + hint; right = state machine, stats, audio panel
  (monitor/record controls, waveforms, spectrograms, 30 s timeline, key/value), ASR, timers, resources, participants, log. One column below 900 px.
- Settings: right drawer 520 px (full width on phones): presets → data-flow module strip → modules.
- Spacing 4-pt scale (4/8/12/16/24); cards 14 px padding, 16 px gaps.

## Elevation & Depth

Flat. Depth = surface step + hairline. `shadow-1` (1 px) on cards, `shadow-2` only for the drawer. Scrim `#000000a6` behind the drawer.

## Shapes

Radius: 6 (inputs), 8 (buttons, chips, segments), 10 (cards, video), pill (badges, indicators).

## Components

- **Card**: surface-1, hairline border, uppercase section label.
- **Buttons**: neutral (surface-2) / primary (accent) / destructive (danger, e.g. 停止). Disabled = 50 % opacity + not-allowed cursor.
- **Segmented control** (monitor off/raw/clean, presets): neutral buttons, selected = primary fill.
- **Toggle**: native checkbox with accent color, 20 px target.
- **Slider + number**: range input (accent) paired with a mono number field and unit; default shown under the help text.
- **Module (settings)**: `<details>`; summary = name + live status + per-module reset; first row = enable/mode (gate) on surface-3;
  dependent parameters indented with a 2 px hairline rail; irrelevant ones hidden; blocked ones dimmed with a ⛔ reason line.
- **Module strip**: one chip per module in data-flow order with its status; inactive modules use a hairline border and reduced opacity; click jumps to the module.
- **Badge / pill**: surface-3, pill radius. Live pill: tinted danger background, ink text, red dot.
- **Recording indicator**: solid recording red, white mono text `● 錄音中 m:ss / 10:00`, slow pulse (disabled with reduced motion).
- **Meters**: 8–10 px track on surface-2, fill in the status color, threshold marks in warning/ink-muted.
- **Timelines**: rows raw VAD (viz-raw), clean VAD (viz-clean), speech (success), decision (decision palette), state (state palette), P(group) line in primary.
- **Charts / sparklines**: 1.5 px lines, primary for the main series, ink-subtle/warning for the secondary, viz-grid hairline baseline, light area fill (13 % alpha).

## Motion

160 ms `cubic-bezier(.2,.7,.2,1)` for hover/selection; 250 ms drawer slide; pulses only for live/recording. `prefers-reduced-motion` disables animation.

## Accessibility

- WCAG 2.1 AA: all text ≥ 4.5:1 (≥ 3:1 large) on its actual background, checked in headless Chrome for both themes on the dashboard and the settings drawer (0 failures, about 370 text nodes each).
- Visible `:focus-visible` ring (2 px focus color). Status always has a text label.
- Language `zh-TW`; numbers and units are not split across lines.

## Do's and Don'ts

- Do add new colors as tokens (both themes) and read them via CSS variables, also from canvas code (`pal()` → `K`).
- Do keep color for meaning; use ink levels for hierarchy.
- Don't use brand fonts or logos of the reference products; don't hard-code hex values in components.
- Don't show camera frames in screenshots for docs; use the synthetic test pattern or no camera.
