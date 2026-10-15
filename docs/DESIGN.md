---
version: alpha
name: presence-demo-console
description: "The presence-demo console uses the PureSound web design system (the PureSound playground's tokens, base and components), pinned and vendored, so both demos look and behave as one product family. This file records only how presence-demo applies those tokens; the token values themselves live upstream."
source:
  repository: https://github.com/mcw519/puresound
  path: puresound/web/static/css
  revision: 121bf623b491cb7f18805bec65099543ecc5323f
  vendored: presence_demo/static/vendor/puresound
  license: Apache-2.0 (CSS), SIL OFL 1.1 (IBM Plex fonts)
colors:
  # Upstream token values (light / dark) at the pinned revision, listed for reference only.
  page: "#f6f6f8"
  canvas: "#ffffff"
  sunken: "#f0f1f4"
  line: "#e3e4ea"
  line-strong: "#cfd1da"
  ink: "#14151c"
  ink-2: "#5e6170"
  ink-3: "#8a8d9a"
  nav: "#010120"
  nav-3: "#1c1d3d"
  primary: "#14151c"
  accent: "#fc4c02"
  mint: "#c8f6f9"
  mint-ink: "#146168"
  periwinkle: "#bdbbff"
  magenta: "#ef2cc1"
  mic: "#6fd3db"
  good: "#1e8d5b"
  warn: "#a15c07"
  bad: "#b42318"
  dark-page: "#0d0e20"
  dark-canvas: "#17182e"
  dark-sunken: "#121326"
  dark-ink: "#ececf1"
  dark-ink-2: "#a4a6b5"
  dark-primary: "#ececf1"
  dark-mint-ink: "#9fe8ee"
  dark-good: "#3ecf8e"
  dark-warn: "#f2b54a"
  dark-bad: "#ff7a6e"
typography:
  title: { fontFamily: IBM Plex Sans, fontSize: 20px, fontWeight: 600, lineHeight: 1.2 }
  lead: { fontFamily: IBM Plex Sans, fontSize: 15px, fontWeight: 600, lineHeight: 1.3 }
  body: { fontFamily: IBM Plex Sans, fontSize: 14px, fontWeight: 400, lineHeight: 1.45 }
  ui: { fontFamily: IBM Plex Sans, fontSize: 13px, fontWeight: 500 }
  small: { fontFamily: IBM Plex Sans, fontSize: 12px, fontWeight: 400 }
  numeric: { fontFamily: IBM Plex Mono, fontSize: 12px, fontWeight: 400 }
rounded:
  control: 4px
  panel: 6px
  pill: 999px
spacing:
  grid: 4px
  panel-pad: 16px
  gap: 16px
  control-h: 32px
  control-h-small: 26px
components:
  sidebar: { backgroundColor: "{colors.nav}", textColor: "#ffffff" }
  panel: { backgroundColor: "{colors.canvas}", rounded: "{rounded.panel}", padding: "{spacing.panel-pad}" }
  button: { backgroundColor: "{colors.canvas}", textColor: "{colors.ink}", rounded: "{rounded.control}", height: "{spacing.control-h}" }
  button-primary: { backgroundColor: "{colors.primary}", textColor: "#ffffff", rounded: "{rounded.control}" }
  scope: { backgroundColor: "{colors.nav}", rounded: "{rounded.control}" }
---

# presence-demo console — DESIGN.md

Format: [DESIGN.md](https://github.com/google-labs-code/design.md) (YAML tokens + prose).

## Source of truth: the PureSound web design system

Since v0.10.2 the UI matches the **PureSound playground** (`puresound web`). Instead of keeping a second copy of
its palette, presence-demo loads the playground's own stylesheets, pinned to the same PureSound revision
as the `puresound-streaming` dependency:

| Loaded file | Upstream | Modified? |
| --- | --- | --- |
| `/static/vendor/puresound/tokens.css` | `puresound/web/static/css/tokens.css` | no |
| `/static/vendor/puresound/base.css` | `css/base.css` (IBM Plex `@font-face`, reset, focus ring, reduced motion) | font URLs made relative |
| `/static/vendor/puresound/components.css` | `css/components.css` | no |
| `/static/vendor/puresound/fonts/` | `fonts/` (IBM Plex Sans / Mono, latin subset) | no |

The PureSound Python package does not ship these web assets, so they cannot be loaded from the installed
dependency; they are vendored with their licences (`vendor/puresound/LICENSE`, `NOTICE`, `fonts/LICENSE.txt`)
and `tests/test_design_tokens.py` fails if the copy drifts from the pinned checksums or if the app's own CSS
redefines an upstream role. To update, follow `vendor/puresound/README.md`.

The app's inline `<style>` only **maps presence-specific roles onto upstream tokens** and lays out this app.
The vendored assets are public and are served without the access token; everything else keeps requiring it.

## Role mapping

| presence role | Upstream token | Use |
| --- | --- | --- |
| page / panel / sunken | `--page` / `--canvas` / `--sunken` | page ground, cards, stat tiles and gates |
| text levels | `--ink`, `--ink-2` | `--ink-3` and `--on-nav-3` are below 4.5:1 in places, so here they are used for rules and decoration only, never text |
| primary action (開始) | `--primary` / `--on-primary` | the one primary action is ink, not colour (upstream rule) |
| stop / destructive | `--bad`, `--bad-soft` | outlined danger button |
| accent | `--accent` | focus, selection (presets, module strip, monitor mode), work in progress (live pill), P(group) line, sliders. Nothing else |
| ACTIVE / MAYBE_AWAY / AWAY | `--good` / `--warn` / `--bad` | state chip fill, with `--on-status` text (white light, `#0d0e20` dark) |
| IDLE | `--mint-ink` | state chip |
| ENDED, silence | `--ink-2` | state chip, decision |
| raw audio (input) | `--mint` | waveform, timeline row (playground: mint = the recording as it came in) |
| clean audio (output) | `--periwinkle` | waveform, timeline row (playground: periwinkle = what a model returned) |
| 聲學誤判-超出範圍 | `--magenta` | playground: interferers |
| scopes | `--nav`, `--nav-line`, `--nav-3`, `--on-nav-2` | waveforms, spectrograms, timeline, sparklines and the camera placeholder sit on the always-dark ground, like the playground's audio players, in both themes |

Camera overlays are drawn on the camera image, so they use the **dark-theme** token values regardless of theme:
participant `#3ecf8e`, candidate `#f2b54a`, not visible `#7c7f92`, face ROI `#6fd3db` (`--mic`), pose ROI `#bdbbff` (`--periwinkle`).

## Layout (playground shell)

- **Sidebar** (`--nav`, 224 px): brand, navigation (監看 / 設定 / 事件紀錄 with count), and a footer with the connection
  status dot, the theme switch **自動 / 淺色 / 深色** (auto follows the OS, like the playground) and the version.
  Below 760 px it becomes a top bar.
- **Page header** (sticky, translucent page ground): title 20/600 + one-line description; actions on the right:
  recording indicator, camera/mic pill, 設定 (secondary), 開始/停止 (primary).
- **Workbench**: main column (camera view) + right column at `--inspector-w` (340 px, 420 px ≥ 1400 px) with the panels;
  one column below 1100 px.
- **Settings**: right drawer styled as the playground Inspector (canvas, hairline header, overlay shadow, backdrop).
- **事件紀錄**: floating panel, bottom right; **error banner**: dark toast style with a `--nav-bad` left rule, like playground error toasts.

## Components

All sizes come from the tokens: controls 32 px (26 px small), radius 4 px controls / 6 px panels / pill for chips,
panel padding and gaps 16 px. Buttons are sentence case, 13/500. Section titles are 15/600 sentence case (no uppercase labels).
Numbers use IBM Plex Mono with tabular figures. Module chips, presets and the monitor mode use the playground chip
selection style (accent border + accent-soft ring). Meters and countdowns are 6 px pill tracks on `--sunken`.

## Accessibility

- WCAG 2.1 AA: checked in headless Chrome on the dashboard and the settings drawer, light and dark: no text below 4.5:1.
  Legend swatches are graphics (≥ 3:1).
- Focus ring from upstream `base.css` (2 px `--focus`); `prefers-reduced-motion` honoured upstream.
- Status is never colour-only: every state, decision and counter has a text label.

## Do's and Don'ts

- Do use upstream tokens; add a presence-only role as a mapping in the inline `:root` block, never a new hex value.
- Do read colours for canvas drawing through CSS variables (`pal()` → `K`).
- Don't edit the vendored files (except the documented font URL change); update them from upstream instead.
- Don't use PureSound's logo or name as this app's brand.
- Don't show camera frames in screenshots; use the synthetic test pattern.
