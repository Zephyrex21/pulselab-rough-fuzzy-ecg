# PulseLab Design System

## Design direction

PulseLab uses an Apple-inspired research-product aesthetic: calm surfaces, restrained colour, generous spacing, strong typographic hierarchy, rounded-but-precise components, subtle translucency and responsive layouts. It is an original visual system and does not reproduce Apple's proprietary UI assets.

## Appearance modes

The application supports `light` and `dark` themes through the root `data-theme` attribute. The selected theme is saved in browser storage under `pulselab-theme` and restored on the next visit. Theme variables are centralised in `web/styles.css`.

## Semantic tokens

| Token | Purpose |
|---|---|
| `--bg` | Application canvas |
| `--sidebar` | Navigation surface |
| `--surface` | Main card and control surface |
| `--surface-2` | Muted card and input surface |
| `--surface-3` | Hover and selected secondary surface |
| `--text` | Main body text |
| `--text-strong` | High-emphasis text and headings |
| `--muted` / `--muted-2` | Secondary and tertiary text |
| `--border` / `--border-strong` | Hairline and control borders |
| `--accent` / `--accent-strong` | Primary action and selected navigation |
| `--accent-soft` | Selected backgrounds and focus treatment |
| `--shadow` / `--shadow-float` | Card and overlay elevation |
| `--chart-line` / `--chart-grid` | Theme-aware chart presentation |
| `--input-bg` | Input surface |

## Components

- **Sidebar:** product identity, the two research engines, navigation and local-session status.
- **Top bar:** current location, active research pipeline and appearance control.
- **Page introduction:** section eyebrow, clear title, descriptive summary and live API/dataset status.
- **Cards:** consistent surface, radius, border and restrained elevation.
- **KPI tiles:** tabular values, compact labels and muted secondary surfaces.
- **Controls:** visible focus treatment, clear active state and comfortable pointer targets.
- **Tables:** sticky, quiet headers; readable row spacing; horizontal overflow on narrow screens.
- **Status labels:** distinguish trained model availability, raw dataset availability, feature exports, held-out evaluation and retrospective record replay.

## Interaction and accessibility principles

- Navigation, theme controls, and mobile drawer expose accessible names and current/expanded states.
- The theme control is a real stateful toggle and respects the saved appearance choice.
- The mobile navigation drawer closes through its backdrop or Escape key.
- Motion is restrained and reduced for users who request reduced motion.
- Keyboard focus is visible.
- Layouts collapse for tablet and phone widths without removing essential controls.
- Colour is not the only way to communicate model labels; labels remain visible in result tables.
- The browser only renders data returned by the Python API; it does not generate ECG signals or train classifiers locally.

## ECG class colours

The MIT-BIH AAMI class palette is retained for continuity: N green, S blue, V red, F amber, Q violet. These are data-visualisation accents, not clinical severity indicators.

## Product-language guidance

Describe PulseLab as an academic research prototype. Clearly distinguish the MIT-BIH AAMI arrhythmia-classification engine from the BIDMC multimodal respiratory-rate research engine. Only show BIDMC predictions when the source-hash-verified saved Python models are loaded; distinguish record-replay estimates from held-out evaluation metrics. Avoid diagnosis, clinical accuracy or validated-performance claims.
