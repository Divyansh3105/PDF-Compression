---
name: PDF Compression
description: A check-in counter for PDFs. The file steps on the scale and comes off lighter, with its tag.
colors:
  counter-navy: "#0F1A2B"
  navy-soft: "#3C4A5E"
  glass-grey: "#E6ECF2"
  counter-white: "#FFFFFF"
  rule-grey: "#C3CEDA"
  tag-orange: "#FF5A1F"
  tag-orange-hover: "#FF7440"
  scale-green: "#08744A"
  errata-red: "#B3261E"
  errata-wash: "#FDECEA"
typography:
  display:
    fontFamily: "Barlow Condensed, Barlow, system-ui, sans-serif"
    fontSize: "clamp(2rem, 5vw, 3.25rem)"
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Barlow Condensed, Barlow, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 600
    lineHeight: 1.2
  readout:
    fontFamily: "Barlow Condensed, Barlow, system-ui, sans-serif"
    fontSize: "2rem"
    fontWeight: 600
    lineHeight: 1.1
    fontFeature: "tabular-nums"
  body:
    fontFamily: "Barlow, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Barlow, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 500
    lineHeight: 1.2
rounded:
  sm: "4px"
  md: "6px"
  pill: "14px"
spacing:
  xs: "0.5rem"
  sm: "0.75rem"
  md: "1rem"
  lg: "1.5rem"
  xl: "2.5rem"
components:
  button-primary:
    backgroundColor: "{colors.tag-orange}"
    textColor: "{colors.counter-navy}"
    rounded: "{rounded.sm}"
    height: "2.75rem"
    padding: "0 1.5rem"
  button-primary-hover:
    backgroundColor: "{colors.tag-orange-hover}"
  button-line:
    backgroundColor: "{colors.counter-white}"
    textColor: "{colors.counter-navy}"
    rounded: "{rounded.sm}"
    height: "2.25rem"
    padding: "0 1rem"
  saved-tag:
    backgroundColor: "{colors.tag-orange}"
    textColor: "{colors.counter-navy}"
    rounded: "{rounded.sm}"
    height: "2.25rem"
    padding: "0 0.875rem 0 1.75rem"
  module:
    backgroundColor: "{colors.counter-white}"
    rounded: "{rounded.md}"
    padding: "1.5rem"
---

# Design System: PDF Compression

## Overview

**Creative North Star: "The Check-in Counter"**

The page behaves like a baggage counter. A cool glass-grey hall, a navy shell, white counter surfaces, and one piece of baggage-tag orange for the single thing you do next. A PDF steps on the scale, its before and after weight read out in fixed-width numerals, and it leaves with an orange tag saying how much lighter it got.

The system is calm and dense enough to be a utility. The world lends type, palette, density and one signature move; layout and controls stay standard (a real drop target, real radio inputs, real tables and links). Nothing decorates; motion only explains a state change.

**Key Characteristics:**
- Navy shell, white counter panels, glass-grey ground; orange used for the primary action and the result tag only.
- Condensed numerals in fixed-width cells, so values counting down never move the layout.
- Hairline-ruled modules with a small status tab, like a folder tab.
- Selection is a punched hole, not a coloured fill.
- Plain labels (Compress, Download); the airport vocabulary stays in the look, never in the copy.

## Colors

A cool, civic palette with one hot signal. Light ground chosen for a laptop in daylight, mid-errand.

### Primary
- **Tag Orange** (#FF5A1F): the one primary button and the "N% smaller" tag. Always carries navy text (white fails contrast on it).

### Secondary
- **Scale Green** (#08744A): the Done tab. Success only.

### Neutral
- **Counter Navy** (#0F1A2B): header, text, selected hole, Waiting and Compressing tabs.
- **Navy Soft** (#3C4A5E): secondary text, arrows.
- **Glass Grey** (#E6ECF2): page ground and the progress track.
- **Counter White** (#FFFFFF): module and panel surfaces.
- **Rule Grey** (#C3CEDA): hairlines and module borders.

### Error
- **Errata Red** (#B3261E) with **Errata Wash** (#FDECEA): failures and rejected-file notices.

### Named Rules
**The One Tag Rule.** Orange appears on at most two things per screen: the Compress button and result tags. If a third thing wants orange, it is not the primary action.
**The Navy Text On Orange Rule.** Never set white text on Tag Orange.

## Typography

**Display Font:** Barlow Condensed (with Barlow, system-ui)
**Body Font:** Barlow (with system-ui)

**Character:** A highway-sign grotesque in a condensed cut for headings and readouts, its regular width for reading. Public-signage clarity without a monospace costume.

### Hierarchy
- **Display** (600, clamp(2rem, 5vw, 3.25rem), 1.05): the page headline only.
- **Title** (600, 1.5rem, 1.2): panel and section headings.
- **Readout** (600, 2rem, 1.1, tabular numerals): before and after weights.
- **Body** (400, 1rem, 1.5): descriptions, limited to about 36rem measure.
- **Label** (500, 0.875rem): tab text and cell labels, sentence case.

### Named Rules
**The Fixed Cell Rule.** Any number that changes (sizes, percentages) uses tabular numerals in a cell with a minimum width, so updates never reflow neighbours.

## Layout

A 72rem container with 1.5rem side padding (1rem under 480px). Top row: drop zone (2fr) beside the Image quality panel (1fr), stretched to equal height. Below: the file list, one full-width module per file, newest first. Under 860px everything stacks in the same order. Spacing rhythm is 0.5, 0.75, 1, 1.5, 2.5rem; more space sits above headings than below.

## Elevation & Depth

Flat. Depth comes from tonal layering (white modules on glass grey) and a 1px Rule Grey border. There are no drop shadows. The only inset shadow is the punched-hole effect on selected radios and the tag's hole.

## Shapes

Small, practical corners: 4px on controls and the drop target, 6px on modules. The status tab is a tab (6px top corners, square bottom) fixed to the module's top-left. The saved tag has a square left end with a punched hole and a 14px rounded right end, like a luggage tag. The drop target is a 2px dashed navy rectangle.

## Components

### Buttons
- **Shape:** 4px corners, 2.75rem tall (2.25rem for line buttons).
- **Primary:** Tag Orange fill, Counter Navy text, 600 weight; lightens on hover, 1px press. Disabled is a grey fill (#D5DCE5) with #4A5667 text.
- **Line:** white with a 1.5px navy outline; inverts to navy on hover. Used for Download.
- **Focus:** 3px Tag Orange outline, 3px offset, everywhere.

### Chips (Saved tag)
Orange tag with navy text and a punched hole at its square end. Appears only on success, landing after the readout count-down.

### Cards / Containers (Modules)
White, 1px Rule Grey border, 6px radius, no shadow. A file module has a status tab above its top-left corner (Waiting and Compressing navy, Done green, Not compressed red), a 4px orange progress strip at the bottom while running, and an expandable "How each method did" table.

### Inputs / Fields
Radio rows for image quality: a 1.25rem hole on the left, name and one-line description on the right, hairline between rows. Selected = filled navy hole with an inset shadow. The drop target is a real button wrapping a hidden file input; it also accepts drag and drop.

### Navigation
None. Single page; the header carries the product name and the retention note.

### Result readout (signature)
Before and After weights side by side with an arrow. On completion the After value counts down from the original size to the compressed size (about 1.1s, exponential ease-out), then the tag and Download button settle in. Reduced motion skips the count and shows the final values.

## Do's and Don'ts

### Do:
- **Do** keep Compress as the only orange button on screen.
- **Do** show real numbers from the API; never an invented claim.
- **Do** name actions plainly: Compress, Download.
- **Do** use tabular numerals in fixed-width cells for every changing value.
- **Do** honour prefers-reduced-motion (no Lenis, no count-down, no entrances).

### Don't:
- **Don't** put white text on Tag Orange.
- **Don't** add drop shadows; depth is tonal.
- **Don't** let airport words (check-in, boarding, baggage) into UI copy.
- **Don't** use identical rounded cards as page structure; modules are ruled and tabbed.
