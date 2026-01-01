# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Everyday people who must email or upload a PDF that is too big (job applications, forms, invoices, scans) and want it smaller in under a minute. They are not technical, often on a laptop, usually mid-errand. Confirmed with the owner.

## Product Purpose
A small web tool that shrinks PDFs. The user drops one or more files, picks how hard to squeeze embedded images (Low / Medium / High), and downloads the smaller file. Success: the file gets visibly smaller, stays openable, and the user never wonders what happened.

## Positioning
Every output is re-opened and checked (same page count) before it is offered, so a corrupt or blank PDF is never handed over. Strategies are compared honestly, including "No reduction" when nothing helps.

## Operating Context
Flask app, single page, `templates/index.html`. Uploads go to `POST /compress` (returns a job id), progress is polled from `GET /status/<id>`, results download from `/download/<id>`. Batch upload, up to 50 MB per file, results deleted after one hour. Level only affects embedded images; text-only PDFs shrink little.

## Capabilities and Constraints
- Backend API stays as it is; the redesign changes `templates/index.html` only.
- Real progress per file, per-strategy comparison table, download per strategy.
- Levels: Low (images 80%, JPEG 75), Medium (60%, JPEG 50), High (40%, JPEG 30).
- Encrypted and invalid files are rejected with a message.
- Stack: existing Flask/Jinja, plain HTML/CSS/JS. Owner's global defaults: GSAP and Lenis from a CDN; React Bits is skipped because there is no React.

## Evidence on Hand
Real behaviour only: sizes, saved percent, strategy names and timings come from the API. No testimonials, customer counts or benchmarks exist; none may be invented.

## Product Principles
1. One obvious action first: the drop zone.
2. Show what actually happened (sizes, saved %, which strategy) and never claim more.
3. Calm utility: motion explains state changes, it does not decorate.
4. Failure is a direction, not a mood: say what went wrong and what to do.

## Accessibility & Inclusion
Keyboard operable, visible focus, reduced motion respected, readable contrast, works at phone width.
