---
name: cursor-video
description: Make videos inside Cursor with no extra login, using the agent's built-in image generation and Video Creator Prototype, a local GPU renderer that exports MP4 with hardware encoding. Use when the user asks to create, generate, or make a video, clip, reel, short, ad, trailer, slideshow, or motion piece, unless they explicitly ask for an external video model.
---

# Video Creator

Produce a video from the current Cursor session only: generate scene images with the built-in `GenerateImage` tool (namespace `cursor`), write a storyboard into the current workspace, and hand off to Video Creator Prototype (`studio.html`) for GPU rendering. The agent never runs the renderer; the user opens it in Chrome or Edge.

All paths below are relative to the current workspace root.

- Storyboards: `video-projects/<slug>/storyboard.json`
- Generated images: wherever `GenerateImage` saves them (by default the workspace `assets/` folder)
- Renderer: `studio.html` from Video Creator Prototype. Find it with a glob for `**/studio.html` in the workspace; if it is not there, the user has it in their own download location.
- Schema: the "스토리보드 형식" section of the project's `README.md`; example at `video-projects/_template/storyboard.json` when the studio folder is the workspace.

## Content rules

These apply before any image is generated. If a request needs one of these, explain why and offer an original alternative instead.

- No real, identifiable people (celebrities, politicians, private individuals) by name or likeness, unless the user supplies their own photo of themselves as a reference.
- No copyrighted characters, franchises, or artworks, and no "in the style of <living artist>" prompts. Describe the visual qualities instead (lighting, palette, medium).
- No real brand logos, trademarks, or product packaging. Use invented, generic products.
- No content meant to deceive: fake news footage, impersonation, fabricated evidence, or anything presented as a real event.
- No sexual content involving minors, no graphic violence, no hate content.
- `music` only with a file the user says they have rights to use.
- When the video depicts realistic people or events, suggest adding an "AI 생성 영상" caption on the last scene.

## Workflow

1. **Brief.** Get subject, length, and aspect ratio. Defaults: 20–30 seconds, `16:9` (`9:16` for shorts/reels), captions in the user's language. Ask only if the subject is missing.
2. **Shot list.** 5–8 scenes of 3–5 seconds. For each scene decide: what the frame shows, `motion`, optional `title` or `caption`, and transition. Use `fade-black` for chapter breaks and `crossfade` otherwise.
3. **Style block.** Write one shared style paragraph (camera, lens, lighting, color grade, film stock, era) and prepend it to every image prompt so the scenes match.
4. **Generate images.** Call `GenerateImage` once per scene:
   - `aspect_ratio` equal to the video aspect (`16:9`, `9:16`, `1:1`, `4:3`, `3:4`).
   - `filename` as `<slug>-01.png`, `<slug>-02.png`, … so names never collide across projects. The studio matches images by file name only.
   - For a recurring character or product, pass the first good image in `reference_image_paths` for later scenes.
   - Never ask for text, captions, logos, or watermarks inside images; the studio renders text.
   - Record the path each call returns.
5. **Write the storyboard** to `video-projects/<slug>/storyboard.json`. Set `image` to the returned path and `title` to `<slug>`.
6. **Hand off.** Tell the user, in their language:
   - Open `studio.html` and connect this workspace folder once with **작업 폴더 연결**. After that, press ↻ and the newest project loads automatically.
   - Press **영상 내보내기** (`Ctrl+Enter`). The MP4 is written to `renders/<slug>.mp4` inside the connected folder.
   - Without a connected folder, drag `storyboard.json` and the generated images into the studio instead.

## Storyboard skeleton

```json
{
  "title": "<slug>",
  "aspect": "16:9",
  "quality": "1080p",
  "fps": 30,
  "transition": 0.8,
  "effects": { "vignette": 0.35, "grain": 0.04, "letterbox": false },
  "look": { "contrast": 1.05, "saturation": 1.0, "warmth": 0 },
  "scenes": [
    { "image": "assets/<slug>-01.png", "duration": 4, "motion": "zoom-out", "title": "Title" },
    { "image": "assets/<slug>-02.png", "duration": 4, "motion": "pan-right", "caption": "Caption" },
    { "image": "assets/<slug>-03.png", "duration": 4, "motion": "zoom-in", "focus": [0.5, 0.35], "transitionType": "fade-black" }
  ]
}
```

Other fields: `width`/`height` instead of `quality`; `transitionType` (`crossfade`, `fade-black`, `cut`); `fadeIn`/`fadeOut` seconds; `music` (file name) and `musicVolume`; per scene `intensity` (0–0.6), `background` (color or gradient array for text-only cards).

## Choosing motion

| Shot | `motion` | Notes |
|---|---|---|
| Portrait, product, detail | `zoom-in` | Set `focus` on the subject, e.g. face at `[0.5, 0.35]` |
| Opening wide or reveal | `zoom-out` | Pairs well with a `title` |
| Landscape, skyline, horizontal action | `pan-left` / `pan-right` | Alternate direction between consecutive pans |
| Tall subject (tower, waterfall, full-body person) | `pan-up` / `pan-down` | |
| Text card or very busy frame | `static` | |

Keep `intensity` at 0.1–0.2 for a realistic camera feel; above 0.3 looks artificial.

## Realism defaults

For photoreal requests, write prompts that read like a camera description (lens, aperture, light direction, time of day, real materials, film stock), keep `saturation` near 0.95, and add `"letterbox": true` for a cinematic 2.39:1 look on `16:9`.

## Music

The studio mixes one audio track (`music`, looped, faded out at the end). The agent cannot create audio files; set `music` only when the user provides a file they have rights to use, using its file name.

## Limits to state plainly

- Motion is camera movement over still images plus transitions, not generated subject motion.
- Rendering and MP4 export happen in the user's browser; the agent cannot verify the final file.
