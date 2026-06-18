# Presentation media assets

Generated plots, videos, and GIFs for the Quarto deck will be placed in this folder.

Recommended filenames used by `maddpg_presentation_quarto.qmd`:

- `reward_comparison.png`
- `iddpg_trained.gif`
- `maddpg_trained.gif`

The deck already references these paths. If the files exist, the rendered HTML will show them automatically; if they are missing, it keeps a clean placeholder.

Quarto/reveal.js can also display GIFs directly with Markdown:

```markdown
![](assets/maddpg_trained.gif){.gif-media fig-alt="MADDPG trained rollout"}
```

For smoother playback in browsers, an MP4 export is also a good option:

```html
<video class="gif-media" src="assets/maddpg_trained.mp4" autoplay loop muted playsinline controls></video>
```
